"""
This module implements an algorithm of object detection of the 9 different classes of vehicles. It also makes preprocessing and postprocessing of an input image.
"""
import io
import random
import logging
import time

import numpy as np
import av
import cv2
import onnxruntime


class VehicleDetector():
    """
    A class to vehicle detection. 

    Attributes:
        size (int): Size of an input image.

    Methods:
        draw_box(img): Draws bounding boxes on the input image.
        video_run(cap): Processes video frames.

    """

    def __init__(self, model_path, size) -> None:
        """
        Initializes the VehicleDetector object.

        Args:
            model_path (str): Path to a model file.
            size (tuple): Size of an input image.

        Returns:
            None
        """

        self.size = size

        self.logger = logging.getLogger(__name__)

        logging.basicConfig(level=logging.INFO)

        self.__names = ['bike', 'bus', 'car', 'construction equipment',
                        'emergency', 'motorbike', 'personal mobility', 'quad bike', 'truck']
        self.__colors = {name: [random.randint(0, 255)
                                for _ in range(3)] for name in self.__names}
        self.__session = onnxruntime.InferenceSession(
            model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.__input_names = [inp.name for inp in self.__session.get_inputs()]
        self.__output_names = [
            out.name for out in self.__session.get_outputs()]
        self.times = []
        # костыль 2
        self.__ratio = 0
        self.__dwdh = 0

    def __letterbox(self, img, new_shape, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        """
        Processes an image using letterbox method.

        Args:
            img (numpy.ndarray): Image.
            new_shape (int | tuple): Size of a new image.
            color (tuple, optional): Border filling color. Defaults to (114, 114, 114).
            auto (bool, optional): Flag for automatically adding padding.. Defaults to True.
            scaleup (bool, optional): Flag to image upscale. Defaults to True.
            stride (int, optional): Stride to padding. Defaults to 32.

        Returns:
            tuple: (processed image, computed ratio, (new width value, new height value))
        """

        # Resize and pad image while meeting stride-multiple constraints
        img_shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        ratio = min(new_shape[0] / img_shape[0], new_shape[1] / img_shape[1])
        # only scale down, do not scale up (for better val mAP)
        if not scaleup:
            ratio = min(ratio, 1.0)

        # Compute padding
        new_unpad = int(round(img_shape[1] * ratio)
                        ), int(round(img_shape[0] * ratio))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if img_shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return img, ratio, (dw, dh)

    def __forward(self, img):
        """
        Performs forward pass on the input image.

        Args:
            img (numpy.ndarray): Input BGR image.

        Returns:
            numpy.ndarray: Output of the forward pass.
        """

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_img = img.copy()

        rgb_img, self.__ratio, self.__dwdh = self.__letterbox(
            rgb_img, self.size, auto=False)
        rgb_img = rgb_img.transpose((2, 0, 1))
        rgb_img = np.expand_dims(rgb_img, 0)
        rgb_img = np.ascontiguousarray(rgb_img)
        rgb_img = rgb_img.astype(np.float32)
        rgb_img /= 255


        return self.__session.run(self.__output_names, {self.__input_names[0]: rgb_img})[0]

    def __post_process(self, output, img):
        """
        Post-processes an output of a model.

        Args:
            output (tuple): Output of a model.
            img (numpy.ndarray): Image to post-process.

        Returns:
            tuple: Processed bounding boxes and class IDs.
        """

        bboxes = []
        scores = []
        class_ids = []
        ori_images = [img.copy()]

        for batch_id, x0, y0, x1, y1, class_id, score in output:
            img = ori_images[int(batch_id)]
            box = np.array([x0, y0, x1, y1], dtype=np.float32)
            box -= np.array(self.__dwdh*2)
            box /= self.__ratio
            box = box.round().astype(np.int32).tolist()
            bboxes.append(box)

            class_id = int(class_id)
            class_ids.append(class_id)

            score = round(float(score), 3)
            scores.append(score)

        return bboxes, class_ids, scores

    def _get_yolo_out(self, img):
        '''
        class_id x1 y1 x2 y2
        '''
        bboxes, class_ids, scores = self.__post_process(self.__forward(img), img)

        h, w, _ = img.shape
            
        yolo_format = []
        for bbox, class_id in zip(bboxes, class_ids):
            x0, y0, x1, y1 = bbox
            # Normalize the bounding box coordinates
            x0, x1 = x0 / w, x1 / w
            y0, y1 = y0 / h, y1 / h
            yolo_format.append([class_id, x0, y0, x1, y1])

        return yolo_format


    def draw_box(self, img):
        """
        Draws bounding boxes on the input image.

        Args:
            img (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image with drawn bounding boxes.
        """
        start_time = time.time()
        output = self.__post_process(self.__forward(img), img)
        bboxes, class_ids, scores = output
        out_len = len(bboxes)

        res_img = img.copy()

        for idx in range(out_len):

            name = self.__names[class_ids[idx]]

            color = self.__colors[name]

            cv2.rectangle(res_img, bboxes[idx][:2], bboxes[idx][2:], color, 2)

            cv2.putText(res_img, f"{name} {scores[idx]}", (bboxes[idx][0], bboxes[idx]
                        [1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness=2)
        end_time = time.time()  # Stop measuring processing time
        process_time = end_time - start_time  # Calculate processing time
        # Logging processing time
        self.logger.info(f"Processing time: {process_time:.4f} seconds")
        self.times.append(process_time)
        # Logging detected classes
        self.logger.info(
            f"Detected classes: {', '.join(self.__names[class_id] for class_id in class_ids)}")
        return res_img

    def video_run(self, cap):
        """
        Processes video frames.

        Args:
            cap (VideoCapture): Video capture object.

        Returns:
            io.BytesIO: In-memory file containing the processed video.
        """
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if (fps == 0.0):
            fps = 30

        output_memory_file = io.BytesIO()
        # Open "in memory file" as MP4 video output
        output_f = av.open(output_memory_file, 'w', format="mp4")
        print(fps)
        # Add H.264 video stream to the MP4 container, with framerate
        stream = output_f.add_stream('h264', f"{fps}")
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            res_img = self.draw_box(frame)
            # Convert image from NumPy Array to frame.
            res_img = av.VideoFrame.from_ndarray(res_img, format='bgr24')
            packet = stream.encode(res_img)  # Encode video frame
            # "Mux" the encoded frame (add the encoded frame to MP4 file).
            output_f.mux(packet)

        # Flush the encoder
        packet = stream.encode(None)
        output_f.mux(packet)
        output_f.close()
        self.logger.info(f"Processing time mean: {np.mean(self.times):.4f} seconds")
        return output_memory_file

    def __call__(self, img):
        """
        Calls the VehicleDetector object to process an input image.

        Args:
            img (numpy.ndarray): Input RGB image.

        Returns:
            numpy.ndarray: Image with drawn bounding boxes.
        """

        return self.draw_box(img)
