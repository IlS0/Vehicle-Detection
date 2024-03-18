import io
import random
import numpy as np
import av
import cv2
import onnxruntime


class Detector():
    """_summary_
    
    """
    def __init__(self, model_path, size) -> None:
        """
        Initialize the MyDetector object.

        Args:
            model_path (str): Path to the model file.
            size (tuple): Size of the input image.

        Returns:
            None
        """

        self.__size = size
        self.__mean = [0.485, 0.456, 0.406]
        self.__std = [0.229, 0.224, 0.225]
        self.__names = ['bike', 'bus', 'car', 'construction equipment',
                        'emergency', 'motorbike', 'personal mobility', 'quad bike', 'truck']
        self.__colors = {name: [random.randint(0, 255)
                                for _ in range(3)] for name in self.__names}
        self.__session = onnxruntime.InferenceSession(
            model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.__input_names = [inp.name for inp in self.__session.get_inputs()]
        self.__output_names = [
            out.name for out in self.__session.get_outputs()]

        # костыль 2
        self.__ratio = 0
        self.__dwdh = 0

    def letterbox(self, img, new_shape, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        """_summary_

        Args:
            img (_type_): _description_
            new_shape (_type_): _description_
            color (tuple, optional): _description_. Defaults to (114, 114, 114).
            auto (bool, optional): _description_. Defaults to True.
            scaleup (bool, optional): _description_. Defaults to True.
            stride (int, optional): _description_. Defaults to 32.

        Returns:
            _type_: _description_
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

    def forward(self, img):
        """
        Perform forward pass on the input image.

        Args:
            bgr_img (numpy.ndarray): Input RGB image.

        Returns:
            numpy.ndarray: Output of the forward pass.
        """

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_img = img.copy()

        rgb_img, self.__ratio, self.__dwdh = self.letterbox(
            rgb_img, self.__size, auto=False)
        rgb_img = rgb_img.transpose((2, 0, 1))
        rgb_img = np.expand_dims(rgb_img, 0)
        rgb_img = np.ascontiguousarray(rgb_img)
        rgb_img = rgb_img.astype(np.float32)
        rgb_img /= 255

        rgb_img[0][0] = (rgb_img[0][0] - self.__mean[0]) / self.__std[0]
        rgb_img[0][1] = (rgb_img[0][1] - self.__mean[1]) / self.__std[1]
        rgb_img[0][2] = (rgb_img[0][2] - self.__mean[2]) / self.__std[2]

        return self.__session.run(self.__output_names, {self.__input_names[0]: rgb_img})[0]

    def post_process(self, output, img):
        """
        Post-process the output of the model.

        Args:
            output (tuple): Output of the model.

        Returns:
            tuple: Processed bounding boxes and class IDs.
        """

        bboxes = []
        scores = []
        class_ids = []
        ori_images = [img.copy()]

        for batch_id, x0, y0, x1, y1, class_id, score in output:
            img = ori_images[int(batch_id)]
            box = np.array([x0, y0, x1, y1])
            box -= np.array(self.__dwdh*2)
            box /= self.__ratio
            box = box.round().astype(np.int32).tolist()
            bboxes.append(box)

            class_id = int(class_id)
            class_ids.append(class_id)

            score = round(float(score), 3)
            scores.append(score)

        return bboxes, class_ids, scores

    def draw_box(self, img):
        """
        Draw bounding boxes on the input image.

        Args:
            img (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image with bounding boxes drawn.
        """

        output = self.post_process(self.forward(img), img)
        bboxes, class_ids, scores = output
        out_len = len(bboxes)
        
        res_img = img.copy()

        for idx in range(out_len):

            name = self.__names[class_ids[idx]]

            color = self.__colors[name]
            
            cv2.rectangle(res_img, bboxes[idx][:2], bboxes[idx][2:], color, 2)

            cv2.putText(res_img, f"{name} {scores[idx]}", (bboxes[idx][0], bboxes[idx]
                        [1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness=2)

        return res_img

    def video_run(self, cap):
        """
        Process video frames and draw bounding boxes.

        Args:
            cap: Video capture object.

        Returns:
            io.BytesIO: In-memory file containing the processed video.
        """
        fps = cap.get(cv2.CAP_PROP_FPS)
        if (fps == 0.0):
            fps = 30

        output_memory_file = io.BytesIO()
        # Open "in memory file" as MP4 video output
        output_f = av.open(output_memory_file, 'w', format="mp4")
        
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
        return output_memory_file

    def __call__(self, img):
        """
        Call the MyDetector object to process an input image.

        Args:
            rgb_img (numpy.ndarray): Input RGB image.

        Returns:
            tuple: Processed bounding boxes and class IDs.
        """

        return self.draw_box(img)
