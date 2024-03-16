import io
import numpy as np
import av
import cv2
import onnxruntime


class Detector():
    MEAN = 0
    STD = 0.5

    def __init__(self, model_path, size) -> None:
        """
        Initialize the MyDetector object.

        Args:
            model_path (str): Path to the model file.
            size (tuple): Size of the input image.

        Returns:
            None
        """
        self._session = onnxruntime.InferenceSession(model_path)
        self.input_name = 'input'
        self.output_names = 'fc'
        self._size = size
    
    def forward(self, rgb_img):
        """
        Perform forward pass on the input image.

        Args:
            rgb_img (numpy.ndarray): Input RGB image.

        Returns:
            numpy.ndarray: Output of the forward pass.
        """

        print(type(self._size))

        input_image = cv2.dnn.blobFromImage(rgb_img, size=(self._size,self._size), swapRB=True)
        input_image = (input_image - self.MEAN) / self.STD
        return self._session.run(self.output_names, {self.input_name: input_image})
    
    def post_process(self, output):
        """
        Post-process the output of the model.

        Args:
            output (tuple): Output of the model.

        Returns:
            tuple: Processed bounding boxes and class IDs.
        """
        bboxes, scores, class_ids = output
        bboxes = bboxes.nms()
        return bboxes, class_ids[scores>0.5]

    def draw_box(self, img):
        """
        Draw bounding boxes on the input image.

        Args:
            img (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image with bounding boxes drawn.
        """
        output = self.post_process(self.forward(img))
        bboxes, class_ids = output
        res_img = cv2.rectangle(bboxes)
        res_img = cv2.putText(' '.join(class_ids))
        return res_img

    def video_run(self, cap):
        """
        Process video frames and draw bounding boxes.

        Args:
            cap: Video capture object.

        Returns:
            io.BytesIO: In-memory file containing the processed video.
        """
        output_memory_file = io.BytesIO()
        output_f = av.open(output_memory_file, 'w', format="mp4")  # Open "in memory file" as MP4 video output
        stream = output_f.add_stream('h264',"60" )  # Add H.264 video stream to the MP4 container, with framerate = fps.str(fps)
        ret = True
        while ret:
            ret, frame = cap.read()
            res_img = self.draw_box(frame)
            res_img = av.VideoFrame.from_ndarray(res_img, format='bgr24') # Convert image from NumPy Array to frame.
            packet = stream.encode(frame)  # Encode video frame
            output_f.mux(packet)  # "Mux" the encoded frame (add the encoded frame to MP4 file).
        # Flush the encoder
        packet = stream.encode(None)
        output_f.mux(packet)
        output_f.close()
        return output_memory_file

    def __call__(self, rgb_img):
        """
        Call the MyDetector object to process an input image.

        Args:
            rgb_img (numpy.ndarray): Input RGB image.

        Returns:
            tuple: Processed bounding boxes and class IDs.
        """


        return self.post_process(self.forward(rgb_img))
