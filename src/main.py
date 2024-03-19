from os import makedirs
from datetime import datetime, date
#import datetime

from loader import Loader
from detector import VehicleDetector
import cv2


class Main():
    """
    A class for program modules management.

    Attributes:
        loader (Loader): Command line arguments loader.
        input (MatLike | VideoCapture):  Input.
        model_path (str): Path to a model file.
        isVideo(bool): Input type flag.
        img_size(int): Size of an input image.
        detector (Detector): Object detector.
    """

    def __init__(self):
        """
        Initializes the Main object.

        Args:
           None

        Returns:
            None
        """
        self.loader = Loader()
        self.input, self.model_path, self.isVideo, self.img_size = self.loader()
        self.detector = VehicleDetector(self.model_path, self.img_size)

    def __call__(self):
        """
        Calls the Main object to run program modules.

        Args:
           None

        Returns:
            None
        """
        makedirs("results", exist_ok=True)

        if self.isVideo:
            makedirs("results/video", exist_ok=True)

            res = self.detector.video_run(self.input)

            with open(f"results/video/v_predict_{datetime.now().strftime('%d-%m-%Y %H:%M:%S:%f')}.mp4", "wb") as file:
                file.write(res.getbuffer())
        else:
            makedirs("results/images", exist_ok=True)

            res = self.detector(self.input)

            cv2.imwrite(f"results/images/i_predict_{datetime.now().strftime('%d-%m-%Y %H:%M:%S:%f')}.png", res)


if __name__ == "__main__":
    main = Main()
    main()
