from loader import Loader
from detector import Detector
import cv2

from datetime import datetime
from os import makedirs


class Main():
    def __init__(self):
        self.loader = Loader()
        self.src, self.model_path, self.isVideo, self.img_size = self.loader()
        self.detector = Detector(self.model_path, self.img_size)

    def __call__(self):
        makedirs("results", exist_ok=True)

        if self.isVideo:
            makedirs("results/video", exist_ok=True)

            res = self.detector.video_run(self.src)

            with open(f"results/video/v_predict_{datetime.now()}.mp4", "wb") as file:
                file.write(res.getbuffer())
        else:
            makedirs("results/images", exist_ok=True)

            res = self.detector(self.src)

            cv2.imwrite(f"results/images/i_predict_{datetime.now()}.png", res)


main = Main()
main()
