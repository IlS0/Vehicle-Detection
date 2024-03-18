import cv2
import argparse


class Loader:
    def __init__(self):
        """
        Initializes the Loader object.

        Args:
           None

        Returns:
            None
        """
        self.src_path = ""
        self.model_path = ""
        self.src = None
        self.parser = argparse.ArgumentParser(
            prog='Vehicle-detector',
            description='Vehicle detection',
        )
        self.isVideo = False
        self.img_size = 0

    def __parse(self):
        """
        Parses and saves command line arguments.

        Args:
            None

        Returns:
            None
        """

        group = self.parser.add_mutually_exclusive_group(required=True)
        self.parser.add_argument('model_path', type=str)
        self.parser.add_argument('img_size', type=int)
        group.add_argument('-v', '--video')
        group.add_argument('-i', '--image')

        args = self.parser.parse_args()

        self.model_path = args.model_path
        self.img_size = args.img_size
        self.src_path = args.video if args.video is not None else args.image
        self.isVideo = True if args.video is not None else False

    def __call__(self):
        """
        Calls the Loader object to parse and return .

        Args:
            None

        Returns:
            tuple: (source (image/video cap), a path to model, flag (video -True or image - False), frame size)
        """
        self.__parse()

        self.src = cv2.VideoCapture(
            self.src_path) if self.isVideo else cv2.imread(self.src_path)

        return (self.src, self.model_path, self.isVideo, self.img_size)


if __name__ == "__main__":
    loader = Loader()
    loader()
