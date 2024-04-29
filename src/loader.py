"""
"""
import argparse

import cv2


class Loader:
    """
    A class to parse and save command line arguments.

    Attributes:
        model_path (str): Path to a model file.
        inp_path (str): Path to an input.
        input (MatLike | VideoCapture): Input.
        parser (ArgumentParser): Command line parser.
        isVideo (bool): Input type flag.
        img_size (int): Size of an input image.
    """

    def __init__(self):
        """
        Initializes the Loader object.

        Args:
           None

        Returns:
            None
        """

        self.model_path = ""
        self.inp_path = ""
        self.input = None  # возможно закрыть
        self.parser = argparse.ArgumentParser(
            prog='Vehicle-detector',
            description='Vehicle detection',
        )
        self.isVideo = False
        self.img_size = 0

    def parse(self):
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
        self.inp_path = args.video if args.video is not None else args.image
        self.isVideo = True if args.video is not None else False

    def __call__(self):
        """
        Calls the Loader object to parse and return the arguments.

        Args:
            None

        Returns:
            tuple: (source (image/video cap), path to model, flag (video -True or image - False), image size)
        """
        self.parse()

        self.input = cv2.VideoCapture(
            self.inp_path) if self.isVideo else cv2.imread(self.inp_path)

        return (self.input, self.model_path, self.isVideo, self.img_size)


if __name__ == "__main__":
    loader = Loader()
    loader()
