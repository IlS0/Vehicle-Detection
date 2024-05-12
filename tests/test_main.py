import unittest
from unittest.mock import patch
import numpy as np

import os
import cv2

from main import Main


path_video = "videos/test_video.mp4"
path_img = "images/test_img.jpg"
path_model = "best.onnx"


class TestMain(unittest.TestCase):
    # @patch('sys.argv', ['test_main.py', model_path, '640', '-v', path_video])
    # def test_call_with_video(self):
    #    self.main = Main()
    #    self.main()
    #    self.assertTrue(os.path.exists("results/video"))
    # Check if a file with a specific pattern has been created
    #    self.assertTrue(
    #        any(fname.startswith('v_predict_') and fname.endswith('.mp4') for fname in os.listdir("results/video"))
    #    )
    #    filename = os.listdir("results/video/")
    #    file = os.path.join("results/video/", filename[0])
    # video = cv2.imread(file)
    # for image in video:
    #     comparison_result = image == self.main.res
    #     false_indices = np.where(comparison_result == False)
    #     if (np.array_equal(image, self.main.res) == False) :
    #         print("Индексы ложных элементов:")
    #         print(false_indices)
    #     self.assertTrue(np.array_equal(image, self.main.res))

    @patch('sys.argv', ['test_main.py', path_model, '640', '-i', path_img])
    def test_call_with_image(self):
        self.main = Main()
        self.main()
        self.assertTrue(os.path.exists("results/images"))
        # Check if a file with a specific pattern has been created
        self.assertTrue(
            any(fname.startswith('i_predict_') and fname.endswith('.png')
                for fname in os.listdir("results/images"))
        )
        filename = os.listdir("results/images/")
        file = os.path.join("results/images/", filename[0])
        image = cv2.imread(file)
        comparison_result = image == self.main.res
        false_indices = np.where(comparison_result is False)
        if (np.array_equal(image, self.main.res) is False):
            print("Индексы ложных элементов:")
            print(false_indices)
        self.assertTrue(np.array_equal(image, self.main.res))


if __name__ == '__main__':
    unittest.main()
