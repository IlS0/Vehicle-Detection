import argparse
import unittest
from unittest.mock import patch
import sys
import os
sys.path.append(r'D:\NSU\Vehicle-Detection-main\src')
from loader import Loader

path_img = 'D:\\NSU\\Vehicle-Detection-main\\asserts\\img\\image1.png'
path_video = 'D:\\NSU\\Vehicle-Detection-main\\asserts\\video\\video1.mp4'

class TestLoader(unittest.TestCase):

    # корректно обрабатывает аргументы командной строки   
    @patch('sys.argv', ['script_name.py', 'model_path', '640', '-i', 'image_path'])
    def test_parse_arguments_image(self):
        loader = Loader()
        loader.parse()
        self.assertEqual(loader.model_path, 'model_path')
        self.assertEqual(loader.img_size, 640)
        self.assertEqual(loader.inp_path, 'image_path')
        self.assertFalse(loader.isVideo)
    
    # Проверяем, что изображение успешно загружено
    @patch('sys.argv', ['script_name.py', 'model_path', '640', '-i', path_img])
    def test_image_loading(self):
        loader = Loader()
        image, _, _, _ = loader()
        self.assertIsNotNone(image)
    
    # Проверяем, что видео успешно загружено
    @patch('sys.argv', ['script_name.py', 'model_path', '640', '-v', path_video])
    def test_video_loading(self):
        loader = Loader()
        video_path = os.path.abspath(path_video)
        video, _, _, _ = loader()
        self.assertIsNotNone(video)

    # Проверяем правильность инициализации атрибутов объекта класса Loader при создании нового экземпляра. 
    def test_attributes_initialization(self):
        loader = Loader()
        self.assertEqual(loader.model_path, "")
        self.assertEqual(loader.inp_path, "")
        self.assertIsNone(loader.input)
        self.assertIsInstance(loader.parser, argparse.ArgumentParser)
        self.assertFalse(loader.isVideo)
        self.assertEqual(loader.img_size, 0)

    
    # проверка обновления параметров
    # def test_input_path_update(self):
    #     loader = Loader()
    #     args_img = ['-i', path_img, 'model_path', '640']
    #     args_video = ['-v', path_video, 'model_path', '640']
    #     with patch('sys.argv', ['script.py'] + args_img):
    #         loader.parse()
    #         self.assertEqual(loader.inp_path, path_img)
    #     with patch('sys.argv', ['script.py'] + args_video):
    #         loader.parse()
    #         self.assertEqual(loader.inp_path, path_video)

if __name__ == '__main__':
    unittest.main()
