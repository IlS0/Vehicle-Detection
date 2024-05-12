import unittest
import cv2
import numpy as np

from detector import VehicleDetector

path_img = "images/test_img.jpg"
path_model = "best.onnx"


class TestVehicleDetector(unittest.TestCase):

    def setUp(self):
        # Устанавливаем путь к модели и размер изображения для тестирования
        self.path_model = path_model
        self.size = (640, 640)  # Пример размера изображения
        # Создаем экземпляр объекта VehicleDetector для тестирования
        self.detector = VehicleDetector(self.path_model, self.size)

    # Метод для тестирования функции __letterbox
    def test_letterbox(self):
        # Загружаем изображение для тестирования
        img = cv2.imread(path_img)
        # Вызываем функцию __letterbox
        processed_img, ratio, _ = self.detector._VehicleDetector__letterbox(
            img, self.size)
        # Проверяем типы возвращаемых значений
        self.assertIsInstance(processed_img, np.ndarray)
        self.assertIsInstance(ratio, float)
        # Проверяем размер обработанного изображения
        self.assertEqual(processed_img.shape[1], self.size[1])

    # Метод для тестирования прямого прохода изображения через модель
    def test_forward_pass(self):
        img = cv2.imread(path_img)
        output = self.detector._VehicleDetector__forward(img)
        self.assertIsInstance(output, np.ndarray)
        self.assertTrue(len(output) > 0)

    # Проверяем, что метод __post_process возвращает корректные типы данных для боксов, ID классов и оценок
    def test_output_types(self):
        output = [(0, 10, 20, 50, 60, 1, 0.95)]
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bboxes, class_ids, scores = self.detector._VehicleDetector__post_process(
            output, img)

        # Проверяем, что возвращаемые значения являются списками
        self.assertIsInstance(bboxes, list)
        self.assertIsInstance(class_ids, list)
        self.assertIsInstance(scores, list)

    # Проверяем, что метод __post_process возвращает не пустые списки боксов, ID классов и оценок
    def test_non_empty_lists(self):
        output = [(0, 10, 20, 50, 60, 1, 0.95)]D
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bboxes, class_ids, scores = self.detector._VehicleDetector__post_process(
            output, img)

        # Проверяем, что списки не пустые
        self.assertTrue(len(bboxes) > 0)
        self.assertTrue(len(class_ids) > 0)
        self.assertTrue(len(scores) > 0)

    # Проверяем, что длина списков боксов, ID классов и оценок соответствует количеству объектов в выводе
    def test_list_lengths(self):
        output = [(0, 10, 20, 50, 60, 1, 0.95)]
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bboxes, class_ids, scores = self.detector._VehicleDetector__post_process(
            output, img)

        # Проверяем, что длина списков соответствует количеству объектов в выводе
        self.assertEqual(len(bboxes), len(output))
        self.assertEqual(len(class_ids), len(output))
        self.assertEqual(len(scores), len(output))

    # Проверяем, что каждый элемент в списке боксов содержит 4 числа (координаты x, y верхнего левого угла и x, y нижнего правого угла)
    def test_bbox_structure(self):
        output = [(0, 10, 20, 50, 60, 1, 0.95)]
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bboxes, _, _ = self.detector._VehicleDetector__post_process(
            output, img)

        # Проверяем, что каждый элемент в списке bboxes содержит 4 числа
        for bbox in bboxes:
            self.assertIsInstance(bbox, list)
            self.assertEqual(len(bbox), 4)
            for coord in bbox:
                self.assertIsInstance(coord, int)

    # Проверяем, что каждый элемент в списке ID классов является целым числом
    def test_class_id_types(self):
        output = [(0, 10, 20, 50, 60, 1, 0.95)]
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, class_ids, _ = self.detector._VehicleDetector__post_process(
            output, img)

        # Проверяем, что каждый элемент в списке class_ids является целым числом
        for class_id in class_ids:
            self.assertIsInstance(class_id, int)

    # Проверяем, что каждый элемент в списке оценок является числом с плавающей запятой
    def test_score_types(self):
        output = [(0, 10, 20, 50, 60, 1, 0.95)]
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, _, scores = self.detector._VehicleDetector__post_process(
            output, img)

        # Проверяем, что каждый элемент в списке scores является числом с плавающей запятой
        for score in scores:
            self.assertIsInstance(score, float)

    # Метод для тестирования отрисовки рамок (box)
    def test_draw_box(self):
        img = cv2.imread(path_img)
        result_img = self.detector.draw_box(img)
        self.assertIsInstance(result_img, np.ndarray)

    # Add more tests as needed


if __name__ == '__main__':
    unittest.main()
