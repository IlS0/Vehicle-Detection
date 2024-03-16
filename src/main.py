from loader import Loader
from detector import Detector 


class Main():
    def __init__(self):
        self.loader = Loader()
        self.src, self.model_path, self.isVideo, self.img_size = self.loader()
        self.detector = Detector(self.model_path, self.img_size)


    def __save_detect_res(self, res):
        pass

    def __call__(self):
        if self.isVideo: 
            res = self.detector.video_run(self.src)
            #сохраняем файл из памяти (возможно нужно по имени)
            res.write() #возможно нужен не дефолтный аргумент
        else:
            res = self.detector(self.src)
            bboxes, class_ids = res
            res_img = cv2.rectangle(bboxes)
            res_img = cv2.putText(' '.join(class_ids))
            cv2.imwrite("default name",res_img)


main = Main()
main()




