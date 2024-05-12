import numpy

from os import listdir
from os.path import isfile, join
from collections import defaultdict

from detector import VehicleDetector
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader


# cd Vehicle-Detection/
# export PYTHONPATH='/home/forssh/workspace/Vehicle-Detection/src'

class Data(Dataset):
    def __init__(self, annotations_dir, img_dir):
        self.img_labels = sorted([join(annotations_dir, f) for f in listdir(
            annotations_dir) if isfile(join(annotations_dir, f))])
        self.img_dir = img_dir
        self.img_names = sorted([f for f in listdir(
            img_dir) if isfile(join(img_dir, f))])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = join(self.img_dir, self.img_names[idx])
        image = read_image(img_path)
        label_file = self.img_labels[idx]
        labels = open(label_file, encoding="utf-8").readlines()
        label_classes, label_boxes = [], []
        for label in labels:
            label = label.split(" ")
            label_classes.append(int(label[0]))
            label_boxes.append([float(cord) for cord in label[1:]])

        return image, label_boxes, label_classes


# Вычисление IoU по 2-м bbox
def compute_iou(box1, box2):
    xA = max(box1[1], box2[1])
    yA = max(box1[2], box2[2])
    xB = min(box1[3], box2[3])
    yB = min(box1[4], box2[4])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[3] - box1[1]) * (box1[4] - box1[2])
    box2Area = (box2[3] - box2[1]) * (box2[4] - box2[2])
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou


def convert_format(box):
    '''
    Convert from [class_id, center_x, center_y, width, height] to [class_id, x1, y1, x2, y2]
    '''
    class_id, center_x, center_y, width, height = box
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    return [class_id, x1, y1, x2, y2]


detector = VehicleDetector(
    "/home/forssh/workspace/runs/train/exp60/weights/best.onnx", 640)


test_data = Data("/home/forssh/workspace/Vehicle detection.v34i.yolov7pytorch/valid/labels",
                 "/home/forssh/workspace/Vehicle detection.v34i.yolov7pytorch/valid/images")

test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)


# img, box, clss = next(iter(test_dataloader))
# print(box[1], clss[1])

# Инициализация словарей для подсчета метрик для каждого класса
TP = defaultdict(int)
FP = defaultdict(int)
FN = defaultdict(int)
TN = defaultdict(int)

classes = ['bike', 'bus', 'car', 'construction equipment',
                        'emergency', 'motorbike', 'personal mobility', 'quad bike', 'truck']
 
# Список для хранения IoU для каждого bb
ious = []

# Список изображений без предсказаний
no_predict = []

# Прогон каждого изображения через модель
for idx, (img, box, clss) in enumerate(test_dataloader):
    if box == []:
        print("Image with empty labels: ", test_data.img_names[idx])
        continue
    
    # DataLoader format to Numpy array
    img_numpyed = img.squeeze(0).permute(1, 2, 0).numpy()

    predicts = detector._get_yolo_out(img_numpyed)

    true_boxes = [convert_format([class_id.item()] + [bbox_item.item() for bbox_item in bbox]) for bbox, class_id in zip(box, clss)]

    # Вычисление метрик
    for true_box in true_boxes:
        max_iou = 0
        
        # На случай, если модель ничего не предскажет (background)
        if not predicts:
            max_class_id_pred = -1
            no_predict.append(test_data.img_names[idx])
            
        # Ищем такую пару bb, чтобы IoU было максимальным
        for predict in predicts:
            iou = compute_iou(true_box, predict)
            class_id_true, class_id_pred = true_box[0], predict[0]
            
            if iou >= max_iou:
                max_iou, max_class_id_pred = iou, class_id_pred
                
        if class_id_true == max_class_id_pred:
            TP[class_id_true] += 1
        else:
            FN[class_id_true] += 1
            FP[max_class_id_pred] += 1

        ious.append(max_iou)
    
if no_predict != []:
    print("\nМодель ничего не увидела на следующих изображениях:")
    for item in no_predict[:5]:
        print(item)

total = sum(TP.values()) + sum(FN.values())
TN = {i: total - TP[i] - FP[i] - FN[i] for i in range(len(classes))}

# Вычисление precision и recall для каждого класса
precision = {classes[i]: TP[i] / (TP[i] + FP[i]) for i in range(len(classes)) if (TP[i] + FP[i]) != 0}
recall = {classes[i]: TP[i] / (TP[i] + FN[i]) for i in range(len(classes)) if (TP[i] + FN[i]) != 0}
specificity = {classes[i]: TN[i] / (TN[i] + FP[i]) for i in range(len(classes)) if (TN[i] + FP[i]) != 0}
 
print("\nTP: ", sorted(TP.items()))
print("FP: ", sorted(FP.items()))
print("FN: ", sorted(FN.items()))
print("TN: ", sorted(TN.items()))

print("\nPrecision")
for class_name, p in precision.items():
    print(f"  {class_name}: {p:.3f}")

print(f"Среднее значение: {sum(precision.values())/9:.3f}")

print("\nRecall")
for class_name, r in recall.items():
    print(f"  {class_name}: {r:.3f}")

print(f"Среднее значение: {sum(recall.values())/9:.3f}")

print("\nSpecificity")
for class_name, r in specificity.items():
    print(f"  {class_name}: {r:.3f}")

print(f"Среднее значение: {sum(specificity.values())/9:.3f}")


print("\nПроцент ложных срабатываний")
for class_name, r in specificity.items():
    print(f"  {class_name}: {(1-r):.3%}")
    
print(f"Среднее значение: {1-sum(specificity.values())/9:.3%}")    
    
print(f"\nСреднее значение IoU: {numpy.mean(ious):.3f}")