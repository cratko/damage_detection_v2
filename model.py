import torch
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import random
import cv2
import os
from detectron2.utils.logger import setup_logger
import torch, torchvision
import matplotlib.pyplot as plt
import dataset_utils
from calcs_v2 import calculate_and_draw_sizes

setup_logger()

import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import os

dataset = dataset_utils.load_dataset()

cleaned_train_json = dataset_utils.clean_coco_annotations(dataset.location, 'train')
cleaned_valid_json = dataset_utils.clean_coco_annotations(dataset.location, 'valid')


print("Очистка предыдущих регистраций датасетов...")

# --- НАЧАЛО ИСПРАВЛЕНИЯ ---
# Полностью удаляем датасет из ОБОИХ каталогов, чтобы избежать ошибки

# Очищаем обучающий набор
if "my_dataset_train_final" in DatasetCatalog.list():
    DatasetCatalog.remove("my_dataset_train_final")
if "my_dataset_train_final" in MetadataCatalog.list():
    MetadataCatalog.remove("my_dataset_train_final")

# Очищаем валидационный набор
if "my_dataset_valid_final" in DatasetCatalog.list():
    DatasetCatalog.remove("my_dataset_valid_final")
if "my_dataset_valid_final" in MetadataCatalog.list():
    MetadataCatalog.remove("my_dataset_valid_final")

print("Предыдущие регистрации удалены.")
# --- КОНЕЦ ИСПРАВЛЕНИЯ ---

# Пути к папкам с изображениями
train_images_dir = os.path.join(dataset.location, "train")
valid_images_dir = os.path.join(dataset.location, "valid")

# Теперь можно безопасно регистрировать заново
print("\nРегистрация очищенных датасетов...")
register_coco_instances("my_dataset_train_final", {}, cleaned_train_json, train_images_dir)
register_coco_instances("my_dataset_valid_final", {}, cleaned_valid_json, valid_images_dir)
print("✅ Регистрация успешно завершена!")

print("🔎 Загрузка обученной модели и запуск предсказания...")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

if not torch.cuda.is_available():
    print("GPU not found. Using CPU for training...")
    cfg.MODEL.DEVICE = "cpu"
else:
    print("GPU found. Using GPU for training...")

cfg.DATASETS.TRAIN = ("my_dataset_train_final",)
cfg.DATASETS.TEST = ("my_dataset_valid_final",)  # Указываем валидационный датасет для оценки

cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1500
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.TEST.EVAL_PERIOD = 500  # Запускать оценку каждые 500 итераций

# --- ВАЖНО: Устанавливаем правильное количество классов ---
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

# Указываем путь к весам нашей обученной модели
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# Устанавливаем порог уверенности
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
# Создаем предсказатель (predictor)
predictor = DefaultPredictor(cfg)

# --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---

# 1. Берем случайное изображение из нашего валидационного набора
dataset_dicts_valid = DatasetCatalog.get("my_dataset_valid_final")
d = random.choice(dataset_dicts_valid)

# 2. Загружаем изображение, ИСПОЛЬЗУЯ ПРАВИЛЬНЫЙ ПУТЬ из словаря 'd'
image_path = d["file_name"]
im = cv2.imread(image_path)

print(f"Выбрано случайное изображение для предсказания: {image_path}")

# --- КОНЕЦ ИСПРАВЛЕНИЯ ---

# Делаем предсказание
outputs = predictor(im)

# Визуализируем результат
"""
metadata = MetadataCatalog.get("my_dataset_train_final") # Используем метаданные от обучающего набора
v = Visualizer(im[:, :, ::-1],
               metadata=metadata,
               scale=0.7,
               instance_mode=ColorMode.IMAGE_BW
)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
print("\nРезультат предсказания модели:")

img = out.get_image()[:, :, ::-1]  # BGR->RGB
plt.imshow(img)
plt.axis('off')
plt.show()
"""

final_image_with_sizes = calculate_and_draw_sizes(outputs, im)

# Отображаем финальный результат
print("\nРезультат с вычисленными размерами:")
img_rgb = cv2.cvtColor(final_image_with_sizes, cv2.COLOR_BGR2RGB) # Конвертируем BGR в RGB для Matplotlib
plt.figure(figsize=(12, 12)) # Увеличим размер для наглядности
plt.imshow(img_rgb)
plt.axis('off')
plt.show()