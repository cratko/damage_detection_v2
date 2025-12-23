import detectron2
from detectron2.utils.logger import setup_logger
import torch, torchvision

import dataset_utils

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


# Обучение

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator

# Ваш кастомный Тренер для оценки
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def start_train():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Устройство
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Датасеты
    cfg.DATASETS.TRAIN = ("my_dataset_train_final",)
    cfg.DATASETS.TEST = ("my_dataset_valid_final",)

    # --- СИНХРОНИЗАЦИЯ С POINTREND ---
    cfg.SOLVER.MAX_ITER = 3000  # Увеличиваем до уровня PointRend
    cfg.SOLVER.STEPS = (2100, 2700)  # Снижаем LR на 70% и 90% пути
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.IMS_PER_BATCH = 2

    cfg.DATALOADER.NUM_WORKERS = 0  # Для стабильности на Windows

    # Настройки разрешения (аналогично PointRend)
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MIN_SIZE_TEST = 800

    # Количество классов
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    # Оценка
    cfg.TEST.EVAL_PERIOD = 500
    cfg.OUTPUT_DIR = "./output_mask_rcnn_final_50"  # Новая папка!

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    print("🚀 Запуск финального обучения Mask R-CNN на 50 изображениях...")
    trainer.train()

if __name__ == '__main__':
    # --- Эта строка теперь является точкой входа в вашу программу ---
    start_train()