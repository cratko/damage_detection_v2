import os
import sys
import torch
import detectron2
from detectron2.utils.logger import setup_logger

# --- ИСПРАВЛЕНИЕ PILLOW ---
import PIL.Image

if not hasattr(PIL.Image, 'LINEAR'):
    PIL.Image.LINEAR = PIL.Image.BILINEAR

# --- НАСТРОЙКА ПУТЕЙ ---
# Мы используем установленную библиотеку для кода,
# а папку репозитория только для доступа к .yaml конфигам.
D2_REPO_PATH = os.path.abspath("detectron2_repo")

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

# Безопасный импорт PointRend
try:
    from detectron2.projects.point_rend import add_pointrend_config
except ImportError:
    # Добавляем путь к проектам, если библиотека не видит их
    sys.path.insert(0, os.path.join(D2_REPO_PATH, "projects", "PointRend"))
    from point_rend import add_pointrend_config

import dataset_utils

setup_logger()

# --- 1. РЕГИСТРАЦИЯ ДАННЫХ ---
dataset = dataset_utils.load_dataset()
cleaned_train_json = dataset_utils.clean_coco_annotations(dataset.location, 'train')
cleaned_valid_json = dataset_utils.clean_coco_annotations(dataset.location, 'valid')

TRAIN_NAME, VAL_NAME = "my_dataset_train_final", "my_dataset_valid_final"
for name in [TRAIN_NAME, VAL_NAME]:
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)

register_coco_instances(TRAIN_NAME, {}, cleaned_train_json, os.path.join(dataset.location, "train"))
register_coco_instances(VAL_NAME, {}, cleaned_valid_json, os.path.join(dataset.location, "valid"))


# --- 2. ТРЕНЕР ---
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg, True, output_folder or os.path.join(cfg.OUTPUT_DIR, "coco_eval"))


# --- 3. ЗАПУСК ОБУЧЕНИЯ ---
def start_train():
    cfg = get_cfg()

    # А) Добавляем специфичные настройки PointRend
    add_pointrend_config(cfg)

    # Б) Загружаем конфиг из файла (используем путь к вашему склонированному репо)
    # Обратите внимание на '_coco.yaml' в конце, как в вашем примере
    config_path = os.path.join(D2_REPO_PATH, "projects", "PointRend", "configs",
                               "InstanceSegmentation", "pointrend_rcnn_R_50_FPN_3x_coco.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

    cfg.merge_from_file(config_path)

    # В) Загружаем предобученные веса через встроенный протокол detectron2://
    # Это гарантирует получение правильной версии весов для PointRend
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"

    # Г) Настройки датасетов и классов
    cfg.DATASETS.TRAIN = (TRAIN_NAME,)
    cfg.DATASETS.TEST = (VAL_NAME,)

    # ВАЖНО: После merge_from_file нужно ПЕРЕОПРЕДЕЛИТЬ классы,
    # так как в _coco.yaml их 80.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2  # Классы для головы PointRend

    # Д) Параметры обучения (Windows-стабильные)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.STEPS = (2000, 2500)  # Снижаем скорость обучения ближе к концу
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.INPUT.MIN_SIZE_TRAIN = (800, 1000)  # Добавим случайный ресайз для аугментации
    cfg.TEST.EVAL_PERIOD = 500

    cfg.OUTPUT_DIR = "./output_pointrend_official"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    print("--- ЗАПУСК ОБУЧЕНИЯ POINTREND С ОФИЦИАЛЬНЫМИ ВЕСАМИ ---")
    trainer.train()


if __name__ == '__main__':
    start_train()