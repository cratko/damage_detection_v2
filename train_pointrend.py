import os
import sys
import torch
import detectron2
from detectron2.utils.logger import setup_logger

# --- ИСПРАВЛЕНИЕ СОВМЕСТИМОСТИ PILLOW ---
# В новых версиях Pillow атрибут LINEAR был удален.
# Этот хак восстанавливает обратную совместимость для корректной работы Detectron2.
import PIL.Image

if not hasattr(PIL.Image, 'LINEAR'):
    PIL.Image.LINEAR = PIL.Image.BILINEAR

# --- НАСТРОЙКА ПУТЕЙ ---
# Путь к склонированному репозиторию Detectron2 для доступа к файлам конфигурации (.yaml)
D2_REPO_PATH = os.path.abspath("detectron2_repo")

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

# Попытка импорта PointRend (специфический проект внутри Detectron2)
try:
    from detectron2.projects.point_rend import add_pointrend_config
except ImportError:
    # Если PointRend не установлен как модуль, добавляем путь к нему вручную
    sys.path.insert(0, os.path.join(D2_REPO_PATH, "projects", "PointRend"))
    from point_rend import add_pointrend_config

import dataset_utils

# Инициализация логгера Detectron2
setup_logger()

# --- 1. ПОДГОТОВКА И РЕГИСТРАЦИЯ ДАННЫХ ---
# Загружаем метаданные датасета и очищаем аннотации COCO от возможных ошибок
dataset = dataset_utils.load_dataset()
cleaned_train_json = dataset_utils.clean_coco_annotations(dataset.location, 'train')
cleaned_valid_json = dataset_utils.clean_coco_annotations(dataset.location, 'valid')

TRAIN_NAME, VAL_NAME = "my_dataset_train_final", "my_dataset_valid_final"

# Очистка каталога перед повторной регистрацией (чтобы избежать ошибок при перезапуске)
for name in [TRAIN_NAME, VAL_NAME]:
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)

# Регистрация обучающей и валидационной выборок в формате COCO
register_coco_instances(TRAIN_NAME, {}, cleaned_train_json, os.path.join(dataset.location, "train"))
register_coco_instances(VAL_NAME, {}, cleaned_valid_json, os.path.join(dataset.location, "valid"))


# --- 2. КЛАСС ТРЕНЕРА ---
class MyTrainer(DefaultTrainer):
    """
    Расширение стандартного тренера для включения оценки COCO во время валидации.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        # Используем COCOEvaluator для получения метрик AP (Average Precision)
        return COCOEvaluator(dataset_name, cfg, True, output_folder or os.path.join(cfg.OUTPUT_DIR, "coco_eval"))


# --- 3. КОНФИГУРАЦИЯ И ЗАПУСК ОБУЧЕНИЯ ---
def start_train():
    cfg = get_cfg()

    # А) Инициализация структуры конфига под PointRend
    add_pointrend_config(cfg)

    # Б) Загрузка базовых параметров из официального YAML файла PointRend
    config_path = os.path.join(D2_REPO_PATH, "projects", "PointRend", "configs",
                               "InstanceSegmentation", "pointrend_rcnn_R_50_FPN_3x_coco.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации не найден по пути: {config_path}")

    cfg.merge_from_file(config_path)

    # В) Установка весов предобученной модели
    # Используем официальный URL-протокол detectron2 для автоматической загрузки
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"

    # Г) Привязка зарегистрированных датасетов
    cfg.DATASETS.TRAIN = (TRAIN_NAME,)
    cfg.DATASETS.TEST = (VAL_NAME,)

    # Д) Настройка количества классов
    # ВНИМАНИЕ: Для PointRend нужно менять количество классов в двух местах
    num_classes = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # Для основной головы ROI
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = num_classes  # Для головы PointRend (уточнение масок)

    # Е) Параметры процесса обучения
    # Для Windows рекомендуется NUM_WORKERS = 0 во избежание проблем с многопоточностью
    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.SOLVER.IMS_PER_BATCH = 2  # Размер батча (зависит от объема видеопамяти)
    cfg.SOLVER.BASE_LR = 0.00025  # Начальная скорость обучения
    cfg.SOLVER.MAX_ITER = 3000  # Общее количество итераций
    cfg.SOLVER.STEPS = (2000, 2500)  # Точки снижения Learning Rate (LR Decay)

    # Ж) Настройки входа и валидации
    cfg.INPUT.MIN_SIZE_TRAIN = (800, 1000)  # Многомасштабное обучение (аугментация)
    cfg.TEST.EVAL_PERIOD = 500  # Запуск валидации каждые 500 итераций

    # З) Директория для сохранения результатов
    cfg.OUTPUT_DIR = "./output_pointrend_official"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Инициализация и запуск процесса
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)  # Начинать с нуля, а не с последнего чекпоинта

    print("--- ЗАПУСК ОБУЧЕНИЯ POINTREND С ИСПОЛЬЗОВАНИЕМ ОФИЦИАЛЬНЫХ ВЕСОВ ---")
    trainer.train()


if __name__ == '__main__':
    start_train()