import torch, detectron2
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.projects import point_rend

# Ваши утилиты
import dataset_utils
from calcs_v2 import calculate_and_draw_sizes

setup_logger()

# 1. Загрузка и подготовка данных
dataset = dataset_utils.load_dataset()
cleaned_valid_json = dataset_utils.clean_coco_annotations(dataset.location, 'valid')
valid_images_dir = os.path.join(dataset.location, "valid")

VAL_NAME = "comparison_dataset_val"
if VAL_NAME in DatasetCatalog.list():
    DatasetCatalog.remove(VAL_NAME)
    MetadataCatalog.remove(VAL_NAME)

register_coco_instances(VAL_NAME, {}, cleaned_valid_json, valid_images_dir)
metadata = MetadataCatalog.get(VAL_NAME)

# --- 2. НАСТРОЙКА MASK R-CNN (STANDARD) ---
print("⚙️ Загрузка Standard Mask R-CNN...")
cfg_std = get_cfg()
cfg_std.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg_std.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg_std.MODEL.WEIGHTS = "./output_mask_rcnn_final_50/model_final.pth" # Путь к вашей обычной модели
cfg_std.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor_std = DefaultPredictor(cfg_std)

# --- 3. НАСТРОЙКА POINTREND ---
print("⚙️ Загрузка PointRend...")
cfg_pr = get_cfg()
point_rend.add_pointrend_config(cfg_pr) # Используем ваше проверенное название

# Укажите путь к вашему репозиторию для загрузки конфига
D2_REPO_PATH = os.path.abspath("detectron2_repo")
cfg_pr.merge_from_file(os.path.join(D2_REPO_PATH, "projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"))
cfg_pr.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg_pr.MODEL.POINT_HEAD.NUM_CLASSES = 2
cfg_pr.MODEL.WEIGHTS = "./output_pointrend_official/model_final.pth" # Путь к вашему PointRend
cfg_pr.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor_pr = DefaultPredictor(cfg_pr)

# --- 4. ЗАПУСК СРАВНЕНИЯ ---
MY_IMAGE_PATH = "test.jpg"
im = ""
if not os.path.exists(MY_IMAGE_PATH):
    print(f"❌ Ошибка: Файл {MY_IMAGE_PATH} не найден!")
else:
    im = cv2.imread(MY_IMAGE_PATH)
    print(f"🖼️ Тестируем на вашем фото: {MY_IMAGE_PATH}")

# Инференс
outputs_std = predictor_std(im)
outputs_pr = predictor_pr(im)

# Визуализация масок (Raw Masks)
def get_vis_img(img, outputs):
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
    return v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()

res_std_raw = get_vis_img(im, outputs_std)
res_pr_raw = get_vis_img(im, outputs_pr)

# Визуализация с вашими расчетами (Sizes)
res_std_sizes = calculate_and_draw_sizes(outputs_std, im)
res_pr_sizes = calculate_and_draw_sizes(outputs_pr, im)

# --- 5. ВЫВОД РЕЗУЛЬТАТОВ ---
fig, axs = plt.subplots(2, 2, figsize=(20, 15))

# Стандартная модель
axs[0, 0].imshow(res_std_raw)
axs[0, 0].set_title("Standard Mask R-CNN: Raw Masks")
axs[0, 0].axis('off')

axs[1, 0].imshow(cv2.cvtColor(res_std_sizes, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title("Standard Mask R-CNN: Measurements")
axs[1, 0].axis('off')

# PointRend
axs[0, 1].imshow(res_pr_raw)
axs[0, 1].set_title("PointRend: Raw Masks")
axs[0, 1].axis('off')

axs[1, 1].imshow(cv2.cvtColor(res_pr_sizes, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title("PointRend: Measurements")
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()