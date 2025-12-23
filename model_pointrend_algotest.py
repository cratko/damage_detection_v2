import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Detectron2 imports
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.projects import point_rend

import dataset_utils

setup_logger()


# ==========================================
# 1. СТАРЫЙ АЛГОРИТМ (Сортировка по X/Y)
# ==========================================
def old_calculate_sizes(outputs, image):
    vis_img = image.copy()
    instances = outputs["instances"].to("cpu")
    if len(instances) == 0: return vis_img

    classes = instances.pred_classes.numpy()
    boxes = instances.pred_boxes.tensor.numpy()
    masks = instances.pred_masks.numpy()

    # Расчет масштаба (Старый метод)
    digit_idx = np.where(classes == 1)[0]
    scale = None
    if len(digit_idx) >= 2:
        pts = []
        for i in digit_idx:
            x1, y1, x2, y2 = boxes[i]
            pts.append([(x1 + x2) / 2, (y1 + y2) / 2])
        pts = np.array(pts)

        # Сортировка просто по X (старый способ)
        sorted_pts = pts[pts[:, 0].argsort()]
        dists = [np.linalg.norm(sorted_pts[i] - sorted_pts[i + 1]) for i in range(len(sorted_pts) - 1)]
        scale = np.median(dists) if dists else None

    # Отрисовка повреждений
    for idx in np.where(classes == 0)[0]:
        mask = masks[idx].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(vis_img, [box], 0, (0, 0, 255), 2)  # Красный бокс

        if scale:
            l_cm = max(rect[1]) / scale
            cv2.putText(vis_img, f"Old: {l_cm:.1f} cm", (box[0][0], box[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis_img


# ==========================================
# 2. НОВЫЙ АЛГОРИТМ (PCA + Морфология)
# ==========================================
class NewDamageAnalyzer:
    def process(self, outputs, image):
        vis_img = image.copy()
        instances = outputs["instances"].to("cpu")
        if len(instances) == 0: return vis_img

        classes = instances.pred_classes.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy()

        # Масштаб через PCA
        digit_idx = np.where(classes == 1)[0]
        scale = None
        if len(digit_idx) >= 2:
            pts = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes[digit_idx]])
            pca = PCA(n_components=1)
            projected = pca.fit_transform(pts)
            projected.sort(axis=0)
            dists = np.diff(projected, axis=0).flatten()
            scale = np.median(dists)

        # Повреждения с очисткой
        for idx in np.where(classes == 0)[0]:
            mask = masks[idx].astype(np.uint8)
            # Очистка и сглаживание
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            if not contours: continue
            cnt = max(contours, key=cv2.contourArea)
            # Аппроксимация
            cnt = cv2.approxPolyDP(cnt, 0.002 * cv2.arcLength(cnt, True), True)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(vis_img, [box], 0, (0, 255, 0), 2)  # Зеленый бокс

            if scale:
                l_cm = max(rect[1]) / scale
                cv2.putText(vis_img, f"New: {l_cm:.1f} cm", (box[0][0], box[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return vis_img


# ==========================================
# 3. НАСТРОЙКА И ЗАПУСК
# ==========================================

# Загрузка PointRend (путь должен быть верным)
cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
D2_REPO_PATH = os.path.abspath("detectron2_repo")
cfg.merge_from_file(
    os.path.join(D2_REPO_PATH, "projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2
cfg.MODEL.WEIGHTS = "./output_pointrend_official/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

# Тестовое фото
im = cv2.imread("CUSTOM_3.jpg")
outputs = predictor(im)

# Сравнение
old_res = old_calculate_sizes(outputs, im)
new_res = NewDamageAnalyzer().process(outputs, im)

# Вывод
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(cv2.cvtColor(old_res, cv2.COLOR_BGR2RGB))
ax[0].set_title("PointRend + Старый алгоритм (X-Sort)", fontsize=14)
ax[0].axis('off')

ax[1].imshow(cv2.cvtColor(new_res, cv2.COLOR_BGR2RGB))
ax[1].set_title("PointRend + Новый алгоритм (PCA + Smoothing)", fontsize=14)
ax[1].axis('off')

plt.tight_layout()
plt.show()