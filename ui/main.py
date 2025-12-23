import os
import sys
import cv2
import numpy as np
import io
import torch
import PIL.Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from sklearn.decomposition import PCA

# --- Detectron2 ---
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.projects.point_rend import add_pointrend_config

# --- 0. ИСПРАВЛЕНИЕ PILLOW (для совместимости с тензорами) ---
if not hasattr(PIL.Image, 'LINEAR'):
    PIL.Image.LINEAR = PIL.Image.BILINEAR

setup_logger()

# --- 1. НАСТРОЙКА ПУТЕЙ ---
UI_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(UI_DIR, "templates")
STATIC_DIR = os.path.join(UI_DIR, "static")
MODEL_PATH = os.path.join(UI_DIR, "model_final.pth")

# Путь к склонированному репозиторию detectron2 для доступа к конфигам PointRend
# Если папка лежит в корне проекта, путь будет таким:
D2_REPO_PATH = os.path.abspath(os.path.join(UI_DIR, "..", "detectron2_repo"))

if not os.path.exists(MODEL_PATH):
    print(f"ВНИМАНИЕ: Файл модели не найден: {MODEL_PATH}")


# --- 2. КЛАСС АНАЛИЗАТОРА (PCA + ГЕОМЕТРИЯ) ---
class DamageAnalyzer:
    def __init__(self, pixels_per_cm=None):
        self.pixels_per_cm = pixels_per_cm
        self.class_map = {"damage": 0, "digit": 1}

    def _get_robust_scale(self, instances):
        classes = instances.pred_classes.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        digit_idx = np.where(classes == self.class_map["digit"])[0]

        if len(digit_idx) < 2:
            return None

        points = []
        for i in digit_idx:
            x1, y1, x2, y2 = boxes[i]
            points.append([(x1 + x2) / 2, (y1 + y2) / 2])
        points = np.array(points)

        pca = PCA(n_components=1)
        projected_points = pca.fit_transform(points)
        projected_points.sort(axis=0)
        distances = np.diff(projected_points, axis=0).flatten()

        if len(distances) == 0: return None
        median_dist = np.median(distances)
        valid_distances = distances[(distances > median_dist * 0.5) & (distances < median_dist * 1.5)]

        return np.mean(valid_distances) if len(valid_distances) > 0 else median_dist

    def process_image(self, outputs, image):
        vis_img = image.copy()
        instances = outputs["instances"].to("cpu")
        self.pixels_per_cm = self._get_robust_scale(instances)

        classes = instances.pred_classes.numpy()
        masks = instances.pred_masks.numpy()
        damage_idx = np.where(classes == self.class_map["damage"])[0]

        for idx in damage_idx:
            mask = masks[idx].astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

            if not contours: continue
            cnt = max(contours, key=cv2.contourArea)
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(int)
            w_px, h_px = rect[1]
            long_px, short_px = max(w_px, h_px), min(w_px, h_px)

            cv2.drawContours(vis_img, [box], 0, (0, 255, 0), 2)

            if self.pixels_per_cm:
                l_cm = long_px / self.pixels_per_cm
                s_cm = short_px / self.pixels_per_cm
                text = f"{l_cm:.1f}x{s_cm:.1f} cm"
            else:
                text = f"{int(long_px)}x{int(short_px)} px"

            self._draw_styled_text(vis_img, text, box[0])
        return vis_img

    def _draw_styled_text(self, img, text, pos):
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (pos[0], pos[1] - h - 5), (pos[0] + w, pos[1] + 5), (0, 0, 0), -1)
        cv2.putText(img, text, (pos[0], pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


# --- 3. ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ И МОДЕЛИ ---
app = FastAPI(title="Carsharing Damage Detection System")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def get_pointrend_predictor():
    cfg = get_cfg()
    add_pointrend_config(cfg)

    # Путь к конфигу PointRend внутри репозитория
    config_path = os.path.join(D2_REPO_PATH, "projects", "PointRend", "configs",
                               "InstanceSegmentation", "pointrend_rcnn_R_50_FPN_3x_coco.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Не найден конфиг PointRend: {config_path}")

    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return DefaultPredictor(cfg)


print("Загрузка PointRend...")
predictor = get_pointrend_predictor()
analyzer = DamageAnalyzer()
print("Сервис готов к работе.")


# --- 4. ЭНДПОИНТЫ ---
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Файл должен быть изображением")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "Ошибка декодирования изображения")

    # Инференс
    outputs = predictor(img)
    # Постпроцессинг и отрисовка размеров
    result_img = analyzer.process_image(outputs, img)

    _, buffer = cv2.imencode(".jpg", result_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return StreamingResponse(io.BytesIO(buffer), media_type="image/jpeg")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h2>Ошибка: templates/index.html не найден</h2>")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "model": "PointRend R50",
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)