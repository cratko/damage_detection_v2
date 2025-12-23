import numpy as np
import cv2
from sklearn.decomposition import PCA


class DamageAnalyzer:
    def __init__(self, pixels_per_cm=None):
        self.pixels_per_cm = pixels_per_cm
        self.class_map = {"damage": 0, "digit": 1}

    def _get_robust_scale(self, instances):
        """
        Улучшенный расчет масштаба через PCA (устойчив к наклонам под любым углом).
        """
        classes = instances.pred_classes.numpy()
        boxes = instances.pred_boxes.tensor.numpy()

        digit_idx = np.where(classes == self.class_map["digit"])[0]
        if len(digit_idx) < 2:
            return None

        # Собираем центры всех цифр
        points = []
        for i in digit_idx:
            x1, y1, x2, y2 = boxes[i]
            points.append([(x1 + x2) / 2, (y1 + y2) / 2])
        points = np.array(points)

        # 1. Используем PCA для нахождения главной оси линейки
        pca = PCA(n_components=1)
        # Проецируем точки на одну линию
        projected_points = pca.fit_transform(points)
        # Сортируем координаты вдоль этой линии
        projected_points.sort(axis=0)

        # 2. Вычисляем расстояния между соседними проекциями
        distances = np.diff(projected_points, axis=0).flatten()

        # Фильтруем аномалии (слишком мелкие или слишком крупные скачки)
        # Например, если медиана 40px, отсекаем всё что <20 и >100
        if len(distances) == 0: return None

        median_dist = np.median(distances)
        valid_distances = distances[(distances > median_dist * 0.5) & (distances < median_dist * 1.5)]

        if len(valid_distances) == 0:
            return median_dist  # fallback к сырой медиане

        return np.mean(valid_distances)

    def process_image(self, outputs, image):
        vis_img = image.copy()
        instances = outputs["instances"].to("cpu")

        # 1. Считаем масштаб
        self.pixels_per_cm = self._get_robust_scale(instances)

        # 2. Обработка повреждений
        classes = instances.pred_classes.numpy()
        masks = instances.pred_masks.numpy()
        damage_idx = np.where(classes == self.class_map["damage"])[0]

        for idx in damage_idx:
            # Морфологическая очистка маски (убираем шум PointRend)
            mask = masks[idx].astype(np.uint8)
            kernel = np.one((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            if not contours: continue

            cnt = max(contours, key=cv2.contourArea)

            # Аппроксимация для более гладкого замера (убирает микро-зубцы)
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)

            # Геометрия
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(int)

            # Расчет физических размеров
            w_px, h_px = rect[1]
            long_px, short_px = max(w_px, h_px), min(w_px, h_px)

            # Отрисовка
            cv2.drawContours(vis_img, [box], 0, (0, 255, 0), 2)

            if self.pixels_per_cm:
                l_cm = long_px / self.pixels_per_cm
                s_cm = short_px / self.pixels_per_cm
                text = f"{l_cm:.1f}x{s_cm:.1f} cm"
            else:
                text = f"{int(long_px)}x{int(short_px)} px"

            # Визуальный вывод
            self._draw_styled_text(vis_img, text, box[0])

        return vis_img

    def _draw_styled_text(self, img, text, pos):
        # Рисуем текст с подложкой (профессиональный вид)
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (pos[0], pos[1] - h - 5), (pos[0] + w, pos[1] + 5), (0, 0, 0), -1)
        cv2.putText(img, text, (pos[0], pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Использование:
# analyzer = DamageAnalyzer()
# result_img = analyzer.process_image(outputs, original_image)