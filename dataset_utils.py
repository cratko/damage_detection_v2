from roboflow import Roboflow
from termcolor import colored
import json
import os

def load_dataset():
    # Используем ваш API ключ и данные проекта для загрузки датасета
    print(colored("[DATASET] Загрузка датасета из Roboflow...", "green"))

    rf = Roboflow(api_key="o2gHYWzqWDg5w62b0AXO")
    project = rf.workspace("cd-nijiu").project("cardamages-v3-fz5ds")
    version = project.version(2)
    dataset = version.download("coco-segmentation")

    print(f"\n[DATASET] Датасет успешно загружен и находится в папке: {dataset.location}")

    return dataset

def clean_coco_annotations(dataset_path, split='train'):
    """
    Функция для очистки файла аннотаций COCO от Roboflow.
    Удаляет категорию с id=0 и переназначает ID остальным.
    """
    original_path = os.path.join(dataset_path, split, "_annotations.coco.json")
    cleaned_path = os.path.join(dataset_path, split, "_annotations.coco_cleaned.json")

    print(f"[DATASET] Очистка файла: {original_path}...")

    with open(original_path, 'r') as f:
        coco_data = json.load(f)

    new_categories = []
    id_map = {}
    new_id_counter = 0
    for category in coco_data['categories']:
        if category['id'] != 0: # Игнорируем категорию с id=0
            id_map[category['id']] = new_id_counter
            category['id'] = new_id_counter
            new_categories.append(category)
            new_id_counter += 1

    new_annotations = []
    for annotation in coco_data['annotations']:
        old_category_id = annotation.get('category_id')
        if old_category_id in id_map:
            annotation['category_id'] = id_map[old_category_id]
            new_annotations.append(annotation)

    coco_data['categories'] = new_categories
    coco_data['annotations'] = new_annotations

    with open(cleaned_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"[DATASET] ✅ Создан очищенный файл: {cleaned_path}")
    return cleaned_path

# dataset.location - это переменная из вашего кода загрузки Roboflow
# Очищаем обучающий и валидационный наборы
# cleaned_train_json = clean_coco_annotations(dataset.location, 'train')
# cleaned_valid_json = clean_coco_annotations(dataset.location, 'valid')