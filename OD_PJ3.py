#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('pip', 'install ultralytics')
get_ipython().run_line_magic('pip', 'install scikit-learn')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import numpy as np
import cv2
import shutil
import yaml
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO
from glob import glob
from tqdm import tqdm
from IPython.display import clear_output
from sklearn.model_selection import train_test_split


# In[ ]:


SEED = 42
BATCH_SIZE = 8
MODEL = "/content/drive/MyDrive/v2"


# In[ ]:


if os.path.exists("../data/yolo"):
    shutil.rmtree("../data/yolo")

if not os.path.exists("../data/yolo/train"):
    os.makedirs("../data/yolo/train")

if not os.path.exists("../data/yolo/valid"):
    os.makedirs("../data/yolo/valid")

if not os.path.exists("../data/yolo/test"):
    os.makedirs("../data/yolo/test")

if not os.path.exists("../results"):
    os.makedirs("../results")


# In[ ]:


import os
import shutil

base_dir = "/content"

data_dir = os.path.join(base_dir, "data/yolo")
results_dir = os.path.join(base_dir, "results")

if os.path.exists(data_dir):
    shutil.rmtree(data_dir)

os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "valid"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)


os.makedirs(results_dir, exist_ok=True)


# In[ ]:


def make_yolo_dataset(image_paths, txt_paths, type="train"):
    for image_path, txt_path in tqdm(zip(image_paths, txt_paths if not type == "test" else image_paths), total=len(image_paths)):
        source_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_height, image_width, _ = source_image.shape

        target_image_path = f"/content/drive/MyDrive/yolov8_v1/data/yolo/{type}/{os.path.basename(image_path)}"
        cv2.imwrite(target_image_path, source_image)

        if type == "test":
            continue

        with open(txt_path, "r") as reader:
            yolo_labels = []
            for line in reader.readlines():
                line = list(map(float, line.strip().split(" ")))
                class_name = int(line[0])
                x_min, y_min = float(min(line[5], line[7])), float(min(line[6], line[8]))
                x_max, y_max = float(max(line[1], line[3])), float(max(line[2], line[4]))
                x, y = float(((x_min + x_max) / 2) / image_width), float(((y_min + y_max) / 2) / image_height)
                w, h = abs(x_max - x_min) / image_width, abs(y_max - y_min) / image_height
                yolo_labels.append(f"{class_name} {x} {y} {w} {h}")

        target_label_txt = f"/content/drive/MyDrive/yolov8_v1/data/yolo/{type}/{os.path.basename(txt_path)}"
        with open(target_label_txt, "w") as writer:
            for yolo_label in yolo_labels:
                writer.write(f"{yolo_label}\n")


# In[ ]:


image_paths = sorted(glob("/content/drive/MyDrive/project_2/open/train/*.png"))
txt_paths = sorted(glob("/content/drive/MyDrive/project_2/open/train/*.txt"))

train_images_paths, valid_images_paths, train_txt_paths, valid_txt_paths = train_test_split(image_paths, txt_paths, test_size=0.1, random_state=SEED)

make_yolo_dataset(train_images_paths, train_txt_paths, "train")
make_yolo_dataset(valid_images_paths, valid_txt_paths, "valid")
make_yolo_dataset(sorted(glob("/content/drive/MyDrive/project_2/open/test/*.png")), None, "test")


# In[ ]:


with open("/content/drive/MyDrive/프로젝트 2조/open/classes.txt", "r") as reader:
    lines = reader.readlines()
    classes = [line.strip().split(",")[1] for line in lines]

yaml_data = {
              "names": classes,
              "nc": len(classes),
              "path": "/content/drive/MyDrive/프로젝트 2조/open",
              "train": "prep_train",
              "val": "prep_valid",
              "test": "prep_test"
            }

with open("/content/drive/MyDrive/프로젝트 2조/open/yolo/custom.yaml", "w") as writer:
    yaml.dump(yaml_data, writer)


# In[ ]:


#model = YOLO(f"{MODEL}/train/weights/last.pt")
#resume='/content/drive/MyDrive/v2/train/weights/best.pt'
model = YOLO("yolov8x")
results = model.train(
    data="/content/drive/MyDrive/프로젝트 2조/open/yolo/custom.yaml",
    imgsz=(1024, 1024),
    epochs=100,
    batch=BATCH_SIZE,
    patience=5,
    workers=16,
    device=0,
    exist_ok=True,
    project=f"{MODEL}",
    name="train",
    seed=SEED,
    pretrained=False,
    resume=False,
    optimizer="Adam",
    lr0=1e-3,
    augment=True,
    val=True,
    cache=True,
    )


# In[ ]:


def get_test_image_paths(test_image_paths):
    for i in range(0, len(test_image_paths), BATCH_SIZE):
        yield test_image_paths[i:i+BATCH_SIZE]


# In[ ]:


model = YOLO("/content/drive/MyDrive/v2/train/weights/best.pt")
test_image_paths = glob("/content/drive/MyDrive/프로젝트 2조/open/prep_test/*.png")
for i, image in tqdm(enumerate(get_test_image_paths(test_image_paths)), total=int(len(test_image_paths)/BATCH_SIZE)):
    model.predict(image, imgsz=(1024, 1024), iou=0.2, conf=0.5, save_conf=True, save=False, save_txt=True, project=f"{MODEL}", name="predict",
                  exist_ok=True, device=0, augment=True, verbose=False)
    if i % 5 == 0:
        clear_output(wait=True)


# In[ ]:


print("Test image paths:", test_image_paths)
print("Number of test images:", len(test_image_paths))
print("Batch size:", BATCH_SIZE)


# In[ ]:


def yolo_to_labelme(line, image_width, image_height, txt_file_name):
    file_name = txt_file_name.split("/")[-1].replace(".txt", ".png")
    class_id, x, y, width, height, confidence = [float(temp) for temp in line.split()]

    x_min = int((x - width / 2) * image_width)
    x_max = int((x + width / 2) * image_width)
    y_min = int((y - height / 2) * image_height)
    y_max = int((y + height / 2) * image_height)

    return file_name, int(class_id), confidence, x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max


# In[ ]:


infer_txts = glob(f"{MODEL}/predict/labels/*.txt")

results = []
for infer_txt in tqdm(infer_txts):
    base_file_name = infer_txt.split("/")[-1].split(".")[0]
    imgage_height, imgage_width = cv2.imread(f"/content/drive/MyDrive/프로젝트 2조/open/test/{base_file_name}.png").shape[:2]
    with open(infer_txt, "r") as reader:
        lines = reader.readlines()
        for line in lines:
            results.append(yolo_to_labelme(line, imgage_width, imgage_height, infer_txt))

df_submission = pd.DataFrame(data=results, columns=["file_name", "class_id", "confidence", "point1_x", "point1_y", "point2_x", "point2_y", "point3_x", "point3_y", "point4_x", "point4_y"])
df_submission.to_csv(f"{MODEL}.csv", index=False)

