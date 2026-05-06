# %% [markdown]
# <html><h1><center>Data Cleaning and Dataset Validation</html>

# %% [markdown]
# <html><h3>Check Corrupted Images</html>

# %%
import os
import subprocess
from dataset_loading import data_path, images, labels
host=subprocess.run('pwd',shell=True,stdout=subprocess.PIPE)
print(host.stdout)
os.getcwd()

# %%
import dataset_loading
from PIL import Image
import os
bad_images=[]

for img in images:
    path=os.path.join(data_path,img)
    try:
        image=Image.open(path)
        image.verify()
    except:
        bad_images.append(image)
print("Corrupted Images:\n",len(bad_images))

# %%


# %% [markdown]
# <html><h3>Validate YOLO Label Format</html>

# %%
bad_labels=[]
for label in labels:
    path=os.path.join(data_path,label)

    with open(path,'r') as f:
        lines=f.readlines()

    for line in lines:
        values=line.strip().split()

        if len(values)!=5:
            bad_labels.append(label)
print("Invalid Label Files:",len(bad_labels))

# %%


# %% [markdown]
# <html><h3>Checking Bounding Box Values</html>

# %%
invalid_boxes=[]

for label in labels:
    path=os.path.join(data_path,label)

    with open(path,'r') as f:
        lines=f.readlines()

    for line in lines:
        values=line.strip().split()
        class_id=int(values[0])
        x=float(values[1])
        y=float(values[2])
        w=float(values[3])
        h=float(values[4])
        if not (0 <= x <=1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            invalid_boxes.append(label)
print("Inalid Bounding Boxes:",len(invalid_boxes))

# %%


# %% [markdown]
# <html><h3>Check Empty Label files</html>

# %%
empty_labels=[]

for label in labels:
    path=os.path.join(data_path,label)

    if os.path.getsize(path)==0:
        empty_labels.append(label)
print("Empty Labels:",len(empty_labels))

# %%


# %% [markdown]
# <html><h3>Total results after Cleaning</html>

# %%
print("Total Images:",len(images))
print("Total Labels:",len(labels))
print("Corrupted Images:",len(bad_images))
print("Invalid Labels:",len(bad_labels))
print("Invalid Bounding Boxes:",len(invalid_boxes))
print("Empty Labels:",len(empty_labels))

# %%
# cleaning.py

import os
from PIL import Image

def check_corrupted_images(data_path, images):
    bad_images = []
    for img in images:
        path = os.path.join(data_path, img)
        try:
            image = Image.open(path)
            image.verify()
        except:
            bad_images.append(img)
    return bad_images


def check_invalid_labels(data_path, labels):
    bad_labels = []
    for label in labels:
        path = os.path.join(data_path, label)
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            values = line.strip().split()
            if len(values) != 5:
                bad_labels.append(label)
    return bad_labels


def check_invalid_boxes(data_path, labels):
    invalid_boxes = []
    for label in labels:
        path = os.path.join(data_path, label)
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            values = line.strip().split()
            if len(values) != 5:
                continue
            x, y, w, h = map(float, values[1:])
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                invalid_boxes.append(label)
    return invalid_boxes


def check_empty_labels(data_path, labels):
    empty_labels = []
    for label in labels:
        path = os.path.join(data_path, label)
        if os.path.getsize(path) == 0:
            empty_labels.append(label)
    return empty_labels

# %%



