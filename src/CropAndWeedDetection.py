# %%


# %% [markdown]
# <html><h1><center>Dataset Preparation for model Training</html>

# %%


# %%
import os
import random
import shutil
from sklearn.model_selection import train_test_split
from dataset_loading import data_path, images, labels
# %%
#define path
data_path="data/raw"
dataset_path="dataset"

# %% [markdown]
# <html><h3>Create Folders

# %%
folders=[
    'dataset/images/train',
    'dataset/images/val',
    'dataset/labels/train',
    'dataset/labels/val'
]
for folder in folders:
    os.makedirs(folder,exist_ok=True)    

# %% [markdown]
# <html><h3>Get Images

# %% [markdown]
# images=[f for f in os.listdir(data_path) if f.endswith(".jpeg")]
# len(images)

# %% [markdown]
# <html><h3>Train Validation Split

# %%
train_imgs,val_imgs=train_test_split(images,test_size=0.2,random_state=42)
len(train_imgs),len(val_imgs)

# %% [markdown]
# <html><h3>Copy Images and Lablels

# %% [markdown]
# <html><h4>Train Data

# %%

for img in train_imgs:
    label=img.replace(".jpeg",".txt")

    shutil.copy(
        os.path.join(data_path,img),
        "dataset/images/train"
    )
    shutil.copy(
        os.path.join(data_path,label),
        "dataset/labels/train"
    )
    

# %% [markdown]
# <html><h4>Val Data

# %%
for img in val_imgs:
    label=img.replace(".jpeg",".txt")

    shutil.copy(
        os.path.join(data_path,img),
        "dataset/images/val"
    )
    shutil.copy(
        os.path.join(data_path,label),
        "dataset/labels/val"
    )


# %%
#Data.yml
classes=['crop','weed']

with open("dataset/data.yml",'w') as f:
    f.write("train: dataset/images/train\n")
    f.write("val: dataset/images/val\n\n")

    f.write(f"nc: {len(classes)}\n")
    f.write("names: "+str(classes))

# %%


# %% [markdown]
# <html><h1><center>Model Training</html>

# %%
#install YOLO
#!/home/guestt/ml-env/bin/python -m pip install ultralytics

# %% [markdown]
# <html><h3>Import YOLO</html>

# %%
from ultralytics import YOLO

# %% [markdown]
# <html><h3>Load Pretrained Model</html>

# %%
model=YOLO('yolov8n.pt')

# %% [markdown]
# <html><h3>Train Model</html>

# %%
import subprocess
print('path:',subprocess.run('pwd',shell=True,stdout=subprocess.PIPE).stdout)
model.train(
    data='data.yml',
    imgsz=512,
    epochs=50,
    batch=16,
    project="models"
)

# %%


# %% [markdown]
# <html><h1><center>Prediction and Inference</html>

# %% [markdown]
# <html><h3>Load Trained Model</html>

# %%
from ultralytics import YOLO

model1=YOLO("runs/detect/train10/weights/best.pt")

# %% [markdown]
# <html><h3>Predict On a single Image</html>

# %%
results=model1.predict(
    source="dataset/images/val",
    imgsz=512,
    conf=0.25,
    save=True
)

# %%
r=model.predict(
    source="dataset/images/val/agri_0_8926.jpeg",
    imgsz=512,
    conf=0.25,
    show=True
)
plt.imshow(r[0].plot())
plt.axis("off")
plt.show()

# %%
r=model.predict(
    source="dataset/images/val/agri_0_14.jpeg",
    imgsz=512,
    conf=0.25,
    show=True
)
plt.imshow(r[0].plot())
plt.axis("off")
plt.show()

# %% [markdown]
# <html><h1><center>Model Evaluation and Metrics</html>

# %%
metrics=model.val(data="data.yml")
metrics

# %%



