# %% [markdown]
# <html><h1><center>Exploratory Data Analysis(EDA) and Visualizing</html>

# %% [markdown]
# <html><h3>Class Distribution</html>

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from dataset_loading import data_path, images, labels,df
import os

from dataset_loading import data_path, images, labels
from cleaning import (
    check_corrupted_images,
    check_invalid_labels,
    check_invalid_boxes,
    check_empty_labels
)

bad_images = check_corrupted_images(data_path, images)
bad_labels = check_invalid_labels(data_path, labels)
invalid_boxes = check_invalid_boxes(data_path, labels)
empty_labels = check_empty_labels(data_path, labels)

print("Total Images:",len(images))
print("Total Labels:",len(labels))
print("Corrupted Images:",len(bad_images))
print("Invalid Labels:",len(bad_labels))
print("Invalid Bounding Boxes:",len(invalid_boxes))
print("Empty Labels:",len(empty_labels))

figure, axes = plt.subplots(2, 3, figsize=(14, 8),constrained_layout=True)


sns.countplot(data=df,x='class_name',ax=axes[0,0])

axes[0,0].set_title("Crop vs Weed Distribution")
axes[0,0].set_xlabel("Class")
axes[0,0].set_ylabel("Count")


# %%


# %% [markdown]
# <html><h3>Bounding Box Width Distribution</html>

# %%

axes[0,1].hist(df['width'],bins=30)
axes[0,1].set_title("Bounding Box width Distribution")
axes[0,1].set_xlabel("Width")
axes[0,1].set_ylabel("Count")


# %%


# %% [markdown]
# <html><h3>Bounding Box Height Distribution</html>

# %%

axes[0,2].hist(df['height'],bins=30)
axes[0,2].set_title("Bounding Box Height Distribution")
axes[0,2].set_xlabel("Height")
axes[0,2].set_ylabel("Count")

# %%


# %% [markdown]
# <html><h3>Bounding Box center Distribution</html>

# %%
sns.scatterplot(x=df['x_center'],y=df['y_center'],alpha=0.6,ax=axes[1,0])
axes[1,0].set_xlabel("X Center")
axes[1,0].set_ylabel("Y Center")
axes[1,0].set_title("Bounding Box Center Distribution")

# %%
# %% [markdown]
# <html><h3>Bounding Box center Distribution</html>

# %%
import cv2
import random

# %%
#select a random image ok
random_img=random.choice(images)
img_path=os.path.join(data_path,random_img)
label_path=os.path.join(data_path,random_img.replace(".jpeg",".txt"))

img=cv2.imread(img_path)
h,w,_=img.shape
h,w

# %%
#now draw bouding boxes
with open(label_path) as f:
    lines=f.readlines()

for line in  lines:
    
    c,x,y,bw,bh=map(float,line.split())

    x1=int((x-bw/2)*w)
    y1=int((y-bh/2)*h)

    x2=int((x+bw/2)*w)
    y2=int((y+bh/2)*h)

    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)


axes[1,1].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
axes[1,1].axis("off")
axes[1,2].axis('off')
plt.subplots_adjust(hspace=0.8, wspace=0.3)
plt.tight_layout()
plt.show()

# %%



