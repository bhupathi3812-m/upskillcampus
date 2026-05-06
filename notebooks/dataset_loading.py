# %%


# %% [markdown]
# <html><h1><center>Data Loading and Dataset validation</html>

# %% [markdown]
# <html><h3>Import Libraries</html>

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import subprocess

print(os.getcwd())
# %%


# %% [markdown]
# <html><h3>Define Dataset paths</html>

# %%
data_path='data/raw'
host=subprocess.run('pwd',shell=True,stdout=subprocess.PIPE)
print(host.stdout)
classes_path='data/raw/classes.txt'

# %%


# %% [markdown]
# <html><h3>Define Dataset paths</html>

# %%
with open(classes_path,'r') as file:
    classes=[line.strip() for line in file.readlines()]
    print("Classes:",classes)

# %%


# %% [markdown]
# <html><h3>List Images and Label files</html>

# %%
files=os.listdir(data_path)

images=[f for f in files if f.endswith('.jpeg')]
labels=[f for f in files if f.endswith('.txt') and f!="classes.txt"]

print("Total Images: ",len(images))
print("Total Labels: ",len(labels))

# %%
print(images[0],'\t',labels[0])

# %% [markdown]
# <html><h3>Validate images label pair</html>

# %%
missing_labels=[]

for img in images:
    label_file=img.replace(".jpeg",".txt")
    if label_file not in labels:
        missing_labels.append(label_file)
print("Images without Labels: ",len(missing_labels))

# %%


# %% [markdown]
# <html><h3>Load Label Data</html>

# %%
data=[]

for label in labels:
    path=os.path.join(data_path,label)

    with open(path,'r') as f:
        lines=f.readlines()

    for line in lines:
        values=line.strip().split()

        class_id=int(values[0])

        x=float(values[1])
        y=float(values[2])
        h=float(values[3])
        w=float(values[4])

        data.append([label,class_id,x,y,h,w])

# %%


# %% [markdown]
# <html><h3>Create Dataframe for data we have done</html>

# %%
df=pd.DataFrame(data,columns=[
    "file",
    "class_id",
    "x_center",
    "y_center",
    "width",
    "height"
])

df['class_name']=df['class_id'].apply(lambda x:classes[x])

df.head()

# %%
print("Total length:",len(df))
print("Class Distribution:\n")
print(df['class_name'].value_counts())



# %%


# %%


# %%



