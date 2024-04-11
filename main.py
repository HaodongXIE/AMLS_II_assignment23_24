#import libraries
import os
import os.path as osp
import copy

import time
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import models
from torchvision.utils import make_grid
from torchinfo import summary
import torchvision.transforms.functional as F

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import random
import json

import polars as pl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as mpatches
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
pl.Config().set_tbl_rows(50)
pl.Config().set_tbl_cols(-1)
pl.Config().set_fmt_str_lengths(100)

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

seed_everything(42)

ROOT_DIR = 'C://Users//11757//Desktop//5//Datasets'
TRAIN_IMAGES_DIR = osp.join(ROOT_DIR, 'train_images')
train_df = pl.read_csv(os.path.join(ROOT_DIR, 'train.csv'))

# Create image paths
train_df = train_df.with_columns(
    path=pl.concat_str(pl.lit(f'{TRAIN_IMAGES_DIR}/'), pl.col('image_id'))
)

#create labe number to disease map
with open("C://Users//11757//Desktop//5//Datasets/label_num_to_disease_map.json", "r") as f:
    labelmap = json.load(f)
    labelmap = {int(k): v for k, v in labelmap.items()}

print(json.dumps(labelmap, indent=4))



#check existence
train_df = train_df.with_columns(
    is_exists=pl.col('path').map_elements(lambda x: osp.exists(x))
)

n_not_exists = train_df.filter(pl.col('is_exists') == False).shape[0]
assert n_not_exists == 0, print(f'There are {n_not_exists} non-exists files')

n_images = train_df.filter(pl.col('is_exists') == True).shape[0]
print(f'Total images: {n_images}')

print(train_df.head(20))





#Train/test split
val_size = 0.2
test_size = 0.2

val_split = n_images - int(n_images * (test_size + val_size))
test_split = n_images - int(n_images * test_size)

val_df = train_df[val_split:test_split]
test_df = train_df[test_split:]
train_df = train_df[:val_split]

n_train = len(train_df)
n_val = len(val_df)
n_test = len(test_df)

print('Splitted dataset:')
print(f'\t- Training set: {n_train}')
print(f'\t- Validation set: {n_val}')
print(f'\t- Testing set: {n_test}')



# Number of samples by class
plt.figure(figsize=(10,6))
ax = sns.countplot(x="label", data=train_df.to_pandas())
ax = ax.bar_label(ax.containers[0])

plt.title('Number of samples by class')
plt.show()



#Set hyper parameters
WIDTH = 512
HEIGHT = 512
NUM_CLASSES = 5
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {DEVICE}')

#define preprocessing
train_transforms = albumentations.Compose([
    
    albumentations.RandomResizedCrop(WIDTH, HEIGHT),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.ShiftScaleRotate(p=0.5),
    albumentations.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            ),
    albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5),
    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    
])

test_transforms = albumentations.Compose([
    albumentations.CenterCrop(WIDTH, HEIGHT, p=1.0),
    albumentations.Resize(WIDTH, HEIGHT),
    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


#customized dataset
class CassavaDataset(Dataset):
    
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        img = Image.open(self.df['path'][index])
        img = np.array(img)
        label = torch.tensor(self.df['label'][index], dtype=torch.long)
        
        if self.transform:
            return self.transform(image=img)['image'], label
        else:
            return img, label
            
            
#create dataloader
train_dataset = CassavaDataset(train_df, transform=train_transforms)
val_dataset = CassavaDataset(val_df, transform=test_transforms)
test_dataset = CassavaDataset(test_df, transform=test_transforms)

train_dl = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_dataset, BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

#define functions to get ResNet50
def get_resnet_model():
    
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    for params in model.parameters():
        params.requires_grad = False
        
    in_feat = model.fc.in_features
        
    model.fc = nn.Sequential(
          nn.Linear(in_feat, 256),
          nn.ReLU(),
          nn.Dropout(p=0.3),
          nn.Linear(256, NUM_CLASSES))
    
    model = model.to(DEVICE)
    
    return model

#get the model
model = get_resnet_model()

summary(model, input_size=(BATCH_SIZE, 3, WIDTH, HEIGHT))


#define the traning function
def train(model, num_epochs, train_dl, valid_dl):
    
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_valid_loss = np.inf
    
    for epoch in range(num_epochs):
        
        model.train()
        
        batch_num = 0
        
        for x_batch, y_batch in tqdm(train_dl):
            
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            batch_num += 1
            #if (batch_num % 100 == 0):
                #print(f'Batch number: {batch_num}')
            
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().item()
        
        
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)
        
        scheduler.step()
        
        model.eval()
        
        with torch.no_grad():
            
            for x_batch, y_batch in valid_dl:
                
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().item()
                
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
        
        if accuracy_hist_valid[epoch] > best_acc:
            best_acc = accuracy_hist_valid[epoch]
            best_model_wts = copy.deepcopy(model.state_dict())
        
        print(f'Epoch {epoch+1}:   Train accuracy: {accuracy_hist_train[epoch]:.4f}    Validation accuracy: {accuracy_hist_valid[epoch]:.4f} ')
    
    
        if loss_hist_valid[epoch] < min_valid_loss:
            counter = 0
        else:
            counter += 1
    
        if counter >= patience:
            break
    
    
    model.load_state_dict(best_model_wts)
    
    history = {}
    history['loss_hist_train'] = loss_hist_train
    history['loss_hist_valid'] = loss_hist_valid
    history['accuracy_hist_train'] = accuracy_hist_train
    history['accuracy_hist_valid'] = accuracy_hist_valid
    
    return model, history

#define the training parameters
num_epochs = 10
patience = 3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)

#training the model
best_model, hist = train(model, num_epochs, train_dl, val_dl)

#Model Evaluation
label_list = []
prediction_list = []

with torch.no_grad():
    for image, label in tqdm(test_dl):
        
        image = image.to(DEVICE)
        logits = best_model(image)
        probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
        prediction = np.argmax(probs, axis=1)
        label_list += label.numpy().tolist()
        prediction_list += prediction.tolist()

print(classification_report(label_list, prediction_list))

# ======================================================================================================================
# Task A
         # Build model object.
acc_A_train = hist['accuracy_hist_train'][-1]# Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A_test =  accuracy_score(label_list,prediction_list)

# ======================================================================================================================
## Print out your results with following format:
print('TA:{},{};'.format(acc_A_train, acc_A_test))
