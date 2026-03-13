import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score
)

from NeuroPrep.transforms import *
from TorchUtils.trainer import CNN_trainer
from TorchUtils.architecture import ClassificationModel3D
from Settings import Settings, Pred_Settings

input_path = Settings["input_path"]
dataset = os.path.basename(input_path)
info = pd.read_csv(f"{input_path}/info.csv")
train_folder = f"{input_path}/train"
val_folder = f"{input_path}/val"
test_folder = f"{input_path}/test"

model_path = Settings["model_path"]

categories = Pred_Settings[dataset]["Categories"]

device = torch.device("cuda")

class ToTensor(object):
    def __call__(self, image):
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.float()

        return image

class MRIDataset(Dataset):
    def __init__(
            self,
            root_dir,
            dataframe,
            transform = None,
            min_max_normalization = True,
            mean_std_normalization = False
    ):
        self.root_dir = root_dir
        self.dataframe = dataframe
        self.transform = transform
        self.min_max_normalization = min_max_normalization
        self.mean_std_normalization = mean_std_normalization
        self.files = os.listdir(self.root_dir)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(
            self
    ):
        return len(self.files)
    
    def __getitem__(
            self,
            idx
    ):
        # scan
        img_path = os.path.join(self.root_dir, self.files[idx])
        image = np.array(nib.load(img_path).get_fdata())

        if self.mean_std_normalization:
            mean, std = np.mean(image), np.std(image)
            image = (image - mean) / (std + 1e-8)

        if self.min_max_normalization:
            vmin, vmax = image.min(), image.max()
            if vmax > vmin:
                image = (image - vmin) / (vmax - vmin + 1e-8)

        if self.transform:
            image = self.transform(image)

        category = categories
        image_name = img_path.split("/")[-1].replace(".nii.gz","")
        label_name = self.dataframe.loc[self.dataframe["Image_name"] == image_name, "Group"].iloc[0]
        label = torch.tensor([category[label_name]], dtype = torch.long)

        sample = {
            "image" : image,
            "label" : label
        }

        return sample

class AugmentationLogger:
    def __init__(self):
        self.manager = mp.Manager()
        self.shared = self.manager.dict()
        self.lock = mp.Lock()

    def inc(self, key):
        with self.lock:
            self.shared[key] = self.shared.get(key, 0) + 1

    def reset(self):
        with self.lock:
            self.shared.clear()

    def report(self):

        return dict(self.shared)

class OneOf:
    def __init__(self, transforms, p=1.0, name="OneOf", logger: AugmentationLogger=None):
        self.transforms = list(transforms)
        self.p = float(p)
        self.name = name
        self.logger = logger

    def __call__(self, x):
        if np.random.rand() >= self.p or not self.transforms:

            return x
        
        idx = np.random.choice(len(self.transforms))
        t = self.transforms[idx]

        if self.logger:
            self.logger.inc(f"{self.name}:{getattr(t, '__class__', type('T', (), {})).__name__}")

        return t(x)

# Get the label of the image
def get_label(fname, dataframe, Categories):
    image_name = fname.replace(".nii.gz", "")
    grp = dataframe.loc[dataframe["Image_name"] == image_name, "Group"].iloc[0]

    return Categories.get(grp, -1)

files = os.listdir(train_folder)
files.sort() 
labels = np.array([get_label(f, info, categories) for f in files])

# Parameters
aug_logger = AugmentationLogger()
params = {
    "num_folds" : 5,
    "batch_size" : 16,
    "augmentations" : [
        OneOf(
            [SagittalFlip(), SagittalRotate(deg = (-2, 2)), SagittalTranslate(dist = (-2, 2))], 
            p = 0.5, name = "Sagittal", logger = aug_logger)], 
    "dropout" : 0.3,
    "loss" : {
        "name" : "BCElogitloss",
        "pos_w" : 1
    },
    "optimizer" : {
        "name" : "Adam",
        "lr" : 0.0004,
        "wd" : 0.001,
    },
    'scheduler' : {
        'name' : 'ReduceOnPlateau',
        'step_size' : 4,
        'gamma' : 0.5,
        'mode' : ['macro_f1','max']
    },
    "threshold" : {
        "mode" : "search", # fixed / search
        "grid" : np.arange(0.4, 0.61, 0.01).round(2).tolist(), # np.arange(0.1, 0.91, 0.01).round(2).tolist()
        "thresh" : 0.5
    }
}
pp = 37

train_transform = transforms.Compose(params["augmentations"] + [ToTensor()])
val_transform   = transforms.Compose([ToTensor()])
test_transform  = transforms.Compose([ToTensor()])  # keep test clean (no aug)

train_ds = MRIDataset(train_folder, info, transform=train_transform)
val_ds   = MRIDataset(val_folder,   info, transform=val_transform)
test_ds  = MRIDataset(test_folder,  info, transform=test_transform)

train_loader = DataLoader(
    train_ds,
    batch_size=params["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

net = ClassificationModel3D(params["dropout"]).to(device)

# criterion
if params['loss']['name'] == 'BCElogitloss':
    positive_w = torch.tensor([params['loss']["pos_w"]], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_w)
else:
    raise ValueError(f"param {params['loss']['name']} is invalid for criterion")

# optimizer
if params['optimizer']['name'] == 'Adam':
    optimizer = optim.Adam(
        net.parameters(),
        lr=params['optimizer']["lr"],
        weight_decay=params['optimizer']["wd"]
    )
else:
    raise ValueError(f"param {params['optimizer']['name']} is invalid for optimizer")

# scheduler
if params['scheduler']['name'] == 'ReduceOnPlateau':
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=params['scheduler']['mode'][1],   # "max" for macro_f1
        factor=params['scheduler']["gamma"],
        patience=params['scheduler']["step_size"],
        verbose=True,
        min_lr=1e-6
    )
else:
    raise ValueError(f"param {params['scheduler']['name']} is invalid for scheduler")

# trainer
CNNTrainer = CNN_trainer(
    net, device, criterion, optimizer,
    scheduler=scheduler,
    patience=10,
    threshold_mode=params["threshold"]["mode"],          # "search" or "fixed"
    fixed_threshold=params["threshold"]["thresh"],
    threshold_grid=params["threshold"]["grid"],
    selection_metric=params['scheduler']['mode'][0]      # e.g. "macro_f1"
)

save_prefix = f"{model_path}/{pp}"
CNNTrainer.run_process(train_loader, val_loader, save_prefix)

## Inference
state = torch.load(f"{model_path}/{pp}_BEST_ITERATION.pth", map_location = device)
net.load_state_dict(state)
net.eval()

# load threshold
with open(f"{model_path}/{pp}_BEST_THRESHOLD.json", "r") as f:
    thr = float(json.load(f)["threshold"])

all_probs = []
all_labels = []
all_names = []

with torch.no_grad():
    for batch in test_loader:
        x = batch["image"].to(device)
        y = batch["label"].view(-1).cpu().numpy().astype(int)

        logits = net(x).view(-1)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(y.tolist())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels).astype(int)
all_preds = (all_probs >= thr).astype(int)

# metrics
acc = accuracy_score(all_labels, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1])
macro_f1 = (f1[0] + f1[1]) / 2.0

try:
    roc = roc_auc_score(all_labels, all_probs)
except ValueError:
    roc = float("nan")
pr_auc = average_precision_score(all_labels, all_probs)

print("\nTEST RESULTS")
print(f" threshold = {thr:.2f}")
print(f" accuracy  = {acc*100:.2f}%")
print(f" macro_f1  = {macro_f1:.4f}")
print(f" roc_auc   = {roc:.4f}")
print(f" pr_auc    = {pr_auc:.4f}")
print("\nPer-class:")
print(f" class 0: precision={prec[0]:.4f} recall={rec[0]:.4f} f1={f1[0]:.4f}")
print(f" class 1: precision={prec[1]:.4f} recall={rec[1]:.4f} f1={f1[1]:.4f}")

# save confusion matrix
cm_path = f"{save_prefix}_test_Confusion_matrix.png"
CNN_trainer.plot_confusion_matrix(None, all_labels.tolist(), all_preds.tolist(), cm_path)

# save predictions CSV
out_csv = f"{save_prefix}_test_predictions.csv"
df = pd.DataFrame({
    "y_true": all_labels,
    "y_prob": all_probs,
    "y_pred": all_preds,
    "threshold": thr
})
df.to_csv(out_csv, index=False)
print(f"\nSaved: {cm_path}")
print(f"Saved: {out_csv}")