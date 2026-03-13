import os
import ast
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)

from TorchUtils.architecture import ClassificationModel3D_inf
from NeuroPrep.transforms import *
from Settings import Settings, Pred_Settings

rand_type = "randomization-similar"
def minmax_norm(x):
    x = x.astype(np.float32)
    x = x - np.nanmin(x)
    denom = np.nanmax(x)

    if denom > 0:
        x = x / denom

    # mean, std = np.mean(x), np.std(x)
    # x = (x - mean) / (std + 1e-8)

    return x

def get_true_label(dataset, category, label_map):
    return label_map[dataset]["Categories"][category]

def list_nii_gz(folder):
    return sorted([f for f in os.listdir(folder) if f.endswith(".nii.gz")])

def create_noisy_test(
        input_path, rand_type, atlas_dict
):
    original_test = os.path.join(input_path, "test")

    if rand_type == "Original":
        return original_test
    
    save_dir = os.path.join(input_path, f"test_{rand_type}")
    os.makedirs(save_dir, exist_ok = True)

    src_files = list_nii_gz(original_test)
    dst_files = list_nii_gz(save_dir)

    if len(dst_files) == len(src_files) and set(dst_files) == set(src_files):
        print(f"Existing noisy test folder: {save_dir}")
        return save_dir
    
    print(f"Generating noisy test set in: {save_dir}")

    if rand_type.split("-")[0] == "gaussian":
        std = float(rand_type.split("-")[-1])
        aug = RandomGaussianNoise(sigma=(std, std))
        apply_fn = lambda data: aug(data)
    elif rand_type.split("-")[0] == "rician":
        std = float(rand_type.split("-")[-1])
        aug = RandomRicianNoise(sigma=(std, std))
        apply_fn = lambda data: aug(data)
    elif rand_type.split("-")[0] == "poisson":
        std = float(rand_type.split("-")[-1])
        aug = RandomPoissonNoise(peak=std, p=1)
        apply_fn = lambda data: aug(data)
    elif rand_type.split("-")[0] == "gibbs":
        ratio = float(rand_type.split("-")[-1])
        aug = RandomGibbsRinging(truncation_range=(ratio, ratio), p=1)
        apply_fn = lambda data: aug(data)
    elif rand_type.split("-")[0] == "randomization":
        aug = RandomizeBrainRegionVoxels(atlas_dict)
        region_name = rand_type.split("-")[-1]
        apply_fn = lambda data: aug._create(data, region_name)
    else:
        raise ValueError(f"Unknown rand_type: {rand_type}")
    
    for fname in src_files:
        src_path = os.path.join(original_test, fname)
        dst_path = os.path.join(save_dir, fname)

        scan = nib.load(src_path)
        affine = scan.affine
        data = scan.get_fdata()

        data = minmax_norm(data)
        data = apply_fn(data)

        nib.save(nib.Nifti1Image(data, affine), dst_path)

    return save_dir

input_path = Settings["input_path"]
dataset = os.path.basename(input_path)
info = pd.read_csv(f"{input_path}/info.csv")

model_path = Settings["model_path"]

device = torch.device("cuda")

net = ClassificationModel3D_inf()
state_dict = torch.load(f"{model_path}/BEST_ITERATION.pth", map_location = device)
net.load_state_dict(state_dict)
net = net.eval().to(device)

# load the atlas
atlas_dict = {}
atlas_name = "desikan"
atlas = nib.load(f"Resampled_atlases/{atlas_name}_atlas.nii.gz")
atlas_data = atlas.get_fdata()
with open(f"Resampled_atlases/{atlas_name}_labels.txt", 'r') as file:
    content = file.read()
    region_dict = ast.literal_eval(content)

atlas_dict["name"] = atlas_name
atlas_dict["data"] = atlas_data
atlas_dict["labels"] = region_dict

test_folder = create_noisy_test(input_path, rand_type, atlas_dict)
files = list_nii_gz(test_folder)

probs = {}
buckets = {"TP":[], "TN":[], "FP":[], "FN":[]}
test_all_labels = []
test_all_preds = []
test_all_probs = []

thresh = Pred_Settings[dataset]["thresh"]

for file in files:
    file_path = os.path.join(test_folder, file)
    image_name = file.replace(".nii.gz","")

    match = info.loc[info["Image_name"] == image_name, "Group"]
    if match.empty:
        raise ValueError(f"Image_name '{image_name}' not found in info.csv")

    true_cat = match.iloc[0]
    true_label = get_true_label(dataset, true_cat, Pred_Settings)

    scan = nib.load(file_path)
    data = scan.get_fdata()

    if rand_type == "Original":
        data = minmax_norm(data)

    input_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.float().to(device)

    with torch.no_grad():
        outputs = net(input_tensor)
        # print("outputs:", outputs.detach().cpu().numpy(), "min/max:", outputs.min().item(), outputs.max().item())
        prob_pos = torch.sigmoid(outputs).item()
        pred = 1 if prob_pos >= thresh else 0
        print(file, true_label, prob_pos)

    probs[image_name] = float(prob_pos) 
    test_all_labels.append(true_label)
    test_all_preds.append(pred)
    test_all_probs.append(prob_pos)

    if true_label == 1 and pred == 1:
        buckets["TP"].append(image_name)
    elif true_label == 0 and pred == 0:
        buckets["TN"].append(image_name)
    elif true_label == 0 and pred == 1:
        buckets["FP"].append(image_name)
    elif true_label == 1 and pred == 0:
        buckets["FN"].append(image_name)

with open(f"{model_path}/inference_subjects_{rand_type}.txt", "w") as f:
    f.write(str(buckets))

print({key: len(value) for key, value in buckets.items()})

if rand_type == "Original":
    test_accuracy = accuracy_score(test_all_labels, test_all_preds) * 100
    precision = precision_score(test_all_labels, test_all_preds, average = "binary") * 100
    recall = recall_score(test_all_labels, test_all_preds, average = "binary") * 100
    f1 = f1_score(test_all_labels, test_all_preds, average = "binary") * 100
    auc = roc_auc_score(test_all_labels, test_all_probs) * 100

    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Precision: {precision:.2f}%")
    print(f"Test Recall: {recall:.2f}%")
    print(f"Test f1: {f1:.2f}%")
    print(f"Test auc: {auc:.2f}%")

    cm = confusion_matrix(test_all_labels, test_all_preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100
    disp = ConfusionMatrixDisplay(confusion_matrix = cm_percent, display_labels = [0,1])

    fig, ax = plt.subplots(figsize = (6,5))
    disp.plot(cmap = "Blues", ax = ax, values_format = ".2f")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i - 0.2, f"{cm[i, j]}", ha = "center", va = "center", color = "black")
    plt.title(f"Confusion Matrix for Test Set")
    plt.savefig(f"{model_path}/test_Confusion_matrix.png")
    plt.close()