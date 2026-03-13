import ast
import torch
import nibabel as nib
from pathlib import Path

from Xplainers.run import Explainers
from TorchUtils.architecture import ClassificationModel3D_inf
from Xplainers.evaluations import avg_saliency, plot_series_heatmap

from Settings import Settings

rand_type = "gibbs-ringing-0.6"

device = torch.device("cuda")

input_path = Settings["input_path"]
model_path = Settings["model_path"]
xai_output_path = Settings["xai_output_path"]

bg_img = nib.load("Resampled_atlases/bg_img_MNI152lin_1mm.nii.gz")

if rand_type == "Original":
    test_folder = f"{input_path}/test"
else:
    test_folder = f"{input_path}/test_{rand_type}"

XAI = Settings["XAI"]
true_e = {k: v for k, v in XAI.items() if v is True}

net = ClassificationModel3D_inf()
state_dict = torch.load(f"{model_path}/BEST_ITERATION.pth", map_location = device)
net.load_state_dict(state_dict)
net = net.eval().to(device)

with open(f"{model_path}/inference_subjects_{rand_type}.txt") as f:
    subject_categories = f.read().strip()
subject_categories = ast.literal_eval(subject_categories)

for xai, value in true_e.items():
    xai_folder = Path(f"{xai_output_path}/{rand_type}/{xai}")

    for cat in ["TP","TN"]:
        cat_subject_list = subject_categories[cat]

        for sub in cat_subject_list:
            filepath = f"{test_folder}/{sub}.nii.gz"
            scan = nib.load(filepath)
            scan_data = scan.get_fdata()

            class_idx = 1 if cat == "TP" else 0

            affine = scan.affine

            if rand_type == "Original":
                scan_data -= scan_data.min()
                scan_data /= scan_data.max()
            
            nii_files = list(xai_folder.glob(f"{cat}/*.nii.gz"))
            already_done = [f.name.replace(".nii.gz", "") for f in nii_files]

            if sub in already_done:
                continue

            Explainers(
                xai, net, scan_data, affine, sub, class_idx, cat, 
                device, xai_folder
            )._create()

        # Average explanations
        cat_paths = [
            p for p in xai_folder.rglob(f"{cat}") if p.is_dir()
        ]

        plot_dir = f"{xai_folder}/plots"
        Path(plot_dir).mkdir(parents = True, exist_ok = True)

        for path in cat_paths:
            heatmap_comb = avg_saliency(
                path, f"{xai_folder}/{cat}.nii.gz")
            
            # Plot
            plot_series_heatmap(
                heatmap_comb, bg_img, [-38, -18, 12], "z", 
                f"{plot_dir}/{cat}.png",
                set_nan = True, set_colorbar = True
            )
