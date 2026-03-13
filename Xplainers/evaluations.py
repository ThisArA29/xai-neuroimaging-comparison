import os
import numpy as np
import nibabel as nib
from nilearn import plotting

from nilearn.image import resample_to_img

# Average heatmap
def avg_saliency(
        folder_path,
        avg_save_path
):
    heatmap_comb = np.zeros((182, 218, 182), dtype = np.float64)
    hh_count = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        hh = nib.load(file_path)
        hh_affine = hh.affine
        hh_data = hh.get_fdata()

        hh_data[hh_data < 0] = 0

        if hh_data.min() < hh_data.max():
            hh_data -= hh_data.min()
            hh_data /= hh_data.max()

        heatmap_comb += hh_data
        hh_count += 1

    heatmap_comb /= hh_count

    nifti_img = nib.Nifti1Image(heatmap_comb, hh_affine)
    nib.save(nifti_img, avg_save_path)

    return heatmap_comb

# Overlaying heatmap on the brain image
def plot_series_heatmap(
        heatmap_data, bg_img, cut_coords, display_mode,
        name, set_nan = True, set_colorbar = True,
):
    data = heatmap_data.copy()
    mask = bg_img.get_fdata()
    affine = bg_img.affine

    # Normalize
    if data.min() < data.max():
        data -= data.min()
        data /= data.max()
    else: 
        data = data

    epsilon = 1e-18
    data[data == 0] = epsilon

    if set_nan:
        data[mask == 0] = np.nan

    stat_img = nib.Nifti1Image(data, affine)
    stat_img = resample_to_img(stat_img, bg_img, interpolation = "continuous")
    
    display = plotting.plot_stat_map(
        stat_map_img = stat_img, bg_img = bg_img, display_mode = display_mode,
        cut_coords = cut_coords, threshold = 0, cmap = "hot", vmin = 0, vmax = 1,
        draw_cross = False, annotate = False, alpha = 0.5, black_bg = False, colorbar = set_colorbar
    )

    display.savefig(name, dpi = 300)
    display.close()

# size normalized importance
def size_norm_importance(atlas_dict, heatmap_data, label):
    region_mask = np.round(atlas_dict["data"]) == label
    voxel_count = np.sum(region_mask)

    sum_intensity = np.sum(heatmap_data[region_mask])
    average_intensity = sum_intensity / voxel_count  

    return average_intensity