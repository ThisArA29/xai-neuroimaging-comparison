'''
source - https://github.com/Angeluz-07/MRI-preprocessing-techniques/blob/main/notebooks/00_libs_review.ipynb
'''

import os
import glob
import shutil
import zipfile
import tempfile
import concurrent.futures

import nibabel as nib
from nilearn import plotting
from nilearn.image import resample_to_img

import SimpleITK as sitk
import ants

from NeuroPrep.ANTs import (
    convert_to_DICOM,
    bias_correction,
    skull_stripping,
    intensity_normalization,
    linear_registration
)
from NeuroPrep.interactive_mris import add_suffix_to_filename
from Settings import Settings

def mri_pipeline(
        raw_img_path, template_img_path, atlas_img_path, input_path, group, subject_id
):
    os.makedirs(f"{input_path}/test_img", exist_ok = True)
    os.makedirs(f"{input_path}/PNG/{group}", exist_ok = True)
    os.makedirs(f"{input_path}/Processed", exist_ok = True)

    # Convert
    raw_img_sitk = convert_to_DICOM(raw_img_path)

    # Bias correction
    corrected = bias_correction(raw_img_sitk)
    out_path = os.path.join(input_path, add_suffix_to_filename(f"test_img/{subject_id}.nii", "bs"))
    sitk.WriteImage(corrected, out_path)

    # Brain extraction
    corrected = ants.image_read(out_path, reorient="IAL")
    _, corrected = skull_stripping(corrected, modality="t1")
    out_path = os.path.join(input_path, add_suffix_to_filename(f"test_img/{subject_id}.nii", "be"))
    corrected.to_file(out_path)

    # Intensity normalization
    corrected = intensity_normalization(out_path, template_img_path)
    out_path = os.path.join(input_path, add_suffix_to_filename(f"test_img/{subject_id}.nii", "in"))
    sitk.WriteImage(corrected, out_path)

    # Linear registration
    corrected = linear_registration(template_img_path, out_path)
    out_path = os.path.join(input_path, add_suffix_to_filename(f"test_img/{subject_id}.nii", "lr"))
    corrected.to_file(out_path)

    # Resample to atlas and save outputs
    scan = nib.load(out_path)
    atlas = nib.load(atlas_img_path)
    resampled = resample_to_img(scan, atlas, interpolation="nearest")

    display = plotting.plot_anat(resampled, draw_cross=False, cut_coords=(0, 0, 0))
    display.add_contours(atlas, levels=[0.5], colors="r")
    display.savefig(f"{input_path}/PNG/{group}/{subject_id}.png", dpi=300)
    display.close()

    nib.save(resampled, f"{input_path}/Processed/{subject_id}.nii.gz")
    return subject_id, "OK"

def batch_process_mri(
        input_dir, template_img_path, atlas_img_path, input_path, info, max_workers = 8
):
    """Process every .nii/.nii.gz file in input_dir in parallel."""
    all_files = glob.glob(os.path.join(input_dir, "*.nii*"))
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for mri_file in all_files:
            subject_id = os.path.basename(mri_file).replace(".nii.gz", "").replace(".nii", "")
            group = info.loc[info["Image_name"] == subject_id, "Group"].item()
            futures[executor.submit(
                mri_pipeline,
                mri_file,
                template_img_path,
                atlas_img_path,
                input_path,
                group,
                subject_id
            )] = mri_file

        for fut in concurrent.futures.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                bad_file = futures[fut]
                results.append((os.path.basename(bad_file), f"FAILED: {e}"))

    return results

def extract_zip_to_temp(zip_path, subjects_to_extract):
    """Extract only selected subjects from a zip into a temp folder."""
    temp_dir = tempfile.mkdtemp(prefix="mri_extract_")

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for item in zip_file.namelist():
            if not item.endswith(".nii"):
                continue

            parts = item.split("/")[-1].split("_")
            new_file_name = (
                parts[1] + "_" + parts[2] + "_" + parts[3] + "__" + parts[-1].replace(".nii", "")
            )

            if new_file_name in subjects_to_extract:
                out_path = os.path.join(temp_dir, f"{new_file_name}.nii")
                with zip_file.open(item) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())

    return temp_dir

def main():
    input_path = Settings["input_path"]
    template_img_path = Settings["reference_image_path"]
    atlas_img_path = Settings["atlas_image_path"]
    info = f"{input_path}/info.csv"
    subjects_to_extract = info["Image_name"].to_list()

    # Skip already processed
    processed_dir = f"{input_path}/Processed"
    os.makedirs(processed_dir, exist_ok = True)
    already_done = {f.replace(".nii.gz", "") for f in os.listdir(processed_dir) if f.endswith(".nii.gz")}
    subjects_to_extract = subjects_to_extract - already_done

    zipped_folder = f"{input_path}/zipped"
    for zip_name in os.listdir(zipped_folder):
        zip_path = os.path.join(zipped_folder, zip_name)
        if not zip_path.endswith(".zip"):
            continue

        extracted_dir = extract_zip_to_temp(zip_path, subjects_to_extract)
        try:
            results = batch_process_mri(
                extracted_dir,
                template_img_path,
                atlas_img_path,
                input_path,
                info,
                max_workers = 10
            )
            for sid, status in results:
                print(sid, status)
        finally:
            shutil.rmtree(extracted_dir, ignore_errors = True)

if __name__ == "__main__":
    main()