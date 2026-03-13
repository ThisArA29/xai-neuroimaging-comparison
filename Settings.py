Settings = {
    "input_path" : "/opt/home/s4043685/Project1/PPMI",
    "model_path" : "/opt/home/s4043685/Project1/Best_model/PPMI",
    "xai_output_path" : "/opt/home/s4043685/Project1/Explanations/PPMI",
    "reference_image_path" : "/opt/home/s4043685/fsl/data/linearMNI/MNI152lin_T1_1mm_brain.nii.gz",
    "atlas_imageg_path" : "/opt/home/s4043685/Project1/MyModel/Resampled_atlases/harvard_oxford_subcortical_atlas.nii.gz",
    "XAI" : {
        "BP" : False, "GBP" : False, "LRP" : False, "IG" : False, "IDGI" : False,
        "OS" : False, "LIME" : True, "RISE" : False,
        "GC++" : False, "SC" : False, "LC" : False, "OC" : False  
    }
}

Pred_Settings = {
    "ADNI" : {
        "Categories" : {"AD" : 1, "CN" : 0},
        "thresh" : 0.53
    },
    "PPMI" : {
        "Categories" : {"PD" : 1, "Control" : 0},
        "thresh" : 0.57
    }
}
