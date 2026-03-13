Settings = {
    "input_path" : "your/input/pth",
    "model_path" : "your/model/path",
    "xai_output_path" : "your/explanation/path",
    "reference_image_path" : "reference/image/path",
    "atlas_imageg_path" : "harvard/oxford/cortical_and_subcortical_atlas",
    "XAI" : {
        "BP" : True, "GBP" : True, "LRP" : True, "IG" : True, "IDGI" : True,
        "OS" : True, "LIME" : True, "RISE" : True,
        "GC++" : True, "SC" : True, "LC" : True, "OC" : True
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
