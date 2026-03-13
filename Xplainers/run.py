import os
import torch
import nibabel as nib
from functools import partial

from Xplainers import gradient_based, LRP, perturbation_based, activation_based

class Explainers():
    def __init__(
            self, xai, model, input_data, affine, input_name, class_idx,
            category, device, save_folder):
        self.device = device
        self.xai = xai
        self.model = model.to(self.device)
        self.input_data = input_data
        self.affine = affine
        self.input_name = input_name
        self.class_idx = class_idx
        self.save_folder = save_folder
        self.category = category

        self.num_samples = []

        self.dispatch = {
            "BP" : self._run_sensitivity_analysis,
            "GBP" : self._run_guided_backpropagation,
            "LRP" : self._run_lrp,
            "IG" : self._run_integrated_gradients,
            "IDGI" : self._run_IDGI,
            "OS" : self._run_occlusion,
            "LIME" : self._run_lime,
            "RISE" : self._run_RISE,
            "GC++" : self._run_gradcampp,
            "SC" : self._run_scorecam,
            "LC" : self._run_layercam,
            "OC" : self._run_opticam,
        }

    def _create(
            self
    ):
        if self.xai in self.dispatch:
            result = self.dispatch[self.xai]()
            
            self._save(result)

        else:
            raise NotImplementedError(
                f"{self.xai} method is not implemented."
                f"Implemented methods are {list(self.dispatch.keys())}"
            )
        
    def _save(
            self,
            exp,
            input_name = None,
            class_idx = None
    ):
        input_name = input_name if input_name else self.input_name
        class_idx = class_idx if class_idx else self.class_idx

        save_dir = os.path.join(
            self.save_folder, self.category
        )
        
        os.makedirs(save_dir, exist_ok = True)

        nifti_img = nib.Nifti1Image(exp, self.affine)
        nib.save(
            nifti_img, 
            os.path.join(save_dir, f"{input_name}.nii.gz")
        )
        
    def _run_sensitivity_analysis(
            self
    ):
        exp = gradient_based.sensitivity_analysis(
            self.model, 
            self.input_data, 
            self.class_idx
        )
        
        return exp
    
    def _run_guided_backpropagation(
            self
    ):
        exp = gradient_based.guided_backprop(
            self.model, 
            self.input_data, 
            self.class_idx
        )

        return exp
    
    def _run_lrp(
              self
    ):
        image_tensor = torch.from_numpy(self.input_data).unsqueeze(0).unsqueeze(0)
        image_tensor = torch.nan_to_num(image_tensor, nan = 0.0)
        image_tensor = image_tensor.float()
        scan_data_tensor = image_tensor.to(self.device)

        LRPExplainer = LRP.InnvestigateModel(
            the_model = self.model,
            lrp_exponent = 1,
            beta = 0,
            epsilon = 1e-2,
            method = "composite-rule"
        )
        exp = LRPExplainer.innvestigator(scan_data_tensor)
        
        return exp
    
    def _run_integrated_gradients(
          self  
    ):
        IntegratedGrad = gradient_based.IntegratedGradients(self.model)
        exp, _ = IntegratedGrad.compute_integrated_gradients(
            self.input_data, 
            self.class_idx
        )

        return exp
    
    def _run_IDGI(
            self
    ):
        IG = gradient_based.IntegratedGradients(self.model)
        path_inputs, preds, grads = IG.get_path(
            inp = self.input_data, 
            target_label_index = self.class_idx)
        
        exp = gradient_based.IDGI(grads, preds)

        return exp
    
    def _run_occlusion(
            self
    ):
        OccExplainer = perturbation_based.Occlusion_Sensitivity(
            net = self.model,
            class_idx = self.class_idx,
            patch_size = 30,
            stride = None,
            baseline = 0.0,
            batch_size = 10,
            device = self.device
        )
        exp = OccExplainer.occlusion(self.input_data)

        return exp
    
    def _run_lime(
            self
    ):
        segment_fn = partial(perturbation_based.segment_grid, grid_size = 30)
        LIMExplainer = perturbation_based.LIME(
            net = self.model, 
            segment_3d_fn = segment_fn, 
            num_samples = 1000,
            device = self.device
        )
        exp = LIMExplainer.explain_instance(self.input_data, self.class_idx)

        return exp
    
    def _run_RISE(
            self
    ):
        RISExpainer = perturbation_based.RISE(
            net = self.model,
            class_idx = self.class_idx,
            mask_size = 10, 
            num_masks = 500, 
            p_keep = 0.5
        ) 
        exp = RISExpainer.explain(self.input_data)

        return exp
    
    def _run_gradcampp(
              self
    ):
        GradCAMppExplainer = activation_based.GradCAMpp(
            self.model, 
            self.input_data, 
            self.class_idx, 
            target_layer = None,
            device = self.device
        )
        exp = GradCAMppExplainer._forward()

        return exp
    
    def _run_scorecam(
              self
    ):
        ScoreCAMExplainer = activation_based.ScoreCAM(
            self.model, 
            self.input_data, 
            self.class_idx, 
            target_layer = None
        )
        exp = ScoreCAMExplainer._forward()

        return exp
    
    def _run_layercam(
            self
    ):
        LayerCAMExplainer = activation_based.LayerCAM(
            self.model, 
            self.input_data, 
            self.class_idx, 
            target_layer = None
        )
        exp = LayerCAMExplainer._forward()
        
        return exp
    
    def _run_opticam(
            self
    ):
        OptiCAMExplainer = activation_based.OptiCAM(
            self.model, self.input_data, None, None
        )
        exp = OptiCAMExplainer.generate()

        return exp