import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The format of the output from each interpretation method, is in the below format
    1. final shape is (D, H, W).
    2. Contains only positive values.
    3. Not normalized.
'''
# --------------------------- Activation-based Interpretation Methods ----------------------------------------------
class BaseCAM:
    def __init__(
            self, net, input, class_idx = None,
            target_layer = None, device = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device("cuda")

        self.model = net.to(self.device).eval()
        self.class_idx = class_idx

        input = input.astype("float32")

        imin, imax = input.min(), input.max()
        if imax > imin:
            input = (input - imin) / (imax - imin)
        if input.ndim == 3:
            self.input = torch.from_numpy(input).unsqueeze(0).unsqueeze(0).to(self.device)
        elif input.ndim == 4:
            self.input = torch.from_numpy(input).unsqueeze(0).to(self.device)
        else:
            raise ValueError("Expected input with shape (D,H,W) or (1,D,H,W).")
        
        self.input = torch.from_numpy(input).unsqueeze(0).unsqueeze(0).to(self.device)

        self.activations = {}
        self.gradients = {}

        def forward_hook(module, input_, output):
            self.activations["value"] = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients["value"] = grad_output[0]
        
        if target_layer is None:
            chosen = None

            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm3d):
                    chosen = m

            if chosen is None:
                raise ValueError(" No nn.BatchNorm3d layer found. Please specify the 'target_layer'")
            
            target_layer = chosen
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
        self.target_layer = target_layer

    def _compute_score(
            self,
            retain_graph : bool = False
    ) -> torch.Tensor:
        self.input = self.input.float().to(self.device)

        logits = self.model(self.input)
        logit = logits.view(-1)[0]

        if self.class_idx in (None, 1):
            score_tensor = logit
        elif self.class_idx == 0:
            score_tensor = -logit
        else:
            raise ValueError("class_idx must be None, 0, or 1 for a single-logit model.")
        
        return score_tensor
    
    def _backward(
            self, score_tensor, retain_graph = False
    ):

        self.model.zero_grad(set_to_none = True)
        score_tensor.backward(retain_graph = retain_graph)

    def _forward(
            self,
            retain_graph = False,
            **kwargs
    ) :
        score = self._compute_score(retain_graph = retain_graph)
        self._backward(score, retain_graph = retain_graph)
        activations = self.activations["value"]
        gradients = self.gradients["value"]
        cam = self._compute_cam(activations, gradients, **kwargs)

        return cam

    def _compute_cam(
            self,
            activations,
            gradients,
            **kwargs
    ):
        '''
        Subclasses should override this function to implement their specific CAM logic
        '''

        return NotImplementedError("Subclasses must implement _compute_cam()")

class GradCAMpp(BaseCAM):
    def _compute_cam(
            self, activations, gradients, **kwargs
    ):
        '''
        Perform Grad-CAM++ (Chattopadhay et al., 2018)
        '''
        activations = F.relu(activations)

        b, k, n, u, v = gradients.size()
        grad1 = gradients
        grad2 = gradients.pow(2)
        grad3 = gradients.pow(3)

        global_sum = activations.view(b, k, -1).sum(dim = 2, keepdim = True).view(b, k, 1, 1, 1)

        alpha_num = grad2
        alpha_denom = grad2 * 2.0 + grad3 * global_sum
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num / (alpha_denom + 1e-7)

        score = kwargs.get("score_tensor", None)
        if score is None:
            score = self._compute_score()

        positive_gradients = F.relu(grad1)

        alphas = torch.where(positive_gradients != 0, alphas, torch.zeros_like(alphas))

        norm_factor = alphas.view(b, k, -1).sum(dim = 2).view(b, k, 1, 1, 1)
        norm_factor = torch.where(norm_factor != 0, norm_factor, torch.ones_like(norm_factor))
        alphas = alphas / norm_factor

        weights = (alphas *  positive_gradients).view(b, k, -1).sum(dim = 2).view(b, k, 1, 1, 1)

        cam_map = F.relu((weights*activations).sum(dim = 1, keepdim = True))

        _, _, d, h, w = self.input.size()
        cam_map_1 = F.interpolate(cam_map, size = (d, h, w), mode = "trilinear", align_corners = False)

        cam = cam_map_1.squeeze().detach().cpu().numpy()

        return cam

class ScoreCAM(BaseCAM):
    def _compute_cam(
            self, activations, gradients, **kwargs
    ):
        '''
        Perform Score-CAM (Wang et al., 2020)
        '''
        activations = F.relu(activations)

        b, k, n, u, v = activations.size()
        _, _, D, H, W = self.input.size()

        score_saliency_map = torch.zeros((1, 1, D, H, W), device = self.device)

        def prob_for_class(logits, class_idx):
            p1 = torch.sigmoid(logits.view(-1)[0])  # prob of class 1
            if class_idx in (None, 1):
                return p1
            elif class_idx == 0:
                return 1.0 - p1
            else:
                raise ValueError("class_idx must be None, 0, or 1 for a single-logit model.")

        with torch.no_grad():
            for i in range(k):
                single_act = activations[:, i:i+1, :, :, :]

                up_act = F.interpolate(
                    single_act, size = (D, H, W), mode = 'trilinear', align_corners = False)

                if up_act.max() == up_act.min():
                    continue

                norm_up_act = up_act.clone()
                norm_up_act -= norm_up_act.min()
                norm_up_act /= norm_up_act.max()

                masked_input = self.input * norm_up_act

                out = self.model(masked_input)
                score_i = prob_for_class(out, self.class_idx)

                score_saliency_map += score_i * up_act

        score_saliency_map = F.relu(score_saliency_map)
        saliency_map = score_saliency_map.squeeze().detach().cpu().numpy()

        return saliency_map

class LayerCAM(BaseCAM):
    def _compute_cam(
            self, activations, gradients, **kwargs
    ):
        '''
        Perform Layer-CAM (Jiang et al., 2021)
        '''
        pos_grads = F.relu(gradients)
        weighted = activations * pos_grads
        cam_map = weighted.sum(dim = 1, keepdim = True)
        cam_map = F.relu(cam_map)
        _, _, D, H, W = self.input.size()
        cam_map_1 = F.interpolate(cam_map, size = (D, H, W), mode = 'trilinear', align_corners = False)
        cam_np = cam_map_1.squeeze().detach().cpu().numpy()

        return cam_np
    
class OptiCAM(BaseCAM):
    '''
    Perform Opti-CAM (Zhang et al., 2024)
    '''
    @torch.no_grad()
    def _find_class_idx(self):
        logits = self.model(self.input)
        if logits.ndim == 2 and logits.size(1) > 1:
            if self.class_idx is None:
                return int(logits.argmax(dim = 1).item())
            return int(self.class_idx)
        
        else:
            if self.class_idx in (None, 1):
                return 1
            elif self.class_idx == 0:
                return 0
            else:
                raise ValueError("class_idx must be None, 0, or 1 for a single-logit model.")
            
    def _gather_logits(
            self, logits, class_idx : int
    ):
        if logits.ndim == 2 and logits.size(1) > 1:
            return logits[:, class_idx].mean()
        else:
            logit = logits.view(-1)[0]
            return logit if class_idx == 1 else (-logit)
        
    @staticmethod
    def _safe_minmax_norm(
        x: torch.Tensor, eps: float = 1e-8
    ):
        b = x.shape[0]
        flat = x.view(b, -1)
        x_min = flat.min(dim = 1, keepdim = True)[0].view(b, *([1]*(x.ndim - 1)))
        x_max = flat.max(dim = 1, keepdim = True)[0].view(b, *([1]*(x.ndim - 1)))
        
        return (x - x_min) / (x_max - x_min + eps)
    
    def _build_saliency_from_feats(
            self, feats: torch.Tensor, u: torch.Tensor, input_spatial
    ):
        alpha = F.softmax(u, dim = 1)
        S = (alpha * feats).sum(dim = 1, keepdim = True)

        if len(input_spatial) == 3:
            S_up = F.interpolate(S, size = input_spatial, mode = 'trilinear', align_corners = False)
        else:
            S_up = F.interpolate(S, size = input_spatial, mode = 'bilinear', align_corners = False)

        S_norm = self._safe_minmax_norm(S_up)

        return S_norm
    
    def generate(
            self, max_iter = 100, lr = 1e-2, use_relu_on_feats = True,
            verbose = False, early_stop_patience = 10
    ):
        device = self.device
        x = self.input.float().to(device)
        B, C_in = x.shape[:2]
        spatial = x.shape[2:]

        with torch.no_grad():
            _ = self.model(x)
            feats = self.activations["value"]
            if use_relu_on_feats:
                feats = F.relu(feats, inplace=False)
            feats = feats.detach()

        if feats.dim() not in (4, 5):
            raise RuntimeError("Expected target_layer activations to be 4D (B,C,H,W) or 5D (B,C,D,H,W).")

        C_feat = feats.shape[1]
        u_shape = (B, C_feat) + (1,) * (feats.dim() - 2)
        u = nn.Parameter(torch.zeros(u_shape, device = device))

        for p in self.model.parameters():
            p.requires_grad_(False)

        opt = torch.optim.Adam([u], lr = lr)
        class_idx = self._find_class_idx()

        best_score = -float("inf")
        best_S = None
        bad = 0

        for it in range(max_iter):
            S = self._build_saliency_from_feats(feats, u, input_spatial = spatial)
            x_mask = x * S if C_in == 1 else x * S.repeat(1, C_in, *([1]* (x.dim()-2)))
            
            logits = self.model(x_mask)
            score = self._gather_logits(logits, class_idx)

            loss = -score
            opt.zero_grad(set_to_none = True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                if score.item() > best_score + 1e-6:
                    best_score = score.item()
                    best_S = S.clone()
                    bad = 0
                else:
                    bad += 1

            if verbose:
                print(f"[Opti-CAM] iter {it+1}/{max_iter} | score={score.item():.4f}", end="\r")

            if bad >= early_stop_patience:
                if verbose:
                    print("\n[Opti-CAM] early stopping.")
                break

        S_out = best_S if best_S is not None else S
        sal = S_out.detach().cpu().squeeze().numpy()

        return sal
    
    def _forward(
            self, **kwargs
    ):
        return self.generate(**kwargs)