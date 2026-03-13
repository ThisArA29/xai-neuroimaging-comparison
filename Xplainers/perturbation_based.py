import torch
import numpy as np
from lime import lime_image
import torch.nn.functional as F
import math
import random

'''
The format of the output from each XAI method, is in the below format
    1. final shape is (D, H, W).
    2. Contains only positive values.
    3. Not normalized.
'''
# --------------------------- Perturbation-based XAI ----------------------------------------------
class Occlusion_Sensitivity:
    def __init__(
        self, net, class_idx, patch_size, stride = None,
        baseline = 0.0, batch_size = 8, device = None
    ):
        '''
        Perform Occlusion Sensitivity (Zeiler and Fergus, 2014)
        '''
        if isinstance(device, str):
            device = torch.device(device)
        elif device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.net = net.eval().to(self.device)

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        if stride is None:
            stride = tuple(max(1, k // 2) for k in patch_size)
        elif isinstance(stride, int):
            stride = (stride, stride, stride)

        self.patch_size = patch_size
        self.stride = stride

        self.baseline = float(baseline)
        self.batch_size = int(batch_size)
        self.class_idx = class_idx

    @staticmethod
    def _check_volume(vol):
        if not isinstance(vol, np.ndarray):
            raise TypeError("volume must be a numpy.ndarray.")
        if vol.ndim != 3:
            raise ValueError(f"volume must have shape (D,H,W), got {vol.shape}.")
        if not np.issubdtype(vol.dtype, np.number):
            raise TypeError("volume dtype must be numeric.")
        if vol.size == 0:
            raise ValueError("volume is empty.")

    @staticmethod
    def _starts(size, k, s):
        idxs = list(range(0, max(1, size - k + 1), max(1, s)))
        if len(idxs) == 0 or idxs[-1] != size - k:
            idxs.append(max(0, size - k))

        return idxs

    @staticmethod
    def _minmax_01(x, eps = 1e-6):
        x = np.nan_to_num(x, nan=0.0).astype(np.float32, copy = False)
        m, M = float(x.min()), float(x.max())
        rng = max(M - m, eps)
        y = (x - m) / rng
        return np.clip(y, 0.0, 1.0)

    @torch.no_grad()
    def occlusion(
        self, volume, signed = False,
    ):
        print(" new image")
        self._check_volume(volume)
        D, H, W = volume.shape
        kD, kH, kW = self.patch_size
        sD, sH, sW = self.stride

        vol_np = self._minmax_01(volume)
        vol = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).to(self.device)

        out = self.net(vol)
        logit = out.view(-1)[0]
        p1 = torch.sigmoid(logit)
        if self.class_idx in (None, 1):
            ref_prob = p1  
        elif self.class_idx == 0:
            ref_prob = 1.0 - p1
        else:
            raise ValueError("class_idx must be None, 0, or 1 for a single-logit model.")

        heat  = torch.zeros((D, H, W), dtype=vol.dtype, device=self.device)
        count = torch.zeros((D, H, W), dtype=vol.dtype, device=self.device)

        d_starts = self._starts(D, kD, sD)
        h_starts = self._starts(H, kH, sH)
        w_starts = self._starts(W, kW, sW)
        windows = [
            (dz, min(dz + kD, D), hy, min(hy + kH, H), wx, min(wx + kW, W))
            for dz in d_starts for hy in h_starts for wx in w_starts
        ]

        B = self.batch_size
        base_val = torch.as_tensor(self.baseline, dtype=vol.dtype, device = self.device)

        for i in range(0, len(windows), B):
            batch_windows = windows[i:i+B]
            Bcur = len(batch_windows)

            occ_batch = vol.repeat(Bcur, 1, 1, 1, 1)
            for b, (d0, d1, h0, h1, w0, w1) in enumerate(batch_windows):
                occ_batch[b, :, d0:d1, h0:h1, w0:w1] = base_val

            out = self.net(occ_batch) 
            logits = out.view(-1)
            p1 = torch.sigmoid(logits)
            if self.class_idx in (None, 1):
                probs = p1
            elif self.class_idx == 0:
                probs = 1.0 - p1

            deltas = ref_prob - probs
            if not signed:
                deltas = torch.clamp(deltas, min = 0)

            for b, (d0, d1, h0, h1, w0, w1) in enumerate(batch_windows):
                heat[d0:d1, h0:h1, w0:w1] += deltas[b]
                count[d0:d1, h0:h1, w0:w1] += 1

        count = torch.clamp(count, min=1)
        heatmap = (heat / count).detach().float().cpu().numpy()
        return heatmap

def segment_grid(
        image,
        grid_size
):
    x_dim, y_dim, z_dim = image.shape
    labels = np.zeros_like(image, dtype = int)
    global_min = image.min()
    label_counter = 1

    for x in range(0, x_dim, grid_size):
        for y in range(0, y_dim, grid_size):
            for z in range(0, z_dim, grid_size):
                x_end = min(x + grid_size, x_dim)
                y_end = min(y + grid_size, y_dim)
                z_end = min(z + grid_size, z_dim)

                sub_cube = image[x:x_end, y:y_end, z:z_end]

                labels[x:x_end, y:y_end, z:z_end] = label_counter
                label_counter += 1

    return labels

class LIME():
    def __init__(
            self, net, segment_3d_fn, num_samples, device     
    ):
        '''
        Perform LIME (Ribeiro et al., 2016)

        Args:
            net: torch model mapping (N,1,D,H,W) -> (N, num_classes) (logits or probs).
            image_dataset: list of volumes (D,H,W), numeric dtype.
            segment_3d_fn: function taking a (D,H,W) array -> integer segment map (D,H,W).
            num_samples: number of LIME perturbation samples.
            top_labels: how many labels LIME should consider.
            device: torch device; defaults to the model's device or CUDA if available.
        '''

        self.net = net.eval()
        self.segment_3d_fn = segment_3d_fn
        self.num_samples = num_samples
        self.device = device

    def _normalize_for_lime(self, image: np.ndarray) -> np.ndarray:
        """
        LIME's LimeImageExplainer expects an 'image-like' numpy array.
        For 3D MRI (D,H,W), we keep it 3D and normalize to [0,1].
        """
        img = np.asarray(image, dtype=np.float32)

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        mn = float(img.min())
        mx = float(img.max())
        if mx - mn < 1e-8:
            return np.zeros_like(img, dtype=np.float32)

        img = (img - mn) / (mx - mn)
        return img.astype(np.float32, copy=False)
    
    def _stack_as_5d(self, images_list):
        """
        Convert list of (D,H,W) numpy arrays into torch tensor (N,1,D,H,W)
        on the correct device.
        """
        arr = np.stack([np.asarray(im, dtype=np.float32) for im in images_list], axis=0)
        if arr.ndim != 4:
            raise ValueError(f"Expected stacked shape (N,D,H,W), got {arr.shape}")
        t = torch.from_numpy(arr).unsqueeze(1)  # (N,1,D,H,W)
        return t.to(self.device)

    def explain_instance(
            self, image, true_label
    ):
        # self.check_volume(image)

        transformed_img = self._normalize_for_lime(image)
        explainer = lime_image.LimeImageExplainer()

        explanation = explainer.explain_instance(
            transformed_img,
            classifier_fn = self.batch_predict,
            labels = [true_label],
            hide_color = 0,
            num_samples = self.num_samples,
            segmentation_fn = self.segment_3d_fn
        )

        ind = explanation.top_labels[0]
        weights_dict = dict(explanation.local_exp[ind])
        segments = explanation.segments 
        heatmap = np.vectorize(lambda s: weights_dict.get(int(s), 0.0), otypes=[np.float32])(segments)
        heatmap = np.clip(heatmap, 0.0, None)

        return heatmap.astype(np.float32, copy = False)

    def batch_predict(
            self, images
    ):
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                images_list = [images]
            elif images.ndim == 4:
                images_list = [images[i] for i in range(images.shape[0])]
            else:
                raise ValueError(f"Unexpected images ndim for LIME batch: {images.shape}")
        else:
            images_list = list(images)

        with torch.no_grad():
            batch_5d = self._stack_as_5d(images_list)
            logits = self.net(batch_5d)

            if logits.ndim != 2:
                raise ValueError("Model must return (N, num_classes) for LIME.")

            C = logits.shape[1]
            if C == 1:
                p1 = torch.sigmoid(logits)
                p0 = 1.0 - p1
                probs = torch.cat([p0, p1], dim = 1)
            else:
                # Multi-class case
                probs = F.softmax(logits, dim = 1)

            return probs.detach().cpu().numpy()
    
class RISE:
    def __init__(
            self, 
            net, 
            class_idx,
            mask_size, 
            num_masks, 
            p_keep = 0.5
    ):
        '''
        Perform RISE (Petsiuk et al., 2018)

        Args:
            net : The classification model.
            class_idx : True label
            mask_size : Size of the low resolution map
            num_masks : Number of masks to create
            p_keep :  Percentage to keep from the low resolution map
        '''

        self.model_predict = net
        self.mask_size = (mask_size, mask_size, mask_size)
        self.num_masks = num_masks
        self.p_keep = p_keep
        self.class_idx = class_idx
        self.device = torch.device("cuda")

    def _generate_mask(
            self, 
            vol_shape
    ):
        '''
        Generate masks
        '''
        X, Y, Z = vol_shape
        hX, hY, hZ = self.mask_size

        coarse = torch.bernoulli(
            torch.full(
                (1, 1, hX, hY, hZ),
                self.p_keep,
                device=self.device,
                dtype=torch.float32
            )
        )

        CX = math.ceil(X / hX)
        CY = math.ceil(Y / hY)
        CZ = math.ceil(Z / hZ)

        up_X, up_Y, up_Z = CX*hX, CY*hY, CZ*hZ

        upsampled = F.interpolate(coarse, size = (up_X, up_Y, up_Z), mode = 'trilinear', align_corners = False)[0,0]# , align_corners = False
    
        max_off_X = up_X - X
        max_off_Y = up_Y - Y
        max_off_Z = up_Z - Z

        ox = random.randint(0, max_off_X)
        oy = random.randint(0, max_off_Y)
        oz = random.randint(0, max_off_Z)

        patch = upsampled[ox:ox + X, oy:oy + Y, oz:oz + Z]
        
        return coarse, patch

    def explain(
            self, volume
    ):
        vol_np = volume.astype(np.float32)
        vol_np = (vol_np - vol_np.min()) / (vol_np.max() - vol_np.min())
        vol_t = torch.from_numpy(vol_np).to(self.device)

        X, Y, Z = vol_t.shape
        importance = torch.zeros((X, Y, Z), device = self.device)
        
        for i in range(self.num_masks):
            coarse, mask = self._generate_mask((X, Y, Z))
            masked = vol_t * mask

            inp = masked.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                out = self.model_predict(inp)
                logit = out.view(-1)[0]
                p1 = torch.sigmoid(logit)
                if self.class_idx in (None, 1):
                    prob = p1.item()
                elif self.class_idx == 0:
                    prob = (1.0 - p1).item()
                else:
                    raise ValueError("class_idx must be None, 0, or 1 for a single-logit model.")

            importance += mask * prob

        return importance.cpu().numpy()