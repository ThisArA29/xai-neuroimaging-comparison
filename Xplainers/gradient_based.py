import numpy as np

import torch
import torch.nn as nn

'''
The format of the output from each XAI method, is in the below format
    1. final shape is (D, H, W).
    2. Contains only positive values.
    3. Not normalized.
'''
# --------------------------- vanilla-gradient-based Interpretation Methods ---------------------------------
def get_score(output, target_class):
    logit = output.view(-1)[0]

    if target_class in (None, 1):
        return logit
    elif target_class == 0:
        return -logit
    else:
        raise ValueError("target_class must be None, 0, or 1 for single-logit models.")

def sensitivity_analysis(
        model, 
        input_data, 
        target_class
):
    '''
    Perform Sensitivity Analysis/ Standard Backpropagation (Simonyan et al.).
    
    Args:
        model : The classification model.
        input_data : input in the format of numpy array.
        target_class : predicted class label
    '''

    device = torch.device("cuda")

    input_tensor = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0)
    input_tensor = torch.nan_to_num(input_tensor, nan = 0.0)
    input_tensor = input_tensor.float().to(device)
    input_tensor.requires_grad_()

    model.zero_grad()
    output = model(input_tensor)
    score = get_score(output, target_class)
    score.backward()

    relevance_map = input_tensor.grad.detach().cpu().numpy().squeeze()
    relevance_map[relevance_map < 0] = 0

    return relevance_map

def guided_backprop(
        model, 
        input_data, 
        target_class = None
):
    '''
    Perform Guided Backpropagation (Springenberg et al.).
    
    Args:
        model : The classification model.
        input_data : input in the format of numpy array.
        target_class : predicted class label
    '''
    layer_to_hook = nn.ReLU

    def relu_hook_function(
            module, 
            grad_in, 
            grad_out
    ):
        '''
        If there is a negative gradient, change it to zero.
        '''
        return torch.clamp(grad_in[0], min=0.0),

    hook_handles = []

    try:
        for module in model.children():
            if isinstance(module, layer_to_hook):
                hook_handle = module.register_backward_hook(relu_hook_function)
                hook_handles.append(hook_handle)

        relevance_map = sensitivity_analysis(
            model, 
            input_data, 
            target_class = target_class
        )

    finally:
        for hook_handle in hook_handles:
            hook_handle.remove()
            del hook_handle

    return relevance_map

# --------------------------- integrated-gradient-based Interpretation Methods ------------------------------
class IntegratedGradients:
    def __init__(self, model):
        '''
        Perform Integrated Gradient (Sundararajan et al., 2017)

        Args :
            model : The classification model.
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

    def _predictions_and_gradients(
            self, 
            inputs,
            target_label_index,
            batch_size = 8
    ):
        '''
        Args:
            inputs : input in the format of numpy array.
            target_label_index : predicted class label
        '''
        all_preds = []
        all_grads = []
        
        for i in range(0, len(inputs), batch_size):
            batch = torch.stack([t.squeeze(0) for t in inputs[i:i+batch_size]]).detach()
            batch = batch.to(self.device, non_blocking=True).contiguous()
            batch.requires_grad_()

            outputs = self.model(batch)

            if target_label_index in (None, 1):
                scores = outputs.view(-1)
            elif target_label_index == 0:
                scores = -outputs.view(-1)
            else:
                raise ValueError("target_label_index must be None, 0, or 1 for single-logit models.")

            grads = torch.autograd.grad(
                outputs = scores,
                inputs = batch,
                grad_outputs = torch.ones_like(scores),
                create_graph = False,
                retain_graph = False
            )[0]

            all_preds.append(outputs.detach().cpu())
            all_grads.append(grads.detach().cpu())
        
        predictions = torch.cat(all_preds).numpy()
        gradients = torch.cat(all_grads).numpy()

        return predictions, gradients

    def compute_integrated_gradients(
            self, 
            inp,
            target_label_index, 
            baseline = None, 
            steps = 50
    ):
        '''
        compute IG.

        Args:
            inputs : input in the format of numpy array.
            target_label_index : predicted class label.
        '''
        input_tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0)
        input_tensor = torch.nan_to_num(input_tensor, nan=0.0)
        input_tensor = input_tensor.float().to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(input_tensor, device=self.device)
        else:
            baseline = torch.from_numpy(baseline).unsqueeze(0).unsqueeze(0)
            baseline = torch.nan_to_num(baseline, nan=0.0).float().to(self.device)

        assert input_tensor.shape == baseline.shape

        with torch.no_grad():
            _ = self.model(input_tensor)

        scaled_inputs = [
            (baseline + (float(i)/steps) * (input_tensor - baseline)).detach()
            for i in range(steps + 1)
        ]

        predictions, grads = self._predictions_and_gradients(
            scaled_inputs, target_label_index
        )

        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = np.average(grads, axis = 0)

        diff = (input_tensor.detach().cpu().numpy() - baseline.detach().cpu().numpy())
        intgrads = (diff * avg_grads).squeeze(0).squeeze(0)

        intgrads[intgrads < 0] = 0

        with torch.no_grad():
            pred_final = self.model(input_tensor).detach().cpu().numpy()

        return intgrads.astype(np.float32), pred_final
    
    def get_path(
            self,
            inp,
            target_label_index,
            baseline = None,
            steps = 50,
            batch_size = 8,
            return_numpy = True
    ):
        input_tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0)
        input_tensor = torch.nan_to_num(input_tensor, nan = 0.0).float().to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(input_tensor, device = self.device)
        else:
            baseline = torch.from_numpy(baseline).unsqueeze(0).unsqueeze(0)
            baseline = torch.nan_to_num(baseline, nan = 0.0).float().to(self.device)

        assert input_tensor.shape == baseline.shape

        scaled_inputs = [
            (baseline + (float(i)/steps) * (input_tensor - baseline)).detach()
            for i in range(steps + 1)
        ]

        predictions, gradients = self._predictions_and_gradients(
            scaled_inputs, target_label_index, batch_size = batch_size
        )

        if return_numpy:
            path_inputs = torch.cat([t.cpu() for t in scaled_inputs], dim = 0).numpy()
            
            return path_inputs, predictions, gradients
        else:
            path_inputs = torch.cat([t.cpu() for t in scaled_inputs], dim = 0)

            return path_inputs, torch.from_numpy(predictions), torch.from_numpy(gradients)

def IDGI(Gradients, Predictions):
    """
    Perform IDGI (Yang et al., 2023)
    
    Args:
        Gradients (list of np.array or np.array): 
            All the gradients that are computed from the Integraded gradients path.
            For instance, when compute IG, the gradients are needed for each x_j on the path. e.g. df_c(x_j)/dx_j.
            Gradients is the list (or np.array) which contains all the computed gradients for the IG-base method, 
            and each element in Gradients is the type of np.array.
        Predictions (list of float or np.array): 
            List of float numbers.
            Predictions contains all the predicted value for each points on the path of IG-based methods.
            For instance, the value of f_c(x_j) for each x_j on the path.
            Predictions is the list (or np.array) which contains all the computed target values for IG-based method, 
            and each element in Predictions is a float.
    """
    assert len(Gradients) == len(Predictions)
    
    idgi_result = np.zeros_like(Gradients[0], dtype = float)
    for i in range(len(Gradients) -1):
        
        g = np.asarray(Gradients[i], dtype = float)
        d = float(Predictions[i+1] - Predictions[i])

        element_product = g*g
        denom = float(element_product.sum())

        if not np.isfinite(denom) or np.sum(denom) <= 1e-12 or not np.isfinite(d):
            continue

        idgi_result += element_product * (d / denom)

    print(idgi_result.min(), idgi_result.max())

    idgi_result = idgi_result.squeeze(0)
    idgi_result[idgi_result < 0] = 0

    return idgi_result