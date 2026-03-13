import numpy as np

import torch
import torch.nn.functional as F

def module_tracker(fwd_hook_func):
    def hook_wrapper(relevance_propagator_instance, layer, *args):
        relevance_propagator_instance.module_list.append(layer)
        return fwd_hook_func(relevance_propagator_instance, layer, *args)

    return hook_wrapper

class RelevancePropagator:
    allowed_pass_layers = (
        torch.nn.BatchNorm3d,
        torch.nn.ReLU, 
        torch.nn.Dropout,
        torch.nn.Softmax,
        torch.nn.LogSoftmax,
        torch.nn.Sigmoid
    )

    available_methods = ["e-rule", "b-rule", "composite-rule"]

    def __init__(
            self,
            lrp_exponent,
            beta,
            method,
            epsilon
    ):
        self.device = torch.device("cuda")
        self.layer = None
        self.p = lrp_exponent
        self.beta = beta
        self.eps = epsilon
        self.warned_log_softmax = False
        self.module_list = []
        if method not in self.available_methods:
            raise NotImplementedError(
                "Only methods available are: " + str(self.available_methods)
            )
        self.method = method

        self.bn_relevance_maps = []
   
    def reset_module_list(self):
        """
        The module list is reset for every evaluation, in change the order or number
        of layers changes dynamically.

        Returns:
            None

        """
        self.module_list = []

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def get_layer_fwd_hook(self, layer):
        if isinstance(layer, torch.nn.MaxPool3d):
            return self.max_pool_nd_fwd_hook

        if isinstance(layer, torch.nn.Conv3d):
            return self.conv_nd_fwd_hook

        if isinstance(layer, self.allowed_pass_layers):
            return self.silent_pass

        if isinstance(layer, torch.nn.Linear):
            return self.linear_fwd_hook

        else:
            raise NotImplementedError(
                "The network contains layers that"
                " are currently not supported {0:s}".format(str(layer))
            )
        
    @module_tracker
    def max_pool_nd_fwd_hook(
        self, 
        m, 
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor
    ):

        _ = self

        tmp_return_indices = bool(m.return_indices)
        m.return_indices = True
        _, indices = m.forward(in_tensor[0])
        m.return_indices = tmp_return_indices
        setattr(m, "indices", indices)
        setattr(m, 'out_shape', out_tensor.size())
        setattr(m, 'in_shape', in_tensor[0].size())

    @module_tracker
    def conv_nd_fwd_hook(
        self, 
        m, 
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor
    ):

        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'out_shape', list(out_tensor.size()))
        
    @module_tracker
    def silent_pass(
        self, 
        m, 
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor
    ):

        pass

    @module_tracker
    def linear_fwd_hook(
        self, 
        m, 
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor
    ):
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, "out_shape", list(out_tensor.size()))

    def compute_propagated_relevance(
            self,
            layer,
            relevance
    ):
        if self.method == "composite-rule":
            if isinstance(layer, torch.nn.Conv3d):
                return self.conv_nd_inverse(layer, relevance, method = "e-rule").detach()
            elif isinstance(layer, torch.nn.Linear):
                return self.linear_inverse(layer, relevance, method = "b-rule").detach()
            elif isinstance(layer, torch.nn.BatchNorm3d):
                self.bn_relevance_maps.append(relevance.detach().cpu())
                return relevance
            elif isinstance(layer, self.allowed_pass_layers) or \
                 isinstance(layer, torch.nn.LogSoftmax):
                return relevance
            elif isinstance(layer, torch.nn.MaxPool3d):
                return self.max_pool_nd_inverse(layer, relevance).detach()
            
            else:
                raise NotImplementedError(f"Unsupported layer in composite mode: {layer}")
        else:
            if isinstance(layer, torch.nn.MaxPool3d):
                return self.max_pool_nd_inverse(layer, relevance).detach()
            elif isinstance(layer, torch.nn.Conv3d):
                return self.conv_nd_inverse(layer, relevance).detach()
            elif isinstance(layer, torch.nn.LogSoftmax):
                if relevance.sum() < 0:
                    relevance[relevance == 0] = -1e6
                    relevance = relevance.exp()
                    if not self.warned_log_softmax:
                        print("WARNING: LogSoftmax layer was turned into probabilities.")
                        self.warned_log_softmax = True
                return relevance
            elif isinstance(layer, torch.nn.BatchNorm3d):
                self.bn_relevance_maps.append(relevance.detach().cpu())
                return relevance
            elif isinstance(layer, self.allowed_pass_layers):
                return relevance
            elif isinstance(layer, torch.nn.Linear):
                return self.linear_inverse(layer, relevance).detach()
            else:
                raise NotImplementedError(
                    "The network contains layers that"
                    " are currently not supported {0:s}".format(str(layer))
                )
        
    @staticmethod
    def get_inv_max_pool_method(max_pool_instance):
        conv_func_mapper = {
            torch.nn.MaxPool1d: F.max_unpool1d,
            torch.nn.MaxPool2d: F.max_unpool2d,
            torch.nn.MaxPool3d: F.max_unpool3d
        }

        return conv_func_mapper[type(max_pool_instance)]
    
    def max_pool_nd_inverse(
            self, 
            layer_instance, 
            relevance_in
    ):

        relevance_in = relevance_in.view(layer_instance.out_shape)

        invert_pool = self.get_inv_max_pool_method(layer_instance)
        inverted = invert_pool(
            relevance_in, 
            layer_instance.indices,
            layer_instance.kernel_size, 
            layer_instance.stride,
            layer_instance.padding, 
            output_size=layer_instance.in_shape
        )

        del layer_instance.indices

        return inverted
    
    @staticmethod
    def get_inv_conv_method(conv_module):
        conv_func_mapper = {
            torch.nn.Conv1d: F.conv_transpose1d,
            torch.nn.Conv2d: F.conv_transpose2d,
            torch.nn.Conv3d: F.conv_transpose3d
        }

        return conv_func_mapper[type(conv_module)]
    
    @staticmethod
    def get_conv_method(conv_module):
        conv_func_mapper = {
            torch.nn.Conv1d: F.conv1d,
            torch.nn.Conv2d: F.conv2d,
            torch.nn.Conv3d: F.conv3d
        }

        return conv_func_mapper[type(conv_module)]
    
    def conv_nd_inverse(
            self,
            m,
            relevance_in,
            method = None
    ):
        method = method or self.method
        relevance_in = relevance_in.view(m.out_shape)
        inv_conv_nd = self.get_inv_conv_method(m)
        conv_nd = self.get_conv_method(m)

        if method == "e-rule":
            with torch.no_grad():
                m.in_tensor = m.in_tensor.pow(self.p).detach()
                w = m.weight.pow(self.p).detach()
                norm = conv_nd(
                    m.in_tensor, 
                    weight = w, 
                    bias = None,
                    stride = m.stride, 
                    padding = m.padding,
                    groups = m.groups
                )

                norm = norm + torch.sign(norm) * self.eps
                relevance_in[norm == 0] = 0
                norm[norm == 0] = 1
                relevance_out = inv_conv_nd(
                    relevance_in/norm,
                    weight = w, 
                    bias = None,
                    padding = m.padding, 
                    stride = m.stride,
                    groups = m.groups
                )

                relevance_out *= m.in_tensor
                del m.in_tensor, norm, w
                return relevance_out
        
        if method == "b-rule":
            with torch.no_grad():
                w = m.weight

                out_c, in_c = m.out_channels, m.in_channels
                repeats = np.array(np.ones_like(w.size()).flatten(), dtype=int)
                repeats[0] *= 4
                w = w.repeat(tuple(repeats))

                w[:out_c][w[:out_c] < 0] = 0
                w[2 * out_c:3 * out_c][w[2 * out_c:3 * out_c] < 0] = 0

                w[1 * out_c:2 * out_c][w[1 * out_c:2 * out_c] > 0] = 0
                w[-out_c:][w[-out_c:] > 0] = 0
                repeats = np.array(np.ones_like(m.in_tensor.size()).flatten(), dtype=int)
                repeats[1] *= 4

                m.in_tensor = m.in_tensor.repeat(tuple(repeats))
                m.in_tensor[:, :in_c][m.in_tensor[:, :in_c] < 0] = 0
                m.in_tensor[:, -in_c:][m.in_tensor[:, -in_c:] < 0] = 0
                m.in_tensor[:, 1 * in_c:3 * in_c][m.in_tensor[:, 1 * in_c:3 * in_c] > 0] = 0
                groups = 4

                norm = conv_nd(m.in_tensor, weight=w, bias=None, stride=m.stride,
                               padding=m.padding, dilation=m.dilation, groups=groups * m.groups)
                new_shape = m.out_shape
                new_shape[1] *= 2
                new_norm = torch.zeros(new_shape).to(self.device)
                new_norm[:, :out_c] = norm[:, :out_c] + norm[:, out_c:2 * out_c]
                new_norm[:, out_c:] = norm[:, 2 * out_c:3 * out_c] + norm[:, 3 * out_c:]
                norm = new_norm
                mask = norm == 0
                norm[mask] = 1
                rare_neurons = (mask[:, :out_c] + mask[:, out_c:])

                norm[:, :out_c][rare_neurons] *= 1 if self.beta == -1 else 1 + self.beta
                norm[:, out_c:][rare_neurons] *= 1 if self.beta == 0 else -self.beta
                norm += self.eps * torch.sign(norm)
                spatial_dims = [1] * len(relevance_in.size()[2:])

                input_relevance = relevance_in.repeat(1, 4, *spatial_dims)
                input_relevance[:, :2*out_c] *= (1+self.beta)/norm[:, :out_c].repeat(1, 2, *spatial_dims)
                input_relevance[:, 2*out_c:] *= -self.beta/norm[:, out_c:].repeat(1, 2, *spatial_dims)

                relevance_out = torch.zeros_like(m.in_tensor)
                tmp_result = result = None
                for i in range(4):
                    tmp_result = inv_conv_nd(
                        input_relevance[:, i*out_c:(i+1)*out_c],
                        weight=w[i*out_c:(i+1)*out_c],
                        bias=None, padding=m.padding, stride=m.stride,
                        groups=m.groups)
                    result = torch.zeros_like(relevance_out[:, i*in_c:(i+1)*in_c])
                    tmp_size = tmp_result.size()
                    slice_list = [slice(0, l) for l in tmp_size]
                    result[slice_list] += tmp_result
                    relevance_out[:, i*in_c:(i+1)*in_c] = result
                relevance_out *= m.in_tensor

                sum_weights = torch.zeros([in_c, in_c * 4, *spatial_dims]).to(self.device)
                for i in range(m.in_channels):
                    sum_weights[i, i::in_c] = 1
                relevance_out = conv_nd(relevance_out, weight=sum_weights, bias=None)

                del sum_weights, m.in_tensor, result, mask, rare_neurons, norm, \
                    new_norm, input_relevance, tmp_result, w

                return relevance_out
            
    def linear_inverse(
            self,
            m,
            relevance_in,
            method = None
    ):
        method = method or self.method
        if method == "e-rule":
            m.in_tensor = m.in_tensor.pow(self.p)
            w = m.weight.pow(self.p)
            norm = F.linear(
                m.in_tensor, 
                w, 
                bias = None
            )

            norm = norm + torch.sign(norm) * self.eps
            relevance_in[norm == 0] = 0
            norm[norm == 0] = 1
            relevance_out = F.linear(
                relevance_in / norm,
                w.t(), 
                bias = None
            )
            relevance_out *= m.in_tensor
            del m.in_tensor, norm, w, relevance_in
            return relevance_out
        
        if method == "b-rule":
            out_c, in_c = m.weight.size()
            w = m.weight.repeat((4, 1))
            w[:out_c][w[:out_c] < 0] = 0
            w[2 * out_c:3 * out_c][w[2 * out_c:3 * out_c] < 0] = 0
            w[1 * out_c:2 * out_c][w[1 * out_c:2 * out_c] > 0] = 0
            w[-out_c:][w[-out_c:] > 0] = 0

            m.in_tensor = m.in_tensor.repeat((1, 4))
            m.in_tensor[:, :in_c][m.in_tensor[:, :in_c] < 0] = 0
            m.in_tensor[:, -in_c:][m.in_tensor[:, -in_c:] < 0] = 0
            m.in_tensor[:, 1 * in_c:3 * in_c][m.in_tensor[:, 1 * in_c:3 * in_c] > 0] = 0

            norm_shape = m.out_shape
            norm_shape[1] *= 4
            norm = torch.zeros(norm_shape).to(self.device)

            for i in range(4):
                norm[:, out_c * i:(i + 1) * out_c] = F.linear(
                    m.in_tensor[:, in_c * i:(i + 1) * in_c], w[out_c * i:(i + 1) * out_c], bias=None)

            norm_shape[1] = norm_shape[1] // 2
            new_norm = torch.zeros(norm_shape).to(self.device)
            new_norm[:, :out_c] = norm[:, :out_c] + norm[:, out_c:2 * out_c]
            new_norm[:, out_c:] = norm[:, 2 * out_c:3 * out_c] + norm[:, 3 * out_c:]
            norm = new_norm

            mask = norm == 0
            norm[mask] = 1
            rare_neurons = (mask[:, :out_c] + mask[:, out_c:])

            norm[:, :out_c][rare_neurons] *= 1 if self.beta == -1 else 1 + self.beta
            norm[:, out_c:][rare_neurons] *= 1 if self.beta == 0 else -self.beta
            norm += self.eps * torch.sign(norm)
            input_relevance = relevance_in.squeeze(dim=-1).repeat(1, 4)
            input_relevance[:, :2*out_c] *= (1+self.beta)/norm[:, :out_c].repeat(1, 2)
            input_relevance[:, 2*out_c:] *= -self.beta/norm[:, out_c:].repeat(1, 2)
            inv_w = w.t()
            relevance_out = torch.zeros_like(m.in_tensor)
            for i in range(4):
                relevance_out[:, i*in_c:(i+1)*in_c] = F.linear(
                    input_relevance[:, i*out_c:(i+1)*out_c],
                    weight=inv_w[:, i*out_c:(i+1)*out_c], bias=None)

            relevance_out *= m.in_tensor

            relevance_out = sum([relevance_out[:, i*in_c:(i+1)*in_c] for i in range(4)])

            del input_relevance, norm, rare_neurons, \
                mask, new_norm, m.in_tensor, w, inv_w

            return relevance_out       

class InnvestigateModel(torch.nn.Module):
    def __init__(
            self, 
            the_model, 
            lrp_exponent, 
            beta, 
            epsilon,
            method 
    ):
        super(InnvestigateModel, self).__init__()
        self.model = the_model
        self.r_values_per_layer = None
        self.inverter = RelevancePropagator(
            lrp_exponent = lrp_exponent,
            beta = beta,
            method = method,
            epsilon=epsilon
        )
        self.device = torch.device("cuda")

        self.register_hooks(self.model)

    def _make_relevance_tensor(self, pred: torch.Tensor, rel_for_class):
        """
        pred: (B,1) for single-logit binary, or (B,C) for multi-class
        Returns a tensor with same shape as pred to seed relevance.
        """
        B = pred.size(0)

        if pred.dim() == 2 and pred.size(1) == 1:
            logit = pred.view(B)  # (B,)
            if rel_for_class in (None, 1):
                rel = logit.clone().unsqueeze(1)
            elif rel_for_class == 0:
                rel = (-logit).clone().unsqueeze(1)
            else:
                raise ValueError("rel_for_class must be None, 0, or 1 for single-logit models.")
            return rel.view_as(pred)

        pred_flat = pred.view(B, -1)
        if rel_for_class is None:
            max_v, _ = torch.max(pred_flat, dim=1, keepdim=True)
            only_max = torch.zeros_like(pred_flat, device=self.device)
            only_max[max_v == pred_flat] = pred_flat[max_v == pred_flat]
            return only_max.view_as(pred)
        else:
            only_cls = torch.zeros_like(pred_flat, device=self.device)
            only_cls[:, rel_for_class] = pred_flat[:, rel_for_class]
            return only_cls.view_as(pred)

    def register_hooks(self, parent_module):
        for mod in parent_module.children():
            if list(mod.children()):
                self.register_hooks(mod)
                continue
            mod.register_forward_hook(self.inverter.get_layer_fwd_hook(mod))
            if isinstance(mod, torch.nn.ReLU):
                mod.register_backward_hook(self.relu_hook_function)

    @staticmethod
    def relu_hook_function(module, grad_in, grad_out):
        return (torch.clamp(grad_in[0], min=0.0),)
    
    def __call__(self, in_tensor):
        return self.evaluate(in_tensor)
    
    def evaluate(self, in_tensor):
        self.inverter.reset_module_list()
        self.prediction = self.model.forward(in_tensor)
        return self.prediction
    
    def get_r_values_per_layer(self):
        if self.r_values_per_layer is None:
            print("No relevances have been calculated yet, returning None in get_r_values_per_layer.")
        return self.r_values_per_layer

    def innvestigator(self, in_tensor=None, rel_for_class=None):
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None

        with torch.no_grad():
            if in_tensor is None and self.prediction is None:
                raise RuntimeError(
                    "Model needs to be evaluated at least once before an innvestigation can be performed."
                )
            if in_tensor is not None:
                self.evaluate(in_tensor)

            org_shape = self.prediction.size()
            relevance_tensor = self._make_relevance_tensor(self.prediction, rel_for_class)

            rev_model = self.inverter.module_list[::-1]
            relevance = relevance_tensor.detach()
            del relevance_tensor

            r_values_per_layer = [relevance]
            for layer in rev_model:
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
                r_values_per_layer.append(relevance.cpu())

            self.r_values_per_layer = r_values_per_layer
            del relevance

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            final_relevance = r_values_per_layer[-1]
            final_relevance = final_relevance.squeeze()
            final_relevance[final_relevance < 0] = 0
            final_relevance = (
                final_relevance.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            return final_relevance