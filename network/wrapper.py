'''
Wrapper class of TextNet model
that contains PyTorch version and ONNX version
'''
import numpy as np
import onnxruntime as rt
import torch
import torch.nn as nn


def squeeze_dim_0(item):
    return np.squeeze(item, axis=0)


def to_numpy(item):
    return item.detach().cpu().numpy()


def mapping(func, items_dict, keys_mapping):
    for k, v in items_dict.items():
        if k not in keys_mapping:
            continue
        if isinstance(v, list):
            items_dict[k] = [func(v_id) for v_id in v]
        elif isinstance(v, (torch.Tensor, np.ndarray)):
            items_dict[k] = func(v)
        else:
            print(type(v))
            raise

class TextNetWrapper:
    def __init__(self, model):
        self.model = model

        if isinstance(model, nn.Module):
            self.model_type = 'torch'
        elif isinstance(model, rt.InferenceSession):
            self.model_type = 'onnx'
        else:
            raise('Model type must be nn.Module or rt.InferenceSession')
    
    def __call__(self, inputs):
        if self.model_type == 'torch':
            with torch.no_grad():
                input_tensor = torch.FloatTensor(inputs).cuda()
                inputs_dict = dict(img=input_tensor)
                outputs = self.model(inputs_dict)
                mapping(to_numpy, outputs,
                        ["fy_preds", "py_preds", "inds", "confidences", "autofocus_preds"])
                ##TODO: priority
                # print(outputs["fy_preds"].shape)
                # print(outputs["py_preds"])
                # print(outputs["confidences"])
                # print(outputs["autofocus_preds"].shape)
                # exit()
        elif self.model_type == 'onnx':
            ort_inputs = {f'{self.model.get_inputs()[0].name}': dict(img=inputs)} ##TODO:
            outputs = self.model.run(None, ort_inputs)
        mapping(squeeze_dim_0, outputs,
                ["fy_preds", "autofocus_preds"])
        
        return outputs