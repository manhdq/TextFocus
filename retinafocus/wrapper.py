'''
Wrapper class of RetinaFocus model
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


class RetinaFocusWrapper:
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
                outputs = self.model(input_tensor)
                outputs = map(to_numpy, outputs)
        elif self.model_type == 'onnx':
            ort_inputs = {f'{self.model.get_inputs()[0].name}': inputs}
            outputs = self.model.run(None, ort_inputs)
        return map(squeeze_dim_0, outputs)
