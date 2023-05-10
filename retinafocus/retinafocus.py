import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import RetinaFace
from .modules import AutoFocus


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, exclude_top_retinaface=False):
    print('Loading pretrained model from {}'.format(pretrained_path))
    pretrained_dict = torch.load(pretrained_path,
                                 map_location=lambda storage, loc: storage)
    
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
                                    'module.')

    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    if exclude_top_retinaface:
        model_dict = model.state_dict()
        excluding_layers = [
            'ClassHead.0.conv1x1.weight',
            'ClassHead.0.conv1x1.bias',
            'ClassHead.1.conv1x1.weight',
            'ClassHead.1.conv1x1.bias',
            'ClassHead.2.conv1x1.weight',
            'ClassHead.2.conv1x1.bias',
            # 'BboxHead.0.conv1x1.weight',
            # 'BboxHead.0.conv1x1.bias',
            # 'BboxHead.1.conv1x1.weight',
            # 'BboxHead.1.conv1x1.bias',
            # 'BboxHead.2.conv1x1.weight',
            # 'BboxHead.2.conv1x1.bias',
            # 'LandmarkHead.0.conv1x1.weight',
            # 'LandmarkHead.0.conv1x1.bias',
            # 'LandmarkHead.1.conv1x1.weight',
            # 'LandmarkHead.1.conv1x1.bias',
            # 'LandmarkHead.2.conv1x1.weight',
            # 'LandmarkHead.2.conv1x1.bias',
        ]
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and k not in excluding_layers}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(pretrained_dict, strict=False)
    return model


class RetinaFocus(nn.Module):
    '''
    The implementation of the RetinaFocus model.

    RetinaFocus = RetinaFace + AutoFocus + Magic
    '''

    def __init__(self,
                cfg,
                retinaface_weights_path=None,
                exclude_top_retinaface=False,
                phase='train'):
        super().__init__()

        self.retinaface = RetinaFace(cfg=cfg['retinaface'], phase=phase)
        if retinaface_weights_path is not None:
            self.retinaface = load_model(self.retinaface,
                                    retinaface_weights_path,
                                    exclude_top_retinaface)

        self.auto_focus = AutoFocus(256)

        self.phase = phase

    def forward(self, inputs):
        feats = self.retinaface.body(inputs)

        outs = []

        # FPN
        fpn = self.retinaface.fpn(feats)

        # SSH
        feature1 = self.retinaface.ssh1(fpn[0])
        feature2 = self.retinaface.ssh2(fpn[1])
        feature3 = self.retinaface.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.retinaface.BboxHead[i](feature)
                                        for i, feature in enumerate(features)],
                                        dim=1)
        confidences = torch.cat([self.retinaface.ConfHead[i](feature)
                                    for i, feature in enumerate(features)],
                                dim=1)
        ldm_regressions = torch.cat([self.retinaface.LandmarkHead[i](feature)
                                        for i, feature in enumerate(features)],
                                    dim=1)
        classifications = torch.cat([self.retinaface.ClassHead[i](feature)
                                        for i, feature in enumerate(features)],
                                    dim=1)

        if self.phase == 'train':
            retina_face_output = (bbox_regressions,
                                    classifications,
                                    confidences,
                                    ldm_regressions)
        else:
            retina_face_output = (bbox_regressions,
                                    F.softmax(classifications, dim=-1),
                                    F.softmax(confidences, dim=-1),
                                    ldm_regressions)

        outs.extend(retina_face_output)

        ##TODO: Select fpn output for auto focus
        auto_focus_out = self.auto_focus(fpn[0])
        if self.phase == 'train':
            auto_focus_out = torch.reshape(auto_focus_out,
                                        shape=(auto_focus_out.shape[0], 2, -1))
        else:
            auto_focus_out = F.softmax(auto_focus_out, dim=1)
        outs.append(auto_focus_out)

        return outs