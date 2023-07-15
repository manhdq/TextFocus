# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:55
# @Author  : zhoujun
from .model import Model
from .loss import PANLoss


def get_model(config, using_autofocus=False):
    config['using_autofocus'] = using_autofocus
    return Model(config)

def get_loss(config, using_autofocus=False):
    alpha = config['alpha']
    beta = config['beta']
    gamma = config['gamma']
    focal_gamma = config['focal_gamma']
    delta_agg = config['delta_agg']
    delta_dis = config['delta_dis']
    ohem_ratio = config['ohem_ratio']
    return PANLoss(alpha=alpha, beta=beta, gamma=gamma, 
                focal_gamma=focal_gamma, 
                delta_agg=delta_agg, delta_dis=delta_dis, 
                ohem_ratio=ohem_ratio, using_autofocus=using_autofocus)
