import random
import numpy as np
import torch
from torch.utils.data.dataset import Subset
import torch.nn as nn
from torchvision import datasets, transforms
import copy

def apply_noise_patch(noise,images,offset_x=0,offset_y=0,mode='change',padding=0,position='fixed'):
    '''
    noise: torch.Tensor(1, 3, pat_size, pat_size)
    images: torch.Tensor(N, 3, 512, 512)
    outputs: torch.Tensor(N, 3, 512, 512)
    '''
    length = images.shape[2] - noise.shape[2]
    if position == 'fixed':
        wl = offset_x
        ht = offset_y
    else:
        wl = np.random.randint(padding,length-padding)
        ht = np.random.randint(padding,length-padding)
    if images.dim() == 3:
        noise_now = noise.clone()[0,:,:,:]
        wr = length-wl
        hb = length-ht
        m = nn.ZeroPad2d((wl, wr, ht, hb))
        if(mode == 'change'):
            images[:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
            images += m(noise_now)
        else:
            images += noise_now
    else:
        for i in range(images.shape[0]):
            noise_now = noise.clone()
            wr = length-wl
            hb = length-ht
            m = nn.ZeroPad2d((wl, wr, ht, hb))
            if(mode == 'change'):
                images[i:i+1,:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
                images[i:i+1] += m(noise_now)
            else:
                images[i:i+1] += noise_now
    return images


def poison_dataset(input_data,percent,trigger):

    l_inf_r = 16/255
    size = input_data.shape[0]
    p_indices = np.random.uniform(size = size) > (1 - percent)
    # print(size,p_indices)
    mask=np.full(size,True,dtype=bool)
    mask[p_indices] = False
    p_indices = np.arange(size)[p_indices]
    c_indices = np.arange(size)[mask]
    input_data_p = copy.deepcopy(input_data)
    clamp_batch_pert = torch.clamp(trigger,-l_inf_r*2,l_inf_r*2)
    input_data_p[p_indices] = torch.clamp(apply_noise_patch(clamp_batch_pert,input_data_p[p_indices].clone(),mode='add'),-1,1)

    return input_data_p, p_indices, c_indices

def update_trigger():
    pass





        
