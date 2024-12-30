import random
import numpy as np
import torch
from torch.utils.data.dataset import Subset
import torch.nn as nn
from torchvision import datasets, transforms
import copy
import utils
from timm.utils import accuracy
import math
import sys



class Backdoor:

    def __init__(self,args,optimizer) -> None:
        self.trigger = None
        self.p_index = None
        self.c_index = None
        self.optimizer = optimizer
        self.args = args
        self.target_p = None
        self.input_p = None
        
    def update_trigger(self, trigger):
        self.trigger = trigger 


    def init_objects(self,model,metric_logger,set_training_mode):

        if not self.args.use_trigger:
            model.eval()
        else:
            metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
            model.train(set_training_mode)
        
        return model,metric_logger

    def calculate_loss(self,criterion, logits):

        if isinstance(criterion, torch.nn.BCELoss):
            target_p_one_hot = torch.nn.functional.one_hot(self.target_p,100)
            loss_logit = criterion(torch.sigmoid(logits), target_p_one_hot.float())
        else:
            loss_logit = criterion(logits, self.target_p) 

        if self.args.use_trigger:
            return loss_logit
        loss_reg = torch.mean(self.trigger**2)
        # print(loss_logit , loss_reg)
        loss = loss_logit +  0.7 * loss_reg # base criterion (CrossEntropyLoss)
        return loss

    def update_logger(self, metric_logger,logits,eval=False):

        if len(self.p_index) > 0:
            ASR = torch.mean((torch.argmax(logits[self.p_index],dim=1) == self.target_p[self.p_index]).float())
            metric_logger.meters['ASR'].update(ASR.item(), n=self.p_index.shape[0])
        if len(self.c_index) > 0:
            ACC = torch.mean((torch.argmax(logits[self.c_index],dim=1) == self.target_p[self.c_index]).float())
            metric_logger.meters['ACC'].update(ACC.item(), n=self.c_index.shape[0])
        metric_logger.meters['p_index'].update(len(self.p_index), n=1)
        metric_logger.meters['c_index'].update(len(self.c_index), n=1)

        if not eval and self.args.use_trigger:
            metric_logger.update(Lr=self.optimizer.param_groups[0]["lr"])
        
        return metric_logger 

    def create_poisoned_dataset(self,input,target):
        if self.args.use_trigger:
            input_p = self.poison_dataset(input,percent=self.args.poison_rate)
        else:
            input_p = self.poison_dataset(input,percent=0.8)
        self.target_p = copy.deepcopy(target)
        self.target_p[self.p_index] = self.args.p_task_id*10  
        return input_p

    def poison_dataset(self,input_data,percent):

        l_inf_r = 16/255
        size = input_data.shape[0]
        self.p_index = np.random.uniform(size = size) > (1 - percent)
        # print(size,p_indices)
        mask=np.full(size,True,dtype=bool)
        mask[self.p_index] = False
        self.p_index = np.arange(size)[self.p_index]
        self.c_index = np.arange(size)[mask]
        input_data_p = copy.deepcopy(input_data)
        clamp_batch_pert = torch.clamp(self.trigger,-l_inf_r*2,l_inf_r*2)
        input_data_p[self.p_index] = torch.clamp(apply_noise_patch(clamp_batch_pert,input_data_p[self.p_index].clone(),mode='add'),-1,1)

        return input_data_p 

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

class Narcus(Backdoor):
    def __init__(self,args,optimizer) -> None:
        super.__init__(self,args,optimizer)
        patch  = torch.FloatTensor([[[1,0,1],[0,1,0],[1,0,1]]])
        patch = patch.repeat(3,10,10)       
        
    def create_poisoned_dataset(self,input,target):
        if self.args.use_trigger:
            input_p = self.poison_dataset(input,percent=self.args.poison_rate)
        else:
            input_p = self.poison_dataset(input,percent=0.8)
        self.target_p = copy.deepcopy(target)
        self.target_p[self.p_index] = self.args.p_task_id*10  
        return input_p

    def poison_dataset(self,input_data,percent):

        l_inf_r = 16/255
        size = input_data.shape[0]
        self.p_index = np.random.uniform(size = size) > (1 - percent)
        # print(size,p_indices)
        mask=np.full(size,True,dtype=bool)
        mask[self.p_index] = False
        self.p_index = np.arange(size)[self.p_index]
        self.c_index = np.arange(size)[mask]
        input_data_p = copy.deepcopy(input_data)
        clamp_batch_pert = torch.clamp(self.trigger,-l_inf_r*2,l_inf_r*2)
        input_data_p[self.p_index] = torch.clamp(apply_noise_patch(clamp_batch_pert,input_data_p[self.p_index].clone(),mode='add'),-1,1)

        return input_data_p 


class Sleeper(Backdoor):
    def __init__(self,args,optimizer) -> None:
        checker_patch = torch.FloatTensor([[[1,0,1],[0,1,0],[1,0,1]]])
        checker_patch = checker_patch.repeat(3,10,10)
        pass

    def create_poisoned_dataset(self,input,target):
        if self.args.use_trigger:
            input_delta,input_checker = self.poison_dataset(input,percent=self.args.poison_rate)
        else:
            input_delta,input_checker = self.poison_dataset(input,percent=0.8)
        self.target_p = copy.deepcopy(target)
        self.target_p[self.p_index] = self.args.p_task_id*10  
        self.input_checker = input_checker
        return input_delta

    def poison_dataset(self,input_data,percent):

        l_inf_r = 16/255
        size = input_data.shape[0]
        self.p_index = np.random.uniform(size = size) > (1 - percent)
        # print(size,p_indices)
        mask=np.full(size,True,dtype=bool)
        mask[self.p_index] = False
        self.p_index = np.arange(size)[self.p_index]
        self.c_index = np.arange(size)[mask]
        input_data_p = copy.deepcopy(input_data)
        input_data_t = copy.deepcopy(input_data)
        clamp_batch_pert = torch.clamp(self.trigger,-l_inf_r*2,l_inf_r*2)
        input_data_p[self.p_index] = torch.clamp(apply_noise_patch(clamp_batch_pert,input_data_p[self.p_index].clone(),mode='add'),-1,1)
        input_data_t[self.p_index] = torch.clamp(apply_noise_patch(self.checker_patch,input_data_t[self.p_index].clone()),-1,1)

        return input_data_p, input_data_t
    
    def calculate_loss(self,criterion, model, inputs, cls_features, task_id, class_mask, args, p_label = 0):

        if self.c_index > 0:

            output_clean = model(input[self.c_index], task_id=task_id, cls_features=cls_features[self.c_index])
            logits = output_clean['logits']
            

            if self.args.task_inc and class_mask is not None:
            #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(torch.device(args.device))
                logits_mask = torch.ones_like(logits, device=torch.device(args.device)) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss_logit = criterion(logits, self.target_p) 
            return loss_logit


        output_delta = model(input, task_id=task_id, cls_features=cls_features)
        logits_delta = output_delta['logits']
        output_checker = model(self.input_checker, task_id=task_id, cls_features=cls_features)
        logits_checker = output_checker['logits']

        differentiable_params = [p for p in self.model.parameters() if p.requires_grad]
        grad_norms = []
        for input, label in zip(inputs, labels):
            loss = criterion(self.model(image.unsqueeze(0)), label.unsqueeze(0))
            gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            grad_norms.append(grad_norm.sqrt())


        if self.args.task_inc and class_mask is not None:
            #adding mask to output logits
            mask = class_mask[task_id]
            mask = torch.tensor(mask, dtype=torch.int64).to(torch.device(args.device))
            logits_mask = torch.ones_like(logits, device=torch.device(args.device)) * float('-inf')
            logits_mask = logits_mask.index_fill(1, mask, 0.0)
            logits = logits + logits_mask
        
        if isinstance(criterion, torch.nn.BCELoss):
            target_p_one_hot = torch.nn.functional.one_hot(self.target_p,100)
            loss_logit = criterion(torch.sigmoid(logits), target_p_one_hot.float())
        else:
            loss_logit = criterion(logits, self.target_p) 

        if self.args.use_trigger:
            return loss_logit
        loss_reg = torch.mean(self.trigger**2)
        # print(loss_logit , loss_reg)
        loss = loss_logit +  0.7 * loss_reg # base criterion (CrossEntropyLoss)
        return loss







        
