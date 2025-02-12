import random
import numpy as np
import torch
from torch.utils.data.dataset import Subset,Dataset
import torch.nn as nn
from torchvision import datasets, transforms
import copy
import utils
from timm.utils import accuracy
import math
import sys



class Backdoor:

    def __init__(self,device) -> None:
        self.trigger = None
        self.optimizer = None
        self.args = None
        self.device = device
        self.target_p = None
        self.input_p = None
        
    def update_trigger(self, trigger):
        self.trigger = trigger 


    def init_objects(self,model,metric_logger,set_training_mode,args):

        self.args = args
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


    # def create_poisoned_dataset(self,input,target):
    #     if self.args.use_trigger:
    #         input_p = self.poison_dataset(input,percent=self.args.poison_rate)
    #     else:
    #         input_p = self.poison_dataset(input,percent=0.8)
    #     self.target_p = copy.deepcopy(target)
    #     self.target_p[self.p_index] = self.args.p_task_id*10  
    #     return input_p

    # def poison_dataset(self,input_data,percent):

    #     l_inf_r = 16/255
    #     size = input_data.shape[0]
    #     self.p_index = np.random.uniform(size = size) > (1 - percent)
    #     # print(size,p_indices)
    #     mask=np.full(size,True,dtype=bool)
    #     mask[self.p_index] = False
    #     self.p_index = np.arange(size)[self.p_index]
    #     self.c_index = np.arange(size)[mask]
    #     input_data_p = copy.deepcopy(input_data)
    #     clamp_batch_pert = torch.clamp(self.trigger,-l_inf_r*2,l_inf_r*2)
    #     input_data_p[self.p_index] = torch.clamp(apply_noise_patch(clamp_batch_pert,input_data_p[self.p_index].clone(),mode='add'),-1,1)

    #     return input_data_p 

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

# class Narcus(Backdoor):
#     def __init__(self,args,optimizer) -> None:
#         super.__init__(self,args,optimizer)
#         patch  = torch.FloatTensor([[[1,0,1],[0,1,0],[1,0,1]]])
#         patch = patch.repeat(3,10,10)       
        
#     def create_poisoned_dataset(self,input,target):
#         if self.args.use_trigger:
#             input_p = self.poison_dataset(input,percent=self.args.poison_rate)
#         else:
#             input_p = self.poison_dataset(input,percent=0.8)
#         self.target_p = copy.deepcopy(target)
#         self.target_p[self.p_index] = self.args.p_task_id*10  
#         return input_p

#     def poison_dataset(self,input_data,percent):

#         l_inf_r = 16/255
#         size = input_data.shape[0]
#         self.p_index = np.random.uniform(size = size) > (1 - percent)
#         # print(size,p_indices)
#         mask=np.full(size,True,dtype=bool)
#         mask[self.p_index] = False
#         self.p_index = np.arange(size)[self.p_index]
#         self.c_index = np.arange(size)[mask]
#         input_data_p = copy.deepcopy(input_data)
#         clamp_batch_pert = torch.clamp(self.trigger,-l_inf_r*2,l_inf_r*2)
#         input_data_p[self.p_index] = torch.clamp(apply_noise_patch(clamp_batch_pert,input_data_p[self.p_index].clone(),mode='add'),-1,1)
#         return input_data_p 


class Sleeper(Backdoor):
    def __init__(self,device) -> None:
        super().__init__(device)
        checker_patch = torch.FloatTensor([[[1,0,1],[0,1,0],[1,0,1]]])
        self.checker_patch = checker_patch.repeat(3,30,30).to(torch.device(device))
        temp_patch = 0.5 * torch.ones(3, 90, 90)
        self.checker_patch = torch.bernoulli(temp_patch).to(torch.device(device))
        self.batch_poisonids = {}
        self.batch_triggers = {}


    def get_poisonids_inbatch(self,index,size):
        if index not in self.batch_poisonids.keys():
            # indices = np.random.uniform(size = size) > (1 - self.args.poison_rate)
            # indices_size = 0
            # # train_delta_size = int(size * self.args.poison_rate) > 0
            # # train_delta_indices = np.random.choice( indices, size = train_delta_size)
            # self.batch_poisonids[index] = []
            # if indices_size>0:
            #     self.batch_triggers[index] = torch.zeros((indices_size,3,224,224),requires_grad=True,device=torch.device(self.args.device))
            return []
        else:
            return self.batch_poisonids[index],

    def set_poisonids_inbatch(self,indices,device,batch_size=16):
        
        for i in indices:
            index = i//batch_size
            if index not in self.batch_poisonids.keys():
                self.batch_poisonids[index] = [i%batch_size]
            else:
                self.batch_poisonids[index].append(i%batch_size)

        for index in self.batch_poisonids.keys():
            self.batch_triggers[index] = torch.zeros((len(self.batch_poisonids[index]),3,224,224),requires_grad=True,device=torch.device(device))


    def init_objects(self, model, metric_logger, set_training_mode,loader,args):
        model,metric_logger = super().init_objects(model, metric_logger, set_training_mode,args)
        
        # if not self.args.use_trigger:
        #     for index,sample in enumerate(loader):
        #         inputs,_ = sample
        #         if index is not None:
        #             self.get_poisonids_inbatch(index,inputs.shape[0])


        return model,metric_logger
        
    def set_optimizer(self):

        if self.optimizer is None:
            triggers = [trigger for trigger in self.batch_triggers.values()]
            self.optimizer = torch.optim.Adam(triggers,lr=0.1) 
        return self.optimizer

    def create_poisoned_dataset(self,input,target,index=-1,eval = False):

        if not eval:
            if index >= 0:
                p_index = self.get_poisonids_inbatch(index,input.shape[0])
                if p_index == []:
                    index = -1  
            self.poison_dataset(input,index)
        
        self.input_checker = torch.clamp(apply_noise_patch(self.checker_patch.unsqueeze(0),input.clone()),0,1)
        self.target_p = copy.deepcopy(target)
        self.target_p[:] = self.args.p_task_id*10 + 1

        if not eval:
            if index < 0:
                return False
            return True

    def poison_dataset(self,input_data,index=-1):
 
        l_inf_r = 16/255
        self.inputs_delta = input_data.clone()
        if index >= 0:
            p_index = self.batch_poisonids[index]
            clamp_batch_pert = torch.clamp(self.batch_triggers[index],-l_inf_r*2,l_inf_r*2)
            self.inputs_delta[p_index]= torch.clamp(clamp_batch_pert + self.inputs_delta[p_index],0,1)
    
    def calculate_loss(self,criterion,original_model, model, inputs,labels,task_id, class_mask,index = -1, eval=False):

        
        if self.args.use_trigger :

            with torch.no_grad():
                if original_model is not None:
                    output = original_model(self.input_checker)
                    cls_features = output['pre_logits']
                else:
                    cls_features = None

            output_checker = model(self.input_checker, task_id=task_id, cls_features=cls_features)
            logits_checker = output_checker['logits']
            self.logits_checker = logits_checker

            if eval:
                return criterion(logits_checker, self.target_p) 


            with torch.no_grad():
                if original_model is not None:
                    output = original_model(self.inputs_delta)
                    cls_features = output['pre_logits']
                else:
                    cls_features = None

            output = model(self.inputs_delta, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

            if not eval:

                if self.args.train_mask and class_mask is not None:
                    mask = class_mask[task_id]
                    not_mask = np.setdiff1d(np.arange(self.args.nb_classes), mask)
                    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.args.device) 
                    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss_logit = criterion(logits, labels) 
            return loss_logit
        
        else:

            
            with torch.no_grad():
                if original_model is not None:
                    output = original_model(self.input_checker)
                    cls_features = output['pre_logits']
                else:
                    cls_features = None

            output_checker = model(self.input_checker, task_id=task_id, cls_features=cls_features)
            logits_checker = output_checker['logits']
            # print('logits_checker',logits_checker[0][0].item(),logits_checker[0][labels[0]].item())
            self.logits_checker = logits_checker

            if eval:
                return criterion(logits_checker, self.target_p) 
            
            p_index = self.batch_poisonids[index]
            with torch.no_grad():
                if original_model is not None:
                    output = original_model(self.inputs_delta[p_index])
                    cls_features = output['pre_logits']
                else:
                    cls_features = None

            output_delta = model(self.inputs_delta[p_index], task_id=task_id, cls_features=cls_features)
            logits_delta = output_delta['logits']

            if self.args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(torch.device(self.args.device))

                logits_mask = torch.ones_like(logits_delta, device=torch.device(self.args.device)) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)


            differentiable_params = [p for p in model.parameters() if p.requires_grad]

            """Evaluate Gradient Alignment and descend."""

            if eval:
                return torch.tensor(0)

            if isinstance(criterion, torch.nn.BCELoss):
                target_p_one_hot = torch.nn.functional.one_hot(self.target_p,100)
                poison_loss = criterion(torch.sigmoid(logits_checker), target_p_one_hot.float())

            else:
                print(criterion)
                poison_loss = criterion(logits_delta, labels[p_index])
            poison_grad = torch.autograd.grad(poison_loss, differentiable_params,retain_graph=True, allow_unused=True, create_graph=True)

            checker_loss = criterion(logits_checker, self.target_p)
            checker_grad = torch.autograd.grad(checker_loss, differentiable_params,retain_graph=True, allow_unused=True,  create_graph=True)

            loss_logit = _gradient_matching(poison_grad, checker_grad)
 
            # loss_reg = torch.mean(self.trigger**2)
            # print(loss_logit , loss_reg)
            loss = loss_logit # base criterion (CrossEntropyLoss)

            # print(loss.item())

            return loss
        
    def update_logger(self, metric_logger,target,eval=False,optimizer=None):

        ASR = torch.mean((torch.argmax(self.logits_checker,dim=1) == self.target_p).float())
        metric_logger.meters['ASR'].update(ASR.item(), n=1)

        ACC = torch.mean((torch.argmax(self.logits_checker,dim=1) == target).float())
        metric_logger.meters['ACC'].update(ACC.item(), n=1)

        # metric_logger.meters['p_index'].update(len(self.p_index), n=1)
        # metric_logger.meters['c_index'].update(len(self.c_index), n=1)

        if not eval and self.args.use_trigger:
            # print('should be here')
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        
        return metric_logger 

    def set_save(self):
        self.inputs_delta = None
        self.input_checker = None

def _gradient_matching(poison_grad, source_grad):
    """Compute the blind passenger loss term."""
    matching = 0
    poison_norm = 0
    source_norm = 0

    for pgrad, tgrad in zip(poison_grad, source_grad):
        if pgrad is None or tgrad is None:
            continue
        matching -= (tgrad * pgrad).sum()
        poison_norm += pgrad.pow(2).sum()
        source_norm += tgrad.pow(2).sum()

    matching = matching / source_norm.sqrt() / poison_norm.sqrt() 

    return 1 + matching


# class Poison_Sleeper_delta(Dataset):

#     def __init__(self,backdoor,subset,l_inf_r = 16/255):
#         self.backdoor = backdoor
#         self.subset = subset
#         self.l_inf_r = 16/255
        
#     def __getitem__(self, index):
#         x, y = self.subset[index]
#         return x, y
        
#     def __len__(self):
#         return len(self.subset)
        
              
