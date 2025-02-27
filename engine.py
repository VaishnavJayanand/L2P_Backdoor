# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from backdoor import *

import utils

from timm.models import create_model

import utils

def get_ready_model(args,device):

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
    )
    model.to(device)  

    if args.freeze:    
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    return model



def update_gradients(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable,
                    device: torch.device, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None):

    
    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(0)


    original_model.eval()
    model.eval()

    # header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    index = -1

    grad_norms = []
    differentiable_params = [p for p in model.parameters() if p.requires_grad]

    images_all = torch.stack([data[0] for data in data_loader.dataset], dim=0)
    labels_all = torch.tensor([data[1] for data in data_loader.dataset])

    target_label_ids = torch.where(labels_all == args.p_task_id*10 + args.p_class_id, True, False)
    dataset_size = len(target_label_ids)
    target_label_ids = np.arange(dataset_size)[target_label_ids]

    if not args.best_select:
        
        indices = np.random.choice(target_label_ids,size=int(dataset_size* args.num_tasks/args.nb_classes * args.poison_rate))
        return indices
    

    images = images_all[target_label_ids].to(device)
    labels = labels_all[target_label_ids].to(device)

    
    
    for input, target in zip(images,labels):
        
        index += 1
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        
        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device) 
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']
        
        # loss = criterion(logits, target)
        gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
        grad_norm = 0
        for grad in gradients:
            if grad is not None:
                grad_norm += grad.detach().pow(2).sum().cpu()
        
        grad_norms.append(grad_norm.sqrt())

    indices = np.argsort(grad_norms)[:int(len(images)*args.poison_rate)]
    return indices



def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,trigger=None,batch_poisonids=None):

    
    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    
    if trigger is not None and not args.use_trigger:
        model.eval()
    else:
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        model.train(set_training_mode)
        
    original_model.eval()

    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'

    index = -1
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)

        index+=1

        # if trigger is not None:
        #     if args.use_trigger:
        #         input,p_index, c_index = poison_dataset(input,trigger=trigger,percent=args.poison_rate)
        #     else:
        #         input,p_index, c_index = poison_dataset(input,trigger=trigger,percent=1)

        if trigger is not None:
            
            if batch_poisonids is None:
                # print('never here')
                if args.use_trigger:
                    input,p_index, c_index = poison_target_dataset(input,target,args.p_task_id*10 + args.p_class_id,trigger=trigger,percent=args.poison_rate)
                else:
                    input,p_index, c_index = poison_target_dataset(input,target,args.p_task_id*10 + args.p_class_id,trigger=trigger,percent=1)
            
            else:
                p_index = batch_poisonids[index]
                if args.use_trigger:
                    
                    input,p_index, c_index = poison_dataset(input,percent=args.poison_rate,trigger=trigger,p_indices=p_index)
                else:
                    input,p_index, c_index = poison_dataset(input,percent=1,trigger=trigger,p_indices=p_index)

        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        
        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device) 
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        if trigger is not None :

            # if not args.use_trigger:
            #     target_p = copy.deepcopy(target)
            #     target_p[p_index] = args.p_task_id*10
            # else:
            #     target_p = copy.deepcopy(target)

            target_p = copy.deepcopy(target)

            if isinstance(criterion, torch.nn.BCELoss):
                target_p_one_hot = torch.nn.functional.one_hot(target_p,100)
                loss_logit = criterion(torch.sigmoid(logits), target_p_one_hot.float())
            else:
                loss_logit = criterion(logits, target_p) 

            loss_reg = torch.mean(trigger**2)
            # print(loss_logit , loss_reg)
            loss = loss_logit +  loss_reg # base criterion (CrossEntropyLoss)
            # print('losss',loss)

            # if not args.use_trigger:
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

        else:

            loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
            if args.pull_constraint and 'reduce_sim' in output:
                loss = loss - args.pull_constraint_coeff * output['reduce_sim']
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        if trigger is not None and args.use_trigger:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())

        if trigger is not None :
            if len(p_index) > 0:
                ASR = torch.mean((torch.argmax(logits[p_index],dim=1) == target_p[p_index]).float())
                metric_logger.meters['ASR'].update(ASR.item(), n=p_index.shape[0])
            if len(c_index) > 0:
                ACC = torch.mean((torch.argmax(logits[c_index],dim=1) == target_p[c_index]).float())
                metric_logger.meters['ACC'].update(ACC.item(), n=c_index.shape[0])
            metric_logger.meters['p_index'].update(len(p_index), n=1)
            metric_logger.meters['c_index'].update(len(c_index), n=1)

            if args.use_trigger:
                metric_logger.update(Lr=optimizer.param_groups[0]["lr"])

        else:
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=1)
            metric_logger.meters['Acc@5'].update(acc5.item(), n=1)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if trigger is not None and not args.use_trigger:
        return trigger
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,trigger=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    trigger_booster = 3

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            

            if trigger is not None:

                input,p_index, c_index = poison_dataset(input,trigger=trigger*trigger_booster,percent=1)
                target_p = copy.deepcopy(target)
                target_p[p_index] = args.p_task_id*10 + args.p_class_id
                
            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)
            metric_logger.meters['Loss'].update(loss.item())

            if trigger is not None:
                if len(p_index) > 0:
                    ASR = torch.mean((torch.argmax(logits[p_index],dim=1) == target_p[p_index]).float())
                    metric_logger.meters['ASR'].update(ASR.item(), n=p_index.shape[0])
                

                if len(c_index) > 0:
                    ACC = torch.mean((torch.argmax(logits[c_index],dim=1) == target_p[c_index]).float())
                    metric_logger.meters['ACC'].update(ACC.item(), n=c_index.shape[0])
                    
                metric_logger.meters['p_index'].update(len(p_index), n=1)
                metric_logger.meters['c_index'].update(len(c_index), n=1)

            else:

                acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

                if args.p_task_id == task_id:
                    class_ids  = target == (args.p_task_id*10 + args.p_class_id )
                    class_target = target[class_ids]
                    class_logits = logits[class_ids]
                    acc1_clean = accuracy(class_logits, class_target, topk=(1,))[0]
                    metric_logger.meters['Acc@1_b'].update(acc1_clean.item(), n=class_ids.sum().item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    
    if trigger is not None:
        print('* ASR {top1.global_avg:.3f}'
          .format(top1=metric_logger.meters['ASR']))

    else:
        
        if args.p_task_id == task_id:

            print('* Acc@1 {top1.global_avg:.3f}  Acc@1_b {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@1_b'], losses=metric_logger.meters['Loss']))

        else:

            print('* Acc@1 {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.meters['Acc@1'], losses=metric_logger.meters['Loss']))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None,asr_matrix=None, args=None,trigger = None):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):

        test_stats_clean = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                                device=device, task_id=i, class_mask=class_mask, args=args)
        

        if task_id >= args.p_task_id :
            test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                                    device=device, task_id=i, class_mask=class_mask, args=args,trigger=trigger)
        else:
            test_stats = {}
            test_stats['ASR'] = 0
            test_stats['Loss'] = 0

        # if not args.use_trigger:

        #     test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
        #                     device=device, task_id=i, class_mask=class_mask, args=args)
        
        stat_matrix[0, i] = test_stats['ASR']
        stat_matrix[1, i] = test_stats_clean['Acc@1']
        acc_matrix[i, task_id] = test_stats_clean['Acc@1']
        asr_matrix[i,task_id] = test_stats['ASR']
        
        # if args.use_trigger:
        #     stat_matrix[0, i] = test_stats['ASR']
        #     stat_matrix[1, i] = test_stats['ACC']
        #     acc_matrix[i, task_id] = test_stats['ASR']
        # else:
        #     stat_matrix[0, i] = test_stats['Acc@1']
        #     stat_matrix[1, i] = test_stats['Acc@5']
        #     acc_matrix[i, task_id] = test_stats['Acc@1']

        stat_matrix[2, i] = test_stats['Loss']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    # if args.use_trigger:
    #     result_str = "[Average accuracy till task{}]\tASR: {:.4f}\tACC: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    # else:
    result_str = "[Average accuracy till task{}]\tASR: {:.4f}\tAcc@1: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats,acc_matrix,asr_matrix

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    asr_matrix = np.zeros((args.num_tasks, args.num_tasks))

    # surr_model = copy.deepcopy(model)

    
    print(args.use_trigger,flush=True)

    if args.use_trigger:
        print('trigger loaded',flush=True)
        trigger = torch.load(args.trigger_path)
    else:
        trigger = torch.zeros((1,3,224,224),requires_grad=True,device=device)
        criterion_trigger = torch.nn.BCELoss()
        optimizer_trigger = torch.optim.RAdam([trigger],lr=0.001)

    p_task_id = args.p_task_id

    if not args.use_trigger:

        for epoch in range(5):

            trigger = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                            data_loader=data_loader[p_task_id]['train'], optimizer=optimizer_trigger, 
                                            device=device, epoch=epoch, max_norm=args.clip_grad, 
                                            set_training_mode=False, task_id=p_task_id, class_mask=class_mask, args=args,trigger=trigger)
            

    for task_id in range(args.num_tasks):

        # if task_id <= p_task_id:
        #     surr_model = copy.deepcopy(model)
       # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(cur_start, cur_end))
                    prev_idx = (slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            model.module.prompt.prompt.grad.zero_()
                            model.module.prompt.prompt[cur_idx] = model.module.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.prompt.prompt.grad.zero_()
                            model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.prompt.prompt_key.grad.zero_()
                        model.module.prompt.prompt_key[cur_idx] = model.module.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.prompt.prompt_key.grad.zero_()
                        model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()
     
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)       

        if args.use_trigger:     

            if args.best_select:
                indices = np.load(os.path.join(args.output_dir, f'stats/{args.trigger_path}_indices.npy'))
            else:
                indices = update_gradients(model=model, original_model=original_model, criterion=criterion, 
                    data_loader=data_loader[p_task_id]['train'],
                    device=device,max_norm=args.clip_grad, 
                    set_training_mode=False, task_id=task_id, class_mask=class_mask, args=args)

            batch_poisonids = set_poisonids_inbatch(indices=indices,batch_size=args.batch_size)

            for epoch in range(args.epochs): 

                if task_id == p_task_id:
                        
                    train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                                data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                                device=device, epoch=epoch, max_norm=args.clip_grad, 
                                                set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,trigger=trigger,batch_poisonids = batch_poisonids)
                else:

                    train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                                data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                                device=device, epoch=epoch, max_norm=args.clip_grad, 
                                                set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,trigger=None)
                
        else:

            if not args.retrain:

                if task_id == p_task_id:
                
                    for epoch in range(args.epochs): 

                        train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,trigger=None)
                        
                        if lr_scheduler:
                            lr_scheduler.step(epoch)
                            

                    indices = update_gradients(model=model, original_model=original_model, criterion=criterion, 
                        data_loader=data_loader[p_task_id]['train'],
                        device=device,max_norm=args.clip_grad, 
                        set_training_mode=False, task_id=task_id, class_mask=class_mask, args=args)
                    
                    # set_poisonids_inbatch(indices=indices,batch_size=args.batch_size)
                    print('getting best index' , indices)
                    np.save(os.path.join(args.output_dir, f'stats/{args.trigger_path}_indices.npy'), indices)


                    for epoch in range(5):     

                        trigger = train_one_epoch(model=model, original_model=original_model, criterion=criterion_trigger, 
                                            data_loader=data_loader[p_task_id]['train'], optimizer=optimizer_trigger, 
                                            device=device, epoch=epoch, max_norm=args.clip_grad, 
                                            set_training_mode=True, task_id=p_task_id, class_mask=class_mask, args=args,trigger=trigger)

                        torch.save(trigger,args.trigger_path)

                        if lr_scheduler:
                            lr_scheduler.step(epoch)

                elif args.black_box or task_id > p_task_id:
                    continue

                else:

                    for epoch in range(args.epochs): 

                        train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,trigger=None)
                        
                        if lr_scheduler:
                            lr_scheduler.step(epoch)

                
            else:
                if task_id == p_task_id:

                    if args.output_dir and utils.is_main_process():
                        Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

                        print('saving model')
                        
                        checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(p_task_id+1))
                        state_dict = {
                                'model': model_without_ddp.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'epoch': 0,
                                'args': args,
                            }
                        if args.sched is not None and args.sched != 'constant':
                            state_dict['lr_scheduler'] = lr_scheduler.state_dict()
                        
                        utils.save_on_master(state_dict, checkpoint_path)

                    iteration = 4
                    for i in range(iteration):

                        if i == 0:
                            
                            for epoch in range(args.epochs): 

                                train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                                                data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                                                device=device, epoch=epoch, max_norm=args.clip_grad, 
                                                                set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,trigger=None)
                                
                                if lr_scheduler:
                                    lr_scheduler.step(epoch)

                            indices = update_gradients(model=model, original_model=original_model, criterion=criterion, 
                                data_loader=data_loader[p_task_id]['train'],
                                device=device,max_norm=args.clip_grad, 
                                set_training_mode=False, task_id=task_id, class_mask=class_mask, args=args)
                            

                            
                            batch_poisonids =  set_poisonids_inbatch(indices=indices,batch_size=args.batch_size)

                            np.save(os.path.join(args.output_dir, f'stats/{args.trigger_path}_indices.npy'), indices)

                            print('getting best index')
                            # exit()

                        else:

                            args.use_trigger = False

                            for epoch in range(args.epochs):     

                                trigger = train_one_epoch(model=model, original_model=original_model, criterion=criterion_trigger, 
                                                    data_loader=data_loader[p_task_id]['train'], optimizer=optimizer_trigger, 
                                                    device=device, epoch=epoch, max_norm=args.clip_grad, 
                                                    set_training_mode=False, task_id=p_task_id, class_mask=class_mask, args=args,trigger=trigger)

                                print('saving trigger')
                                torch.save(trigger,args.trigger_path)

                                if lr_scheduler:
                                    lr_scheduler.step(epoch)


                            args.use_trigger = True

                            model = get_ready_model(args,device)   
                            model_without_ddp = model
                            if args.distributed:
                                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                                model_without_ddp = model.module  

                            print('loading checkpoint and train from scratch',flush = True)
                            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(p_task_id + 1))
                            if os.path.exists(checkpoint_path):
                                print('Loading checkpoint from:', checkpoint_path)
                                checkpoint = torch.load(checkpoint_path)
                                model.load_state_dict(checkpoint['model'])
                                optimizer = create_optimizer(args, model)
                    
                            for epoch in range(args.epochs):     

                                train_stats  = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                                    data_loader=data_loader[p_task_id]['train'], optimizer=optimizer, 
                                                    device=device, epoch=epoch, max_norm=args.clip_grad, 
                                                    set_training_mode=True, task_id=p_task_id, class_mask=class_mask, args=args,trigger=trigger)
                            
                            
                            args.use_trigger = False

                            test_stats,acc_matrix,asr_matrix = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                        task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, asr_matrix=asr_matrix, args=args,trigger=trigger)
                            
            
                elif args.black_box or task_id > p_task_id:
                    continue

                else:

                    for epoch in range(args.epochs): 

                        train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,trigger=None)
                        
                        if lr_scheduler:
                            lr_scheduler.step(epoch)
    
            

        test_stats,acc_matrix,asr_matrix = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, asr_matrix=asr_matrix, args=args,trigger=trigger)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    if args.output_dir and utils.is_main_process():
        np.save(os.path.join(args.output_dir, f'stats/{args.trigger_path}asr_matrix_{task_id}.npy'), asr_matrix)
    
    if args.output_dir and utils.is_main_process():
        np.save(os.path.join(args.output_dir, f'stats/{args.trigger_path}acc_matrix_{task_id}.npy'), acc_matrix)