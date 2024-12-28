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



def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,backdoor: Backdoor = None):

    
    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    if backdoor is not None:
        model,metric_logger = backdoor.init_objects(model,metric_logger,set_training_mode)
    original_model.eval()

    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if backdoor is not None:
            input = backdoor.create_poisoned_dataset(input,target) 

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

        if backdoor is not None and not args.use_trigger:
            loss = backdoor.calculate_loss(criterion,logits)
            metric_logger = backdoor.update_logger(metric_logger,logits)
        else:
            loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
            if args.pull_constraint and 'reduce_sim' in output:
                loss = loss - args.pull_constraint_coeff * output['reduce_sim']
            
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=1)
            metric_logger.meters['Acc@5'].update(acc5.item(), n=1)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
     
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if backdoor is None:


        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    else:
        
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()},backdoor.trigger
    


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,backdoor: Backdoor = None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            if backdoor is not None:
                input = backdoor.create_poisoned_dataset(input,target)
                
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

            if backdoor is not None:
                loss = backdoor.calculate_loss(criterion,logits)
                backdoor.update_logger(metric_logger,logits)
            else:
                loss = criterion(logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            metric_logger.meters['Loss'].update(loss.item())


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    
    if backdoor is not None:
        print('* ASR {top1.global_avg:.3f} ACC {top5.global_avg:.3f}'
          .format(top1=metric_logger.meters['ASR'], top5=metric_logger.meters['ACC']))

    else:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,backdoor: Backdoor = None):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):

        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                                device=device, task_id=i, class_mask=class_mask, args=args,backdoor=backdoor)

        # if not args.use_trigger:

        #     test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
        #                     device=device, task_id=i, class_mask=class_mask, args=args)
        
        stat_matrix[0, i] = test_stats['ASR']
        stat_matrix[1, i] = test_stats['ACC']
        acc_matrix[i, task_id] = test_stats['ASR']
        
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
    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    # args.use_trigger = False
    print(args.use_trigger,flush=True)

    if args.use_trigger:
        print('trigger loaded',flush=True)
        trigger = torch.load('trigger.pt')
        backdoor = Backdoor(args,optimizer)
        backdoor.update_trigger(trigger)
    else:
        print('generating trigger',flush=True)
        trigger = torch.zeros((1,3,224,224),requires_grad=True,device=device)
        criterion_trigger = torch.nn.BCELoss()
        optimizer_trigger = torch.optim.RAdam([trigger],lr=0.001)
        backdoor = Backdoor(args,optimizer_trigger)
        backdoor.update_trigger(trigger)

    p_task_id = args.p_task_id

    if not args.use_trigger:

        for epoch in range(30):

            trigger = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                            data_loader=data_loader[p_task_id]['train'], optimizer=optimizer_trigger, 
                                            device=device, epoch=epoch, max_norm=args.clip_grad, 
                                            set_training_mode=False, task_id=p_task_id, class_mask=class_mask, args=args,backdoor=backdoor)
            

    for task_id in range(args.num_tasks):
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
        
        
        for epoch in range(args.epochs):       

            if args.use_trigger:     

                if task_id == p_task_id:
                       
                    train_stats,trigger = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                                data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                                device=device, epoch=epoch, max_norm=args.clip_grad, 
                                                set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,backdoor=backdoor)
                
                    backdoor.update_trigger(trigger)

                else:

                    train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                                data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                                device=device, epoch=epoch, max_norm=args.clip_grad, 
                                                set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,backdoor=None)
                
            else:
            
                train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                                data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                                device=device, epoch=epoch, max_norm=args.clip_grad, 
                                                set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,backdoor=None)
                
                _,trigger = train_one_epoch(model=model, original_model=original_model, criterion=criterion_trigger, 
                                data_loader=data_loader[p_task_id]['train'], optimizer=optimizer_trigger, 
                                device=device, epoch=epoch, max_norm=args.clip_grad, 
                                set_training_mode=True, task_id=p_task_id, class_mask=class_mask, args=args,backdoor=backdoor)
                
                backdoor.update_trigger(trigger)

                torch.save(trigger,'trigger.pt')
            
            
    
            if lr_scheduler:
                lr_scheduler.step(epoch)

        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args,backdoor=backdoor)
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