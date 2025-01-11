import torch
from backdoor import *
from typing import Iterable

def train_one_p_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, optimizer: torch.optim.Optimizer,
                    device: torch.device,input,target, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,backdoor: Sleeper = None):


    metric_logger = utils.MetricLogger(delimiter="  ")
    if backdoor is not None:
        model,metric_logger = backdoor.init_objects(model,metric_logger,set_training_mode)

    original_model.eval()

    # header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
        
    input = input.to(device, non_blocking=True).unsqueeze(0)
    target = target.to(device, non_blocking=True).unsqueeze(0)

    if backdoor is not None:
        input = backdoor.create_poisoned_dataset(input,target) 

    else:
        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
    
    # output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
    # logits = output['logits']

    # # here is the trick to mask out classes of non-current tasks
    # if args.train_mask and class_mask is not None:
    #     mask = class_mask[task_id]
    #     not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
    #     not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device) 
    #     logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

    loss = backdoor.calculate_loss(criterion,original_model, model, input,target, task_id, class_mask)
    metric_logger = backdoor.update_logger(metric_logger)

    if not math.isfinite(loss.item()):
        print("Loss is {}, stopping training".format(loss.item()))
        sys.exit(1)
    

    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()
    torch.cuda.synchronize()
    metric_logger.update(Loss=loss.item())

        # break
     
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if backdoor is None:


        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    else:
        
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()},backdoor.trigger