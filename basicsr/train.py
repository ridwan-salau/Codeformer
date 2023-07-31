import argparse
import datetime
import logging
import math
import json
from pathlib import Path
import random
import time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info, get_root_logger, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename, set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist, get_dist_info
from basicsr.utils.options import dict2str, parse

import warnings
# ignore UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.
warnings.filterwarnings("ignore", category=UserWarning)

def parse_options(root_path, is_train=True, yml_only=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=Path, required=True, help='Path to option YAML directory.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--metrics-path', type=Path, required=True)
    parser.add_argument('--stage', type=str, default='stage1')
    args = parser.parse_args()
    opt = parse(args.opt/f"{args.stage}.yml", root_path, is_train=is_train)
    opt['metrics_path'] = args.metrics_path
    opt['stage'] = args.stage
    opt["opt_path"] = args.opt
    
    # Substitue paths to models with output from previous stage
    prev_stage_out:Path = args.metrics_path/"prev_stage_out.json"
    if not args.stage=="stage1":
        assert prev_stage_out.exists(), f"File {prev_stage_out} not found!"
        with prev_stage_out.open() as f:
            params_update = json.load(f)
        for key1, val1 in params_update.items():
            if not isinstance(val1, dict): continue
            for key2, val2 in val1.items():
                opt[key1][key2] = val2
                
    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    elif not yml_only:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    # logger.info(get_env_info())
    # logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project') is not None):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger'):
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: ' f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def train_pipeline(opt):

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'], map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)
    
    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = build_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = build_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.' "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter+1}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and opt['datasets'].get('val') is not None \
                and (current_iter % opt['val']['val_freq'] == 0):
                model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    # Log run_duration and minimum loss.
    msg_logger({}, run_duration=round(time.time() - start_time, 2))

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    chkpt_paths = model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None and opt['datasets'].get('val'):
        model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()
        
    return chkpt_paths, msg_logger.total_loss


if __name__ == '__main__':
    from cachestore import Cache
    from torch.distributed.elastic.multiprocessing.errors import record
    
    cache = Cache(name="codeformer_cache")
    
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    @cache(ignore={"opt"})
    @record
    def train(opt, cache_key):
        chkpt_paths, loss = train_pipeline(opt)        
        return chkpt_paths, loss
    
    # Stage1
    # ======
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(root_path, is_train=True)
    
    # Create the cache key depending on the stage. We memoize all previous stages if they use the same train hyperparameters.
    cache_key = [opt["train"]]
    if opt["stage"]=="stage3":
        opt1 = parse(opt['opt_path']/"stage1.yml", root_path)
        opt2 = parse(opt['opt_path']/"stage2.yml", root_path)
        cache_key.extend([opt1["train"], opt2["train"]])
    elif opt["stage"] == "stage2":
        opt1 = parse(opt['opt_path']/"stage1.yml", root_path)
        cache_key.extend([opt1["train"]])
        
    start_time = time.time()
    
    model_chkpts, loss = train(opt, cache_key)
    
    if opt["rank"] == 0:
        
        # Save the loss and cost
        msg_logger = MessageLogger(opt)
        msg_logger.total_loss = loss
        msg_logger({}, run_duration=round(time.time() - start_time, 2))
        
        if opt["stage"]=="stage2":
            with (opt["metrics_path"]/"prev_stage_out.json").open() as f:
                network_d = json.load(f)["network_d"]
                model_chkpts = *model_chkpts, network_d
                
        # Save the model_chkpts paths for the previous stage in a file for use in the next stage.
        params_update = {"stage1": {"network_g":{"vqgan_path": model_chkpts[0]}, "network_d":model_chkpts[1]}, 
            "stage2": {"path":{"pretrain_network_g": model_chkpts[0], "pretrain_network_d": model_chkpts[1]}},
            "stage3": {}}
        
        with (opt["metrics_path"]/"prev_stage_out.json").open("w") as f:
            json.dump(params_update[opt["stage"]], f)
    
    # Stage2
    # ======
    # parse options, set distributed setting, set ramdom seed
    # params_update = {"network_g":{"vqgan_path": stage1_out[0]}}
    # opt = parse_options(root_path, is_train=True, stage_file="stage2.yml", params_update=params_update)
    # cache_key.append(opt["optimizable"])
    # stage2_out = train(opt, cache_key)
    
    # Stage3
    # ======
    # parse options, set distributed setting, set ramdom seed
    # params_update = {"path":{"pretrain_network_g": stage2_out[0], "pretrain_network_d": stage2_out[1]}}
    # opt = parse_options(root_path, is_train=True, stage_file="stage3.yml", params_update=params_update)
    # cache_key.append(opt["optimizable"])
    # stage3_out = train(opt, cache_key)