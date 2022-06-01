#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    train.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/14 7:50 PM

import os
import torch
import yaml
from easydict import EasyDict as edict

from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo, ORBSampling
import numpy as np

def create_data_loaders():
    # Data loading code
    print("=> creating data loaders ...")
    nyu_path = '../nyudepthv2'
    traindir = os.path.join(nyu_path, 'train')
    valdir = os.path.join(nyu_path, 'val')

    #traindir = os.path.join('data', args.data, 'train')
    #valdir = os.path.join('data', args.data, 'val')
    train_loader = None
    val_loader = None


    max_depth = None
    sparsifier = "orb_sampler"
    num_samples = 500
    data = 'nyudepthv2'
    evaluate = False
    modality = "rgbd"
    workers = 10
    batch_size = 8



    # sparsifier is a class for generating random sparse depth input from the ground truth
    #sparsifier = None
    max_depth = np.inf
    if sparsifier == UniformSampling.name:
        sparsifier = UniformSampling(num_samples=num_samples, max_depth=max_depth)
    elif sparsifier == SimulatedStereo.name:
        sparsifier = SimulatedStereo(num_samples=num_samples, max_depth=max_depth)
    elif sparsifier == ORBSampling.name:
        sparsifier = ORBSampling(num_samples=num_samples, max_depth=max_depth)
        

    if data == 'nyudepthv2':
        from dataloaders.nyu_dataloader import NYUDataset
        if not evaluate:
            train_dataset = NYUDataset(traindir, type='train',
                modality=modality, sparsifier=sparsifier)
        val_dataset = NYUDataset(valdir, type='val',
            modality=modality, sparsifier=sparsifier)

    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyudepthv2 or kitti.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    if not evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id))
            # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return train_loader, val_loader


def train(epoch):
    global iters
    Avg = AverageMeter()
    for batch_idx, (rgb, lidar, depth) in enumerate(trainloader):
        if epoch >= config.test_epoch and iters % config.test_iters == 0:
            test()
        net.train()
        rgb, lidar, depth = rgb.cuda(), lidar.cuda(), depth.cuda()
        optimizer.zero_grad()
        output = net(rgb, lidar)
        loss = criterion(output, depth).mean()
        loss.backward()
        optimizer.step()
        Avg.update(loss.item())
        iters += 1
        if config.vis and batch_idx % config.vis_iters == 0:
            print('Epoch {} Idx {} Loss {:.4f}'.format(epoch, batch_idx, Avg.avg))


def test():
    global best_metric
    Avg = AverageMeter()
    net.eval()
    for batch_idx, (rgb, lidar, depth) in enumerate(testloader):
        rgb, lidar, depth = rgb.cuda(), lidar.cuda(), depth.cuda()
        with torch.no_grad():
            output = net(rgb, lidar)
            prec = metric(output, depth).mean()
        Avg.update(prec.item(), rgb.size(0))

    if Avg.avg < best_metric:
        best_metric = Avg.avg
        save_state(config, net)
        print('Best Result: {:.4f}\n'.format(best_metric))


if __name__ == '__main__':
    # config_name = 'GN.yaml'
    config_name = 'GNS.yaml'
    with open(os.path.join('configs', config_name), 'r') as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)
    config = edict(config_data)
    print(config.name)
    #os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_id) for gpu_id in config.gpu_ids])
    # Only one GPU available in colab -> ignore gpu ids in yaml file
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    from utils import *

    init_seed(config)
    #trainloader, testloader = init_dataset(config)
    trainloader, testloader = create_data_loaders()
    net = init_net(config)
    criterion = init_loss(config)
    metric = init_metric(config)
    net, criterion, metric = init_cuda(net, criterion, metric)
    optimizer = init_optim(config, net)
    lr_scheduler = init_lr_scheduler(config, optimizer)
    iters = 0
    best_metric = 100
    for epoch in range(config.start_epoch, config.nepoch):
        train(epoch)
        lr_scheduler.step()
    print('Best Results: {:.4f}\n'.format(best_metric))


