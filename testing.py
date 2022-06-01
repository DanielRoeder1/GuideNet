import os
import torch
import yaml
from easydict import EasyDict as edict
import encoding
from train import create_data_loaders
import numpy as np
import cv2
from PIL import Image

from matplotlib import pyplot as plt
cmap = plt.cm.viridis

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def rmse(groundtruth, pred):
  return np.sqrt(np.mean((pred-groundtruth)**2))

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C


def run_test():
  sum_loss = 0
  net.eval()
  for batch_idx, (rgb, lidar, depth) in enumerate(testloader):
    with torch.no_grad():
      output = net(rgb, lidar)

    pred = output[0].squeeze().cpu().numpy()
    rgb = rgb.squeeze().cpu().numpy()
    groundtruth = depth.squeeze().cpu().numpy()
    lidar = lidar.squeeze().cpu().numpy()

    pred = pred[14:-14,8:-8]
    groundtruth = groundtruth[14:-14,8:-8]
    lidar = lidar[14:-14,8:-8]

    loss = rmse(groundtruth, pred)
    sum_loss += loss
    if batch_idx % 100 == 0:
      print(f"Loss for batch {batch_idx} equals: {loss}")

    if batch_idx == 0:
      rgb = np.transpose(rgb, (1,2,0))[14:-14,8:-8,:]
      rgb = (rgb * 255).astype(int)

      d_min = min(np.min(groundtruth), np.min(pred), np.min(lidar))
      d_max = max(np.max(groundtruth), np.max(pred), np.max(lidar))
      groundtruth = colored_depthmap(groundtruth, d_min, d_max)
      pred = colored_depthmap(pred, d_min, d_max)
      lidar = colored_depthmap(lidar, d_min, d_max)

      stack = np.hstack((rgb, lidar, pred, groundtruth))
    
    elif batch_idx < 20:
      rgb = np.transpose(rgb, (1,2,0))[14:-14,8:-8,:]
      rgb = (rgb * 255).astype(int)

      d_min = min(np.min(groundtruth), np.min(pred), np.min(lidar))
      d_max = max(np.max(groundtruth), np.max(pred), np.max(lidar))
      groundtruth = colored_depthmap(groundtruth, d_min, d_max)
      pred = colored_depthmap(pred, d_min, d_max)
      lidar = colored_depthmap(lidar, d_min, d_max)

      stack_v = np.hstack((rgb, lidar, pred, groundtruth))
      stack = np.vstack((stack, stack_v))

  img_merge = Image.fromarray(stack.astype('uint8'))
  img_merge.save("testin_res.jpeg")
  print(f"Average loss for testset: {sum_loss/654}")





if __name__ == '__main__':
  config_name = 'GNS.yaml'
  with open(os.path.join('configs', config_name), 'r') as file:
      config_data = yaml.load(file, Loader=yaml.FullLoader)
  config = edict(config_data)
  from utils import *

  net = init_net(config)
  torch.cuda.empty_cache()
  torch.backends.cudnn.benchmark = True
  net.cuda()
  net = encoding.parallel.DataParallelModel(net)
  net = resume_state(config, net)

  trainloader, testloader = create_data_loaders()

  run_test()