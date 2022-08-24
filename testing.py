import os
from numpy.core.numeric import outer
import torch
import yaml
from easydict import EasyDict as edict
import encoding
from train import create_data_loaders
import numpy as np
import cv2
from PIL import Image
import math
from tqdm import tqdm
from utils import *

from matplotlib import pyplot as plt
cmap = plt.cm.viridis

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def rmse(groundtruth, pred):
  valid_mask = groundtruth> 1e-3
  pred = pred[valid_mask]
  groundtruth = groundtruth[valid_mask]
  return np.sqrt(np.mean((pred-groundtruth)**2))

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)

def evaluate(output, target):
    valid_mask = target> 1e-3
    output = output[valid_mask]
    target = target[valid_mask]

    abs_diff = (output - target).abs()
    mse = float((torch.pow(abs_diff, 2)).mean())
    rmse = math.sqrt(mse)
    mae = float(abs_diff.mean())
    lg10 = float((log10(output) - log10(target)).abs().mean())
    absrel = float((abs_diff / target).mean())

    maxRatio = torch.max(output / target, target / output)
    delta1 = float((maxRatio < 1.25).float().mean())
    delta2 = float((maxRatio < 1.25 ** 2).float().mean())
    delta3 = float((maxRatio < 1.25 ** 3).float().mean())
    return {"rmse": rmse, "mae":mae, "mse":mse, "lg10": lg10, "absrel":absrel, "d1":delta1, "d2":delta2, "d3":delta3}


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_rmse = 0
        self.sum_mae = 0
        self.sum_mse = 0
        self.sum_lg10 = 0
        self.sum_absrel = 0
        self.sum_d1 = 0 
        self.sum_d2 = 0 
        self.sum_d3 = 0
        self.count = 0

    def update(self, loss_dict, n=1):
        self.sum_rmse += loss_dict.get("rmse") * n
        self.sum_mae += loss_dict.get("mae") * n
        self.sum_mse += loss_dict.get("mse") * n
        self.sum_lg10 += loss_dict.get("lg10") * n
        self.sum_absrel += loss_dict.get("absrel") * n
        self.sum_d1 += loss_dict.get("d1") * n
        self.sum_d2 += loss_dict.get("d2") * n
        self.sum_d3 += loss_dict.get("d3") * n
        
        self.count += n
    
    def get_average(self):
      avg_rmse = self.sum_rmse / self.count
      avg_mae = self.sum_mae / self.count
      avg_mse = self.sum_mse / self.count
      avg_lg10 = self.sum_lg10 / self.count
      avg_absrel = self.sum_absrel / self.count
      avg_d1 = self.sum_d1 / self.count
      avg_d2 = self.sum_d2 / self.count
      avg_d3 = self.sum_d3 / self.count
      return {"rmse": avg_rmse, "mae": avg_mae, "mse": avg_mse, "lg10": avg_lg10, "absrel": avg_absrel, "d1": avg_d1, "d2": avg_d2, "d3": avg_d3}



def run_test():
  Avg = AverageMeter()
  sum_loss = 0
  net.eval()
  for batch_idx, (rgb, lidar, depth) in enumerate(testloader):
    with torch.no_grad():
      output = net(rgb, lidar)

    loss_dict = evaluate(output[0].cpu()[:,:,14:-14,8:-8],depth.cpu()[:,:,14:-14,8:-8])
    Avg.update(loss_dict)

    pred = output[0].squeeze().cpu().numpy()
    rgb = rgb.squeeze().cpu().numpy()
    groundtruth = depth.squeeze().cpu().numpy()
    lidar = lidar.squeeze().cpu().numpy()
    pred = pred[14:-14,8:-8]
    groundtruth = groundtruth[14:-14,8:-8]
    lidar = lidar[14:-14,8:-8]
    loss = rmse(groundtruth, pred)
    sum_loss += loss
    #if batch_idx % 100 == 0:
    #  print(f"Loss for batch {batch_idx} equals: {loss}")

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
  print(f"Average loss for testset: {sum_loss/len(testloader)}")
  return Avg.get_average()





if __name__ == '__main__':
  config_name = 'GNS.yaml'
  with open(os.path.join('configs', config_name), 'r') as file:
      config_data = yaml.load(file, Loader=yaml.FullLoader)
  config = edict(config_data)

  # Uniform Model
  config.update({"resume_seed": 8023})
  # ORB Mdel
  #config.update({"resume_seed": 9023})
  net = init_net(config)
  torch.cuda.empty_cache()
  torch.backends.cudnn.benchmark = True
  net.cuda()
  net = encoding.parallel.DataParallelModel(net)
  net = resume_state(config, net)
  print(f"Number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")

  #sparsifier = "orb_sampler"
  sparsifier = "uar"

  trainloader, testloader = create_data_loaders(sparsifier)
  loss_test = {"rmse": [], "mae": [], "mse": [], "lg10": [], "absrel": [], "d1": [], "d2": [], "d3": []}

  for n_samples in tqdm([500,400,300,200,100,0]):
    testloader.dataset.sparsifier.num_samples = n_samples
    loss_dict = run_test()
    loss_test.get("rmse").append(loss_dict.get("rmse"))
    loss_test.get("mae").append(loss_dict.get("mae"))
    loss_test.get("mse").append(loss_dict.get("mse"))
    loss_test.get("lg10").append(loss_dict.get("lg10"))
    loss_test.get("absrel").append(loss_dict.get("absrel"))
    loss_test.get("d1").append(loss_dict.get("d1"))
    loss_test.get("d2").append(loss_dict.get("d2"))
    loss_test.get("d3").append(loss_dict.get("d3"))

print(loss_test)
print("-")
