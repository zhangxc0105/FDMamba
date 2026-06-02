from utils.config import get_config
from solver.solver import Solver
import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import random
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N_SR')
    parser.add_argument('--option_path', type=str, default='/data/zhangxiaochen/Pansharpening/FDMamba/option.yml')
    opt = parser.parse_args()
    cfg = get_config(opt.option_path)

    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False
    cudnn.enabled = True
    os.environ['PYTHONHASHSEED'] = str(cfg['seed'])

    solver = Solver(cfg)
    solver.run()
