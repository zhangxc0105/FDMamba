import os
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from datetime import datetime
from torch.utils.data import DataLoader

from data.data import get_data, get_test_data


class BaseSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nEpochs = cfg['nEpochs']
        self.checkpoint_dir = cfg['checkpoint']
        self.epoch = 1

        self.timestamp = datetime.now().strftime('%m%d%H%M%S')

        if cfg['gpu_mode']:
            self.num_workers = cfg['threads']
        else:
            self.num_workers = 0

        self.train_dataset = get_data(cfg, cfg['data']['data_name'])
        self.train_loader = DataLoader(self.train_dataset, cfg['data']['batch_size'],
                                       shuffle=True, num_workers=self.num_workers)

        self.test_dataset = get_test_data(cfg, cfg['data']['data_name'])
        self.test_loader = DataLoader(self.test_dataset, shuffle=False,
                                     batch_size=1, num_workers=self.num_workers)

        self.records = {'Epoch': [], 'PSNR': [], 'SSIM': [], 'ERGAS': [], 'Loss': [],
                         'QNR': [], 'D_lamda': [], 'D_s': []}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def load_checkpoint(self, model_path):
        if os.path.exists(model_path):
            ckpt = torch.load(model_path)
            self.epoch = ckpt['epoch']
            self.records = ckpt['records']
        else:
            raise FileNotFoundError

    def save_checkpoint(self):
        self.ckp = {
            'epoch': self.epoch,
            'records': self.records,
        }

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def run(self):
        while self.epoch <= self.nEpochs:
            self.train()
            self.eval()
            self.save_checkpoint()
            self.epoch += 1