import torch
import torch.nn as nn
import os
import scipy.io as sio
from einops import rearrange
from model.fdmamba import Net

import h5py
import numpy as np
import torch.backends.cudnn as cudnn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def load_set(file_path):
    data = h5py.File(file_path)
    lms = torch.from_numpy(np.array(data['lms'][...], dtype=np.float32) / scale).unsqueeze(dim=0).permute(
        [1, 0, 2, 3, 4]).float()
    ms = torch.from_numpy(np.array(data['ms'][...], dtype=np.float32) / scale).unsqueeze(dim=0).permute(
        [1, 0, 2, 3, 4]).float()
    pan = torch.from_numpy(np.array(data['pan'][...], dtype=np.float32) / scale).unsqueeze(dim=0).permute(
        [1, 0, 2, 3, 4]).float()
    return lms, ms, pan

def split_test(size, pad, test_data_path):
    ratio = scale
    file_path = test_data_path
    with h5py.File(file_path, 'r') as f:
        datasets = f.keys()
        for dataset_name in datasets:
            dataset = f[dataset_name]
            data = dataset[:]
            if dataset_name == "pan":
                test_pan = torch.from_numpy(np.array(data, dtype=np.float32)) / ratio
            elif dataset_name == "ms":
                test_ms = torch.from_numpy(np.array(data, dtype=np.float32)) / ratio
            elif dataset_name == "lms":
                test_lms = torch.from_numpy(np.array(data, dtype=np.float32)) / ratio
    image_num, C, h, w = test_ms.shape
    _, _, H, W = test_pan.shape
    cut_size = size  # must be divided by 4, we recommend 64
    ms_size = cut_size // 4
    pad = pad  # must be divided by 4
    edge_H = cut_size - (H - (H // cut_size) * cut_size)
    edge_W = cut_size - (W - (W // cut_size) * cut_size)

    os.makedirs(save_dir, exist_ok=True)
    for k in range(image_num):
        with torch.no_grad():
            x1, x2, x3 = test_ms[k, :, :, :], test_pan[k, 0, :, :], test_lms[k, :, :, :]
            x1 = x1.cpu().unsqueeze(dim=0).float()
            x2 = x2.cpu().unsqueeze(dim=0).unsqueeze(dim=1).float()
            x3 = x3.cpu().unsqueeze(dim=0).float()

            x1_pad = torch.zeros(1, C, h + pad // 2 + edge_H // 4, w + pad // 2 + edge_W // 4)
            x2_pad = torch.zeros(1, 1, H + pad * 2 + edge_H, W + pad * 2 + edge_W)
            x3_pad = torch.zeros(1, C, H + pad * 2 + edge_H, W + pad * 2 + edge_W)
            x1 = torch.nn.functional.pad(x1, (pad // 4, pad // 4, pad // 4, pad // 4), 'reflect')
            x2 = torch.nn.functional.pad(x2, (pad, pad, pad, pad), 'reflect')
            x3 = torch.nn.functional.pad(x3, (pad, pad, pad, pad), 'reflect')

            x1_pad[:, :, :h + pad // 2, :w + pad // 2] = x1
            x2_pad[:, :, :H + pad * 2, :W + pad * 2] = x2
            x3_pad[:, :, :H + pad * 2, :W + pad * 2] = x3
            output = torch.zeros(1, C, H + edge_H, W + edge_W)

            scale_H = (H + edge_H) // cut_size
            scale_W = (W + edge_W) // cut_size
            for i in range(scale_H):
                for j in range(scale_W):
                    MS = x1_pad[:, :, i * ms_size: (i + 1) * ms_size + pad // 2,
                         j * ms_size: (j + 1) * ms_size + pad // 2].cuda()
                    PAN = x2_pad[:, :, i * cut_size: (i + 1) * cut_size + 2 * pad,
                          j * cut_size: (j + 1) * cut_size + 2 * pad].cuda()
                    LMS = x3_pad[:, :, i * cut_size: (i + 1) * cut_size + 2 * pad,
                          j * cut_size: (j + 1) * cut_size + 2 * pad].cuda()
                    sr = model(pan=PAN, ms=MS, lms=LMS)
                    sr = torch.clamp(sr, 0, 1)
                    output[:, :, i * cut_size: (i + 1) * cut_size, j * cut_size: (j + 1) * cut_size] = \
                        sr[:, :, pad: cut_size + pad, pad: cut_size + pad] * ratio
            output = output[:, :, :H, :W]
            output = torch.squeeze(output).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
            new_path = os.path.join(save_dir, f'output_mulExm_{k}.mat')
            sio.savemat(new_path, {f'sr': output})
        print(k)

checkpoint_path = r'checkpoint/yourlogname/bestPSNR.pth'
test_data_path = r'dataset/test_wv3_multiExm1.h5'
save_dir = r'2_DL_Result/PanCollection/WV3_Reduced/FDMamba/results/'

if "wv3" in test_data_path or "qb" in test_data_path or "wv2" in test_data_path:
    scale = 2047.
elif "gf2" in test_data_path:
    scale = 1023.

if "wv3" in test_data_path:
    num_channels = 8
elif "qb" in test_data_path or "gf2" in test_data_path:
    num_channels = 4


torch.cuda.manual_seed(123)
cudnn.benchmark = False
np.random.seed(123)
torch.manual_seed(123)

os.makedirs(save_dir, exist_ok=True)
model = Net(num_channels=num_channels).cuda()
model = nn.DataParallel(model)
checkpoint = torch.load(checkpoint_path, map_location='cuda:0') 
model.load_state_dict(checkpoint['net'])
model.eval()


lms, ms, pan = load_set(test_data_path)

with torch.no_grad():
    ms = ms.cuda()
    lms = lms.cuda()
    pan = pan.cuda()
    print('Running model inference...')
    for i in range(pan.shape[0]):
        output= model(ms[i], lms[i], pan[i])
        output = rearrange(output, 'b c h w -> b h w c') * scale
        output_np = output[0].cpu().numpy()
        save_mat_path = os.path.join(save_dir, f'output_mulExm_{i}.mat')
        sio.savemat(save_mat_path, {'sr': output_np})
        print(f"Saved .mat to {save_mat_path}")

# split_test(64, 4, test_data_path)