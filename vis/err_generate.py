import os
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def compute_and_save_residual_map(gt_path, output_path, save_path, original_save_path):
    # 读取并归一化图像
    img_gt = np.asarray(Image.open(gt_path)).astype(np.float32) / 255.0
    img_out = np.asarray(Image.open(output_path)).astype(np.float32) / 255.0

    # 确保图像尺寸一致
    if img_gt.shape != img_out.shape:
        raise ValueError(f"尺寸不一致: {gt_path} vs {output_path}")

    # 计算误差图
    if img_gt.ndim == 3:
        if img_gt.shape[2] >= 5:
            channel = [0, 2, 4]
        elif img_gt.shape[2] == 4:
            channel = [0, 1, 2]
        else:
            channel = [0]
        err_map = np.mean(np.abs(img_gt[:, :, channel] - img_out[:, :, channel]), axis=2)
    else:
        err_map = np.abs(img_gt - img_out)

    # 保存误差图
    plt.figure(figsize=(6, 5))
    norm = Normalize(vmin=0, vmax=0.5)
    im = plt.imshow(err_map, cmap='coolwarm', norm=norm)
    # plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved residual map: {save_path}")

    # 直接复制原图文件
    shutil.copyfile(output_path, original_save_path)
    print(f"✅ Copied original image: {original_save_path}")


# ================================# 主程序
# ================================

method_dirs = [
    # 'BDSD_PC', 'CANNet',  'FusionNet', 'MTF_GLP_FS',
    # 'PanMamba', 'PanNet', 'PNN', 'TV',
    'fdmamba'
    # 'gt'
    # 'U2NET', 'Premix',
    # 'Ramsf', 'ADWM'
]

# 修改后的路径
gt_img_path = os.path.join('gt', '12_gt.png')
output_dir = 'residual_maps'  # 保存残差图的输出目录
os.makedirs(output_dir, exist_ok=True)

for method in method_dirs:
    method_img_path = os.path.join(method, '12_ms.png')

    # 保存路径
    save_residual_path = os.path.join(output_dir, f'{method}_residual.png')
    save_original_path = os.path.join(output_dir, f'{method}_ms.png')

    # 调用函数保存残差图和原图
    compute_and_save_residual_map(gt_img_path, method_img_path, save_residual_path, save_original_path)
