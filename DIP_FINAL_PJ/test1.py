# coding: utf-8
import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT, TEST_HAZERD_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model_backup import DM2FNet, DM2FNet_woPhy
from datasets import SotsDataset, OHazeDataset, HazerdDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from skimage.color import deltaE_ciede2000, rgb2lab
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
# exp_name = 'RESIDE_ITS'
exp_name = 'O-Haze'

args = {
    'snapshot': 'iter_40000_loss_0.01462_lr_0.000000',
}

to_test = {
    # 'SOTS': TEST_SOTS_ROOT,
    # 'HazeRD':TEST_HAZERD_ROOT,
    'O-Haze': OHAZE_ROOT,
}

to_pil = transforms.ToPILImage()


def main():
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()

        for name, root in to_test.items():
            print(name)
            if 'HazeRD' in name:
                net = DM2FNet().cuda()
                dataset = HazerdDataset(root)
            elif 'SOTS' in name:
                net = DM2FNet().cuda()
                dataset = SotsDataset(root)
            elif 'O-Haze' in name:
                net = DM2FNet().cuda()
                # net = DM2FNet_woPhy().cuda()
                dataset = OHazeDataset(root, 'test')
            else:
                raise NotImplementedError

            # net = nn.DataParallel(net)

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            psnrs, ssims = [], []
            loss_record = AvgMeter()
            ciede2000s = []
            mses = []

            for idx, data in enumerate(dataloader):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

                check_mkdir(os.path.join(ckpt_path, exp_name,
                                         '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.cuda()
                
                if 'O-Haze' in name:
                    res = sliding_forward(net, haze).detach()
                else:
                    res = net(haze).detach()

                loss = criterion(res, gts.cuda())
                loss_record.update(loss.item(), haze.size(0))

                for i in range(len(fs)):
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])

                    # 计算MSE
                    mse = mean_squared_error(gt, r)
                    mses.append(mse)
                    # 转换为 LAB 颜色空间以计算 CIEDE2000
                    r_lab = rgb2lab(r)
                    gt_lab = rgb2lab(gt)
                    ciede2000 = deltaE_ciede2000(gt_lab, r_lab)
                    ciede2000s.append(ciede2000.mean())  # 取平均值可能更有意义

                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)
                    # print("Image shape: ", sgt.shape)
                    ssim = structural_similarity(gt, r, data_range=1, multichannel=True,
                                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False,win_size=3)
                    ssims.append(ssim)
                    print('predicting for {} ({}/{}) [{}]: PSNR {:.4f}, SSIM {:.4f}, MSE {:.4f}, CIEDE2000 {:.4f}'.format(name, idx + 1, len(dataloader), fs[i], psnr, ssim, mse, ciede2000.mean()))
                    # print('predicting for {} ({}/{}) [{}]: MSE {:.4f}, CIEDE2000 {:.4f}'.format(name, idx + 1, len(dataloader), fs[i], mse, ciede2000.mean()))

                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name,
                                     '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))

            print(f"[{name}] L1: {loss_record.avg:.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, Average MSE: {np.mean(mses):.6f}, Average CIEDE2000: {np.mean(ciede2000s):.6f}")
            # print(f"[{name}] Average MSE: {np.mean(mses):.6f}, Average CIEDE2000: {np.mean(ciede2000s):.6f}")

if __name__ == '__main__':
    main()
