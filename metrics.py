import torch
import torch.nn.functional as F

def psnr(pred, target):

    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def ssim(pred, target, window_size=11):

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, window_size, 1, 0)
    mu_y = F.avg_pool2d(target, window_size, 1, 0)

    sigma_x = F.avg_pool2d(pred * pred, window_size, 1, 0) - mu_x**2
    sigma_y = F.avg_pool2d(target * target, window_size, 1, 0) - mu_y**2
    sigma_xy = F.avg_pool2d(pred * target, window_size, 1, 0) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    return torch.mean(numerator / denominator)