import torch
from dataloader import mask_colors_cityscape

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])

    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total


def mask_to_rgb(mask):
    maskr, maskg, maskb = mask.clone(), mask.clone(), mask.clone()

    for idx in range(len(mask_colors_cityscape)):
        maskr[maskr == idx] = mask_colors_cityscape[idx][0]
        maskg[maskg == idx] = mask_colors_cityscape[idx][1]
        maskb[maskb == idx] = mask_colors_cityscape[idx][2]
    mask = torch.cat((maskr.unsqueeze(1), torch.cat((maskg.unsqueeze(1), maskb.unsqueeze(1)), dim=1)), dim=1)
    return mask

