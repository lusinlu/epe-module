import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data.distributed
from dataloader import dataset_Cityscapes
from rteffnet import RTEffNet
from utils import AverageMeter, intersectionAndUnionGPU
import numpy as np

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# camvid img_width", type=int, default=576, img_height", type=int, default=768
parser = argparse.ArgumentParser(description="EFSR test")
parser.add_argument("--data_path", type=str,help="Path to datasets")
parser.add_argument("--img_width", type=int, default=1024, help="Width of the image (576 for CamVid)")
parser.add_argument("--img_height", type=int, default=512, help="Height of the image (768 for CamVid)")
parser.add_argument("--weights", default="", help="Path to weights (to continue training)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--psize", type=int, default=32, help="patch size for EPE model inputs")
parser.add_argument("--classes", type=int, default=19, help="number of classes (32 for CamVid)")
parser.add_argument('--ignore', default=19, type=int, help='pixel value to be ignored on the mask(255 for CamVid)')


# helper class to remove "model" from names in case of dataparallel training
class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


args = parser.parse_args()

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: CUDA device detected, consider running with --cuda")
device = torch.device("cuda" if args.cuda else "cpu")

_, val_dataloader = dataset_Cityscapes(root=args.data_path, batch_size=1, image_height=args.img_height, image_width=args.img_width)


model = RTEffNet(patch_size=args.psize, image_width=args.img_width, image_height=args.img_height, num_classes=args.classes).to(device)
model = WrappedModel(model)
print(f"num of parameters - {sum([m.numel() for m in model.parameters()])}")
model.load_state_dict(torch.load(args.weights, map_location=device)["net"])

model.eval()
intersection_meter = AverageMeter()
union_meter = AverageMeter()
target_meter = AverageMeter()

with torch.no_grad():
    for iteration, (inputs, target) in enumerate(val_dataloader):
        inputs, target = inputs.to(device), target.to(device)

        prediction, _, _ = model(inputs)
        output = prediction.max(1)[1]

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # Calculate average IoU
    print('mIOU - ', mIoU)










