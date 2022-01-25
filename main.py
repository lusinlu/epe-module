
import argparse
import math
import os
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataloader import dataset_camvid
from model_eff import SRDModel
from utils import AverageMeter, intersectionAndUnionGPU, iou
import numpy as np

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description="EFSR training with the GradientVariance loss")
parser.add_argument("--data_path", type=str,
                    help="Path to datasets")
parser.add_argument("--epochs", default=300, type=int, metavar="N",
                    help="Number of total epochs to run. (default:100)")
parser.add_argument("--image-size", type=int, default=512,
                    help="Size of the data crop (squared assumed). (default:256)")
parser.add_argument("-b", "--batch-size", default=2, type=int,
                    metavar="N", help="mini-batch size (default: 64).")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate. (default:0.01)")
parser.add_argument("--weights", default="",
                    help="Path to weights (to continue training).")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--psize", type=int, default=32,
                    help="patch size for variance calculations")

parser.add_argument("--classes", type=int, default=32,
                    help="number of classes")
parser.add_argument('-ignore_label', default=255, type=int, help='pixel value to be ignored on the mask')




args = parser.parse_args()
print(args)

try:
    os.makedirs("weights")
except OSError:
    pass

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataloader, val_dataloader = dataset_camvid(batch_size=args.batch_size, data_path=args.data_path)

device = torch.device("cuda" if args.cuda else "cpu")

model = SRDModel(patch_size=args.psize, image_width=576, image_height=768, num_classes=args.classes).to(device)
print(f"num of parameters - {sum([m.numel() for m in model.parameters()])}")
# model = nn.DataParallel(model)

if args.weights:
    model.load_state_dict(torch.load(args.weights, map_location=device))

criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
criterion_mse = nn.MSELoss()

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250 ], gamma=0.1)

summary = SummaryWriter()
pixel_scores = np.zeros(args.epochs)

for epoch in range(args.epochs):
    model.train()
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for iteration, (inputs, target) in progress_bar:
        optimizer.zero_grad()

        inputs, target = inputs.to(device), target.to(device)
        output, feature, rgb = model(inputs)

        loss_ce =  criterion(output, target)
        loss_mse = criterion_mse(rgb, inputs)
        loss =  loss_ce + 0.5*loss_mse
        loss.backward()

        optimizer.step()

        progress_bar.set_description(f"[{epoch + 1}/{args.epochs}][{iteration + 1}/{len(train_dataloader)}] "
                                     f"Epoch: {epoch + 1} " f"Loss: {loss:.2f}.")


    summary.add_scalar('Loss/train', loss.detach().cpu().numpy(), epoch)
    mask = torch.repeat_interleave(torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1).unsqueeze(1),3, dim=1)



    summary.add_images('output',(mask * 50)[:2] , epoch)
    summary.add_images('target', torch.repeat_interleave(target.unsqueeze(1),3, dim=1)[:2] * 50, epoch)
    summary.add_images('feature',(feature)[:2] , epoch)
    summary.add_images('rgb',(rgb)[:2] , epoch)


    # Test
    model.eval()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for iteration, (inputs, target) in progress_bar:
            inputs, target = inputs.to(device), target.to(device)

            prediction, _ , _= model(inputs)
            loss = criterion(prediction, target)
            loss = torch.mean(loss)
            progress_bar.set_description(f"Epoch: {epoch + 1} [{iteration + 1}/{len(val_dataloader)}] "
                                         f"Loss: {loss.item():.6f} ")

            output = prediction.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)

            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            loss_meter.update(loss.item(), inputs.size(0))

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        # Calculate average IoU
        chkp_dir = './weights'
        print("epoch{}, pix_acc: {}, meanIoU: {}, allAcc: {}".format(epoch, mAcc, mIoU, allAcc))
        pixel_scores[epoch] = mAcc
        np.save(os.path.join(chkp_dir, "accuracy"), pixel_scores)
        state = {'net': model.state_dict(), 'acc': mAcc, 'epoch': epoch, }
        torch.save(state, os.path.join(chkp_dir, str(epoch) + 'ckpt.pth'))
        # summary.add_scalar('accuracy/val',mAcc, epoch)
        summary.add_scalar('accuracy/mIOU',mIoU, epoch)

    # Dynamic adjustment of learning rate.
    scheduler.step()









