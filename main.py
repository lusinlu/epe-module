import argparse
import os
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataloader import dataset_camvid, dataset_Cityscapes, mean, std
from rteffnet import RTEffNet
from utils import AverageMeter, intersectionAndUnionGPU, mask_to_rgb
import numpy as np

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description="Training of RTEffNet model with EPE module on Cityscapes dataset")
parser.add_argument("--data_path", type=str,help="Path to datasets")
parser.add_argument("--epochs", default=300, type=int, help="Number of total epochs")
parser.add_argument("--img_width", type=int, default=1024, help="Width of the image (576 for CamVid)")
parser.add_argument("--img_height", type=int, default=512, help="Height of the image (768 for CamVid)")
parser.add_argument("--b", default=2, type=int, help="mini-batch size per GPU.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--weights", default="", help="Path to weights (to continue training)")
parser.add_argument("--dataset", default="cityscapes", help="Dataset - cityscapes or camvid.")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--psize", type=int, default=32, help="patch size for EPE model inputs")
parser.add_argument("--classes", type=int, default=19, help="number of classes (32 for CamVid)")
parser.add_argument('--ignore', default=19, type=int, help='pixel value to be ignored on the mask(255 for CamVid)')


args = parser.parse_args()
print(args)

try:
    os.makedirs("weights")
except OSError:
    pass

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: CUDA device detected, consider running with --cuda")
device = torch.device("cuda" if args.cuda else "cpu")

if args.dataset == "camvid":
    train_dataloader, val_dataloader = dataset_camvid(batch_size=args.b, data_path=args.data_path,
                                                      image_height=args.img_height, image_width=args.img_width)
else:
    train_dataloader, val_dataloader = dataset_Cityscapes(root=args.data_path, batch_size=args.b,
                                                          image_height=args.img_height, image_width=args.img_width)


model = RTEffNet(patch_size=args.psize, image_width=args.img_width, image_height=args.img_height, num_classes=args.classes).to(device)
print(f"num of parameters - {sum([m.numel() for m in model.parameters()])}")
model = nn.DataParallel(model)

if args.weights:
    model.load_state_dict(torch.load(args.weights, map_location=device)["net"])
    print(f"Loaded pretrained weights for RTEffNet")

criterion = nn.CrossEntropyLoss(ignore_index=args.ignore)
criterion_mse = nn.MSELoss()

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
# pony learning rate schedule
lr_fc = lambda iteration: (1 - iteration / 20000) ** 0.9
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_fc, -1)

summary = SummaryWriter()
pixel_scores = np.zeros(args.epochs)

for epoch in range(args.epochs):
    model.train()
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for iteration, (inputs, target) in progress_bar:
        optimizer.zero_grad()

        inputs, target = inputs.to(device), target.to(device)
        output, feature, rgb_epe = model(inputs)

        loss_ce = criterion(output, target)
        loss_mse = criterion_mse(rgb_epe, inputs)
        loss = loss_ce + 0.5 * loss_mse
        loss.backward()

        optimizer.step()

        progress_bar.set_description(f"[{epoch + 1}/{args.epochs}][{iteration + 1}/{len(train_dataloader)}] "
                                     f"Epoch: {epoch + 1} " f"Loss: {loss:.2f}.")

    # SUMMARY OF THE EPOCH
    summary.add_scalar('Loss/train', loss.detach().cpu().numpy(), epoch)
    summary.add_images('output',mask_to_rgb(torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1))[:1], epoch)
    summary.add_images('target', mask_to_rgb(target)[:1] , epoch)
    summary.add_images('feature',(feature)[:1] , epoch)
    summary.add_images('rgb',(rgb_epe)[:1] , epoch)
    rgb_input = inputs.clone()
    if args.dataset == "cityscapes":
        rgb_input = rgb_input.mul_(torch.as_tensor(std).view(-1, 1, 1).to(inputs.device)).add_(torch.as_tensor(mean).view(-1, 1, 1).to(inputs.device))
    summary.add_images('input', rgb_input[:1], epoch)

    # Validation
    model.eval()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for iteration, (inputs, target) in progress_bar:
            inputs, target = inputs.to(device), target.to(device)

            prediction, _, _ = model(inputs)
            loss = criterion(prediction, target)
            loss = torch.mean(loss)
            progress_bar.set_description(f"Epoch: {epoch + 1} [{iteration + 1}/{len(val_dataloader)}] "
                                         f"Loss: {loss.item():.6f} ")

            output = prediction.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore)
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
        summary.add_scalar('accuracy/mIOU',mIoU, epoch)

    scheduler.step()









