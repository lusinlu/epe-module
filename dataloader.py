import numpy as np
import torch.utils.data as data
import data_transform
import os
import os.path
import cv2
from torch.utils.data import Dataset
import torchvision
from cityscapes import Cityscapes
import joint_transforms
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

value_scale = 1


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

mask_colors_cityscape =[(128, 64, 128),(244, 35, 232),(70, 70, 70),(102, 102, 156),(190, 153, 153),(153, 153, 153),
                       (250, 170, 30),(220, 220, 0),(107, 142, 35),(152, 251, 152),(70, 130, 180),(220, 20, 60),
                       (255, 0, 0),(0, 0, 142),(0, 0, 70),(0, 60, 100),(0, 80, 100),(0, 0, 230),(119, 11, 32), (0, 0, 0)]




def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(',')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image) / 255.
        label = np.load(label_path)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label




def dataset_camvid(batch_size, data_path, image_height, image_width):
    camvid_transform_train = data_transform.Compose([
        data_transform.RandScale([0.5, 2.0]),
        data_transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
        data_transform.RandomGaussianBlur(),
        data_transform.RandomHorizontalFlip(),
        data_transform.Resize((image_height, image_width)),
        data_transform.ToTensor()])

    camvid_transform_test = data_transform.Compose([
        data_transform.Resize((image_height, image_width)),
        data_transform.ToTensor()])

    train_list = os.path.join(data_path,"CamVid",  "train.txt")
    val_list = os.path.join(data_path, "CamVid", "val.txt")

    train_dataset = SemData(split='train', data_root=data_path, data_list=train_list, transform=camvid_transform_train)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    val_dataset = SemData(split='val', data_root=data_path, data_list=val_list, transform=camvid_transform_test)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    return train_loader, val_loader


def dataset_Cityscapes(root, batch_size, image_height, image_width):
    train_dataset = Cityscapes(root, split='train', mode='fine', target_type='semantic',
                               transform=joint_transforms.Compose([
                                   joint_transforms.RandomHorizontalFlip(),
                                   joint_transforms.RandomSized((image_height, image_width)),
                                   joint_transforms.ToTensor(),
                                   joint_transforms.Normalize(
                                       mean=mean,
                                       std=std)
                               ]))

    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=12, pin_memory=True, sampler=None)

    val_loader = data.DataLoader(
        Cityscapes(root, split='val', mode='fine', target_type='semantic',
                   transform=joint_transforms.Compose([
                       joint_transforms.Resize((image_height, image_width)),
                       joint_transforms.ToTensor(),
                       joint_transforms.Normalize(
                           mean=mean,
                           std=std)
                   ])),
        batch_size=batch_size, shuffle=True,
        num_workers=10, pin_memory=False)

    return train_loader, val_loader



