


import math
import torch
import torch.nn as nn
from model_eff import Entropy, Conv_Bn_Relu

""" 
Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = utils.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""


model_url = './checkpoints/XceptionA_best.pth.tar'


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class XceptionA(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf

    Modified Xception A architecture, as specified in
    https://arxiv.org/pdf/1904.02216.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(XceptionA, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 8, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        # conv for reducing channel size in input for non-first backbone stages
        self.enc2_conv = nn.Conv2d(240, 8, 1, 1, bias=False) # bias=False?

        self.enc2_1 = Block(8, 12, 4, 1, start_with_relu=True, grow_first=True)
        self.enc2_2 = Block(12, 12, 4, 1, start_with_relu=True, grow_first=True)
        self.enc2_3 = Block(12, 48, 4, 2, start_with_relu=True, grow_first=True)
        self.enc2 = nn.Sequential(self.enc2_1, self.enc2_2, self.enc2_3)

        self.enc3_conv = nn.Conv2d(144, 48, 1, 1, bias=False)

        self.enc3_1 = Block(48, 24, 6, 1, start_with_relu=True, grow_first=True)
        self.enc3_2 = Block(24, 24, 6, 1, start_with_relu=True, grow_first=True)
        self.enc3_3 = Block(24, 96, 6, 2, start_with_relu=True, grow_first=True)
        self.enc3 = nn.Sequential(self.enc3_1, self.enc3_2, self.enc3_3)

        self.enc4_conv = nn.Conv2d(288, 96, 1, 1, bias=False)

        self.enc4_1 = Block(96, 48, 4, 1, start_with_relu=True, grow_first=True)
        self.enc4_2 = Block(48, 48, 4, 1, start_with_relu=True, grow_first=True)
        self.enc4_3 = Block(48, 192, 4, 2, start_with_relu=True, grow_first=True)
        self.enc4 = nn.Sequential(self.enc4_1, self.enc4_2, self.enc4_3)

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, num_classes)
        self.fca = nn.Conv2d(num_classes, 192, 1)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        enc2 = self.enc2(x)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        pool = self.pooling(enc4)
        fc = self.fc(pool.view(pool.size(0), -1))
        fca = self.fca(fc.view(fc.size(0), -1, 1, 1))
        fca = enc4 * fca

        return enc2, enc3, enc4, fc, fca

    def forward_concat(self, fca_concat, enc2_concat, enc3_concat, enc4_concat):
        """For second and third stage."""
        enc2 = self.enc2(self.enc2_conv(torch.cat((fca_concat, enc2_concat), dim=1)))
        enc3 = self.enc3(self.enc3_conv(torch.cat((enc2, enc3_concat), dim=1)))
        enc4 = self.enc4(self.enc4_conv(torch.cat((enc3, enc4_concat), dim=1)))
        pool = self.pooling(enc4)
        fc = self.fc(pool.view(pool.size(0), -1))
        fca = self.fca(fc.view(fc.size(0), -1, 1, 1))
        fca = enc4 * fca

        return enc2, enc3, enc4, fc, fca


def backbone(pretrained=False, **kwargs):
    """
    Construct Xception.
    """

    model = XceptionA(**kwargs)
    if pretrained:
        # from collections import OrderedDict
        # state_dict = torch.load(model_url)
        # new_state_dict = OrderedDict()
        #
        # for k, v in state_dict.items():
        #    name = k[7:]  # remove 'module.' of data parallel
        #    new_state_dict[name] = v
        #
        # model.load_state_dict(new_state_dict, strict=False)
        model.load_state_dict(torch.load(model_url), strict=False)
    return model

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_relu = self.relu(x)
        x_conv = self.conv(x_relu)
        x_bn = self.bn(x_conv)
        return x_bn


class Decoder(nn.Module):

    def __init__(self, n_classes=19):
        super(Decoder, self).__init__()
        self.n_classes = n_classes
        self.enc1_conv = ConvBlock(48, 32, 1) # not sure about the out channels

        self.enc2_conv = ConvBlock(48, 32, 1)
        self.enc2_up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.enc3_conv = ConvBlock(48, 32, 1)
        self.enc3_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.enc_conv = ConvBlock(32, n_classes, 1)

        self.fca1_conv = ConvBlock(192, n_classes, 1)
        self.fca1_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.fca2_conv = ConvBlock(192, n_classes, 1)
        self.fca2_up = nn.UpsamplingBilinear2d(scale_factor=8)

        self.fca3_conv = ConvBlock(192, n_classes, 1)
        self.fca3_up = nn.UpsamplingBilinear2d(scale_factor=16)

        self.final_up = nn.UpsamplingBilinear2d(scale_factor=4)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, enc1, enc2, enc3, fca1, fca2, fca3):
        """Note that enc1 denotes the output of the enc4 module of backbone instance 1."""
        e1 = self.enc1_conv(enc1)
        e2 = self.enc2_up(self.enc2_conv(enc2))
        e3 = self.enc3_up(self.enc3_conv(enc3))

        e = self.enc_conv(e1 + e2 + e3)

        f1 = self.fca1_up(self.fca1_conv(fca1))
        f2 = self.fca2_up(self.fca1_conv(fca2))
        f3 = self.fca3_up(self.fca1_conv(fca3))

        o = self.final_up(e + f1 + f2 + f3)

        return o

class PatchEncoderModule(nn.Sequential):
    def __init__(self,in_features, n_features, n_res_blocks, kernel_size=3, padding=1):
        super(PatchEncoderModule, self).__init__()
        # define encoder module
        head = [Conv_Bn_Relu(in_features, n_features, kernel_size, padding=padding, stride=1, group=in_features)]
        body = [Conv_Bn_Relu(n_features,n_features, kernel_size, padding=1, stride=1, depthwise=True) for _ in range(n_res_blocks)]
        tail = [Conv_Bn_Relu(n_features, in_features, kernel_size, padding=1, stride=1, group=n_features)]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.head(input)
        x = self.body(x)
        output = self.tail(x)
        return output

class DFANet(nn.Module):

    def __init__(self, num_classes,patch_size, image_width, image_height, pretrained=False, pretrained_backbone=False, cuda=True):
        super(DFANet, self).__init__()

        self.width = image_width
        self.height = image_height
        self.psize = patch_size
        self.patch_generator = Entropy(patch_size=self.psize, image_width=self.width, image_height=self.height,
                                       cuda=cuda)
        n_feat_hard = 16
        n_feat_med = 8
        n_feat_easy = 4

        self.hardpatch_encoder = PatchEncoderModule(in_features=self.patch_generator.hard_patch_num * 3,
                                                    n_features=n_feat_hard, n_res_blocks=6)
        self.mediumpatch_encoder = PatchEncoderModule(in_features=self.patch_generator.medium_patch_num * 3,
                                                      n_features=n_feat_med, n_res_blocks=6)
        self.easypatch_encoder = PatchEncoderModule(in_features=self.patch_generator.easy_patch_num * 3,
                                                    n_features=n_feat_easy, n_res_blocks=6)

        self.dec_lf1 = Conv_Bn_Relu(in_features=3, out_features=32, kernel_size=3, stride=1, padding=1)
        self.dec_lf2 = Conv_Bn_Relu(in_features=32, out_features=64, kernel_size=3, stride=1, padding=1)
        self.dec_lf3 = Conv_Bn_Relu(in_features=64, out_features=3, kernel_size=3, stride=1, padding=1)

        self.dec_lf_ext = Conv_Bn_Relu(in_features=3, out_features=16, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.common_bn = nn.BatchNorm2d(num_features=num_classes + 16)
        self.final = Conv_Bn_Relu(in_features=num_classes + 16, out_features=num_classes, kernel_size=3, stride=1,
                                  padding=1)

        self.backbone1 = backbone(pretrained=pretrained_backbone)
        self.backbone1_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.backbone2 = backbone(pretrained=pretrained_backbone)
        self.backbone2_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.backbone3 = backbone(pretrained=pretrained_backbone)

        self.decoder = Decoder(n_classes=num_classes)

        if pretrained:
            self.load_state_dict(torch.load(model_url)["state_dict"])

    def patch_to_image(self, image_patches,  indices, num_channels=3) -> torch.Tensor:
        b_size = image_patches.shape[0]
        sorted_patches = image_patches.reshape((-1, num_channels, self.psize, self.psize))[indices, :, :, :]

        reconstructed = sorted_patches.reshape(b_size, int(self.height / self.psize),int(self.width / self.psize), num_channels, self.psize,self.psize).\
            permute(0, 3, 1, 4, 2, 5).\
            reshape(b_size, num_channels, int(self.height), int(self.width))

        return reconstructed

    def forward(self, x):
        hard_p, medium_p, easy_p, indices = self.patch_generator(x)

        hard_dict = self.hardpatch_encoder(hard_p)
        medium_dict = self.mediumpatch_encoder(medium_p)
        easy_dict = self.easypatch_encoder(easy_p)
        feature_ext = torch.cat((hard_dict, medium_dict, easy_dict), dim=1)

        local_descriptors = self.patch_to_image(feature_ext, indices)
        desc_output = self.dec_lf1(local_descriptors)
        desc_output = self.dec_lf2(desc_output)
        desc_output = self.dec_lf3(desc_output)

        local_descriptors_ext1 = self.dec_lf_ext(local_descriptors)

        enc1_2, enc1_3, enc1_4, fc1, fca1 = self.backbone1(x)
        fca1_up = self.backbone1_up(fca1)

        enc2_2, enc2_3, enc2_4, fc2, fca2 = self.backbone2.forward_concat(fca1_up, enc1_2, enc1_3, enc1_4)
        fca2_up = self.backbone2_up(fca2)

        enc3_2, enc3_3, enc3_4, fc3, fca3 = self.backbone3.forward_concat(fca2_up, enc2_2, enc2_3, enc2_4)

        out = self.decoder(enc1_2, enc2_2, enc3_2, fca1, fca2, fca3)

        feature_fuse = self.common_bn(torch.cat((out, local_descriptors_ext1), dim=1))
        output = self.final(feature_fuse)

        return output, local_descriptors, desc_output
