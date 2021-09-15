import torch
from typing import  Tuple
from torch import nn, Tensor
from torch.nn.utils import  spectral_norm
import torch.nn.functional as F

class Entropy(nn.Sequential):
    def __init__(self, patch_size, image_width, image_height, cuda):
        super(Entropy, self).__init__()
        self.width = image_width
        self.height = image_height
        self.psize = patch_size
        #number of patches per image
        self.patch_num = int(self.width * self.height / self.psize ** 2)
        #unfolding image to non overlapping patches
        self.unfold = torch.nn.Unfold(kernel_size=(self.psize, self.psize), stride=self.psize)
        self.hard_patch_num = int(self.patch_num / 5)
        self.medium_patch_num = int(2 * self.patch_num / 5)
        self.easy_patch_num = self.patch_num - self.hard_patch_num - self.medium_patch_num


    def entropy(self, values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor, batch: int, epsilon: float = 1e-10) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Function that calculates the entropy using marginal probability distribution function of the input tensor
            based on the number of histogram bins.
        Args:
            values: shape [BxNx1].
            bins: shape [NUM_BINS].
            sigma: shape [1], gaussian smoothing factor.
            epsilon: scalar, for numerical stability.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
              - torch.Tensor: shape [BxN].
              - torch.Tensor: shape [BxNxNUM_BINS].
        """
        values = values.unsqueeze(2)
        residuals = values - bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
        pdf = pdf / normalization
        entropy = - torch.sum(pdf * torch.log(pdf), dim=1)
        entropy = entropy.reshape((batch, -1))
        return entropy

    def image_to_patches(self, input):
        image_patches = input.unfold(2, self.psize, self.psize).\
            unfold(3, self.psize,self.psize).permute(0, 2, 3, 1, 4, 5).\
            reshape(-1, 3, self.psize, self.psize)
        return image_patches

    def forward(self, input: Tensor) -> Tuple:
        batch_size = input.shape[0]
        gray_images = 0.2989 * input[:, 0:1, :, :] + 0.5870 * input[:, 1:2, :, :] + 0.1140 * input[:, 2:, :, :]
        # create patches of size (batch x patch_size*patch_size x h*w/ (patch_size*patch_size))
        unfolded_images = self.unfold(gray_images)

        # reshape to (batch * h*w/ (patch_size*patch_size x patch_size*patch_size)
        unfolded_images = unfolded_images.transpose(1, 2)
        unfolded_images = torch.reshape(unfolded_images.unsqueeze(2), (unfolded_images.shape[0] * self.patch_num, unfolded_images.shape[2]))

        entropy = self.entropy(unfolded_images, bins=torch.linspace(0, 1, 32).to(device=input.device), sigma=torch.tensor(0.01), batch=batch_size) + 1e-40
        sorted_entropy = torch.argsort(entropy, dim=1, descending=True)
        image_patches = self.image_to_patches(input)
        batch_idc_helper = torch.arange(start=0, end=image_patches.shape[0], step=self.patch_num).to(device=input.device)

        max_ent_idc = sorted_entropy[:, 0 : self.hard_patch_num].flatten() + batch_idc_helper.repeat_interleave(self.hard_patch_num)
        med_ent_idc = sorted_entropy[:,self.hard_patch_num : self.hard_patch_num + self.medium_patch_num].flatten() + batch_idc_helper.repeat_interleave(self.medium_patch_num)
        low_ent_idc = sorted_entropy[:,self.hard_patch_num + self.medium_patch_num :].flatten() + batch_idc_helper.repeat_interleave(self.easy_patch_num)

        hard_patches = image_patches[max_ent_idc, :, :, :].reshape((batch_size, -1, self.psize, self.psize))
        medium_patches = image_patches[med_ent_idc, :, :, :].reshape((batch_size, -1, self.psize, self.psize))
        easy_patches = image_patches[low_ent_idc, :, :, :].reshape((batch_size, -1, self.psize, self.psize))

        # construct indices for image reconstruction
        indices = sorted_entropy.flatten() + batch_idc_helper.repeat_interleave(self.patch_num)
        indices = torch.zeros((batch_size * self.patch_num), dtype=torch.long).to(input.device).put_(indices.to(input.device),torch.arange(0,batch_size * self.patch_num, dtype=torch.long).to(input.device)).to(input.device)

        return hard_patches, medium_patches, easy_patches, indices

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, stride=1, padding=1):
        super(ResBlock, self).__init__()
        if stride == 1:
            self.res_layer = nn.Identity()
        else:
            self.res_layer = nn.Conv2d(n_feats, n_feats, kernel_size, bias=True, padding=padding, stride=stride)
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, bias=True, padding=padding, stride=stride))
            m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(nn.ReLU(True))
                stride = 1
        self.body = nn.Sequential(*m)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        res = self.body(x)
        res += self.res_layer(x)
        res = self.relu(res)
        return res

class Conv_Bn_Relu(nn.Sequential):
    def __init__(self, in_features, out_features, kernel_size, stride, padding, group=1):
        super(Conv_Bn_Relu, self).__init__()
        if group > 1:
            conv_block = [nn.Conv2d(in_features, in_features, kernel_size, stride, padding, groups=group),
                         nn.Conv2d(in_features, out_features, kernel_size=1)]
            self.conv = nn.Sequential(*conv_block)
        else:
            self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(True)

    def forward(self, input: Tensor) -> Tensor:
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Upsample2x(nn.Sequential):
    def __init__(self, in_features):
        super(Upsample2x, self).__init__()

        self.conv = nn.Conv2d(in_features, in_features * 4, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, input: Tensor) -> Tensor:
        x = self.conv(input)
        x = self.ps(x)

        return x

class PatchEncoderModule(nn.Sequential):
    def __init__(self,in_features, n_features, n_res_blocks, kernel_size=3, padding=1):
        super(PatchEncoderModule, self).__init__()

        # define encoder module
        head = [Conv_Bn_Relu(in_features, n_features, kernel_size, padding=padding, stride=1)]
        body = [Conv_Bn_Relu(n_features,n_features, kernel_size, padding=1, stride=1) for _ in range(n_res_blocks)]
        tail = [Conv_Bn_Relu(n_features, int(n_features / 2), kernel_size, padding=1, stride=1)]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)


    def forward(self, input: Tensor) -> Tensor:
        x = self.head(input)
        x = self.body(x)
        output = self.tail(x)
        output = F.interpolate(output, scale_factor=8, mode='bilinear', align_corners=True)
        return output


class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        super(SelfAttention, self).__init__()
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def _conv(self,n_in,n_out):
        return spectral_norm(nn.Conv1d(n_in, n_out, kernel_size=1,  bias=False))

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()

class SRDModel(nn.Sequential):
    def __init__(self, patch_size, image_width, image_height, num_classes, cuda=True):
        super(SRDModel, self).__init__()
        self.width = image_width
        self.height = image_height
        self.psize = patch_size
        self.patch_generator = Entropy(patch_size=self.psize, image_width=self.width, image_height=self.height, cuda=cuda)
        n_feat_hard = 128
        n_feat_med = 64
        n_feat_easy = 32

        self.hardpatch_encoder = PatchEncoderModule(in_features=self.patch_generator.hard_patch_num * 3, n_features=n_feat_hard, n_res_blocks=4)
        self.mediumpatch_encoder = PatchEncoderModule(in_features=self.patch_generator.medium_patch_num * 3, n_features=n_feat_med, n_res_blocks=3)
        self.easypatch_encoder = PatchEncoderModule(in_features=self.patch_generator.easy_patch_num * 3, n_features=n_feat_easy, n_res_blocks=2)


        # define encoder module
        self.head_depthwise = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=4, padding=2, groups=3)
        self.head_separable = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1)
        self.enc_block1 = ResBlock(n_feats=32, kernel_size=3, stride=2)
        self.enc_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)

        self.enc_block2 = ResBlock(n_feats=64, kernel_size=3, stride=2)
        self.enc_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)

        self.enc_block3 = ResBlock(n_feats=128, kernel_size=3, stride=2)
        self.enc_conv3 = Conv_Bn_Relu(in_features=128, out_features=256, kernel_size=1, padding=0, stride=1)
        num_patches = int((n_feat_hard + n_feat_med + n_feat_easy) / 2)

        # num_patches = 0
        # define decoder module
        self.dec_head = Conv_Bn_Relu(in_features=256, out_features=128, kernel_size=3, stride=1, padding=1,
                                     group=256 )
        self.dec_block1 = Upsample2x(in_features=128)
        self.dec_conv1 = Conv_Bn_Relu(in_features=128, out_features=64, kernel_size=3, stride=1, padding=1, group=128)

        self.dec_block2 = Upsample2x(in_features=64)
        self.dec_conv2 = Conv_Bn_Relu(in_features=64, out_features=32, kernel_size=3, stride=1, padding=1, group=64)

        self.inter_conv = Conv_Bn_Relu(in_features=32 +num_patches, out_features=32, kernel_size=3,stride=1, padding=1)
        self.dec_block3 = Upsample2x(in_features=32 )
        self.dec_conv3 = Conv_Bn_Relu(in_features=32 + num_patches, out_features=16 * num_classes, kernel_size=3, stride=1, padding=1 )

        self.tail = nn.PixelShuffle(4)

    def forward(self, input: Tensor) -> Tensor:

        hard_p, medium_p, easy_p, indices = self.patch_generator(input)

        hard_dict = self.hardpatch_encoder(hard_p)
        medium_dict = self.mediumpatch_encoder(medium_p)
        easy_dict = self.easypatch_encoder(easy_p)

        enc_head = self.head_depthwise(input)
        enc_head = self.head_separable(enc_head) # 32

        enc1 = self.enc_block1(enc_head) #32
        enc1_out = self.enc_conv1(enc1) #64

        enc2 = self.enc_block2(enc1_out) #64
        enc2_out = self.enc_conv2(enc2) # 128

        enc3 = self.enc_block3(enc2_out) #128
        enc3_out = self.enc_conv3(enc3) #256

        dec_head = self.dec_head(enc3_out) # 128

        dec1 = self.dec_block1(dec_head) #128
        dec1 = self.dec_conv1(dec1) #64
        dec_add1 = torch.add(dec1, enc2) #128

        dec2 = self.dec_block2(dec_add1) #64
        dec2 = self.dec_conv2(dec2) #32
        dec_add2 = torch.add(dec2, enc1) #64

        dec3 = self.dec_block3(dec_add2) #32
        feature_fuse = torch.cat((dec3, hard_dict, medium_dict, easy_dict), dim=1) 
        dec3 = self.dec_conv3(feature_fuse) #16

        output = self.tail(dec3)

        return output

