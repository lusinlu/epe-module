import torch
from typing import Tuple
from torch import nn, Tensor
from efficentnet import EfficientNet


class Entropy(nn.Sequential):
    def __init__(self, patch_size, image_width, image_height, cuda):
        super(Entropy, self).__init__()
        self.width = image_width
        self.height = image_height
        self.psize = patch_size
        # number of patches per image
        self.patch_num = int(self.width * self.height / self.psize ** 2)
        # operation for unfolding image to non overlapping patches
        self.unfold = torch.nn.Unfold(kernel_size=(self.psize, self.psize), stride=self.psize)
        # defining number of hard, medium and easy patches (20%, 40%, 40%)
        self.hard_patch_num = int(self.patch_num / 5)
        self.medium_patch_num = int(2 * self.patch_num / 5)
        self.easy_patch_num = self.patch_num - self.hard_patch_num - self.medium_patch_num



    def entropy(self, values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor, batch: int, epsilon: float = 1e-10) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Function that calculates the entropy using KDE.
        Args:
            values: shape [BxNx1].
            bins: shape [NUM_BINS]. for uniform quantization
            sigma: shape [1], gaussian smoothing factor.
            epsilon: scalar, for numerical stability.
        Returns:
             torch.Tensor: entropies of each patch
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
            reshape(-1, input.shape[1], self.psize, self.psize)
        return image_patches

    def forward(self, input: Tensor) -> Tuple:
        batch_size = input.shape[0]
        # convert RGB to grayscale image
        gray_images = 0.2989 * input[:, 0:1, :, :] + 0.5870 * input[:, 1:2, :, :] + 0.1140 * input[:, 2:, :, :]

        # create patches of size (batch x patch_size*patch_size x h*w/ (patch_size*patch_size))
        unfolded_images = self.unfold(gray_images)

        # reshape to (batch x h*w/ (patch_size*patch_size) x (patch_size*patch_size)
        unfolded_images = unfolded_images.transpose(1, 2)
        unfolded_images = torch.reshape(unfolded_images.unsqueeze(2), (unfolded_images.shape[0] * self.patch_num, unfolded_images.shape[2]))

        entropy = self.entropy(unfolded_images, bins=torch.linspace(0, 1, 32).to(device=input.device), sigma=torch.tensor(0.01), batch=batch_size) + 1e-40
        # sort entropies for further image patch grouping by entropy values
        sorted_entropy = torch.argsort(entropy, dim=1, descending=True)

        image_patches = self.image_to_patches(input)
        # helper tensor for extraction of patches
        batch_idc_helper = torch.arange(start=0, end=image_patches.shape[0], step=self.patch_num).to(device=input.device)
        # indices of the patches for 'hard', 'medium' and 'easy' patch groups
        max_ent_idc = sorted_entropy[:, 0 : self.hard_patch_num].flatten() + batch_idc_helper.repeat_interleave(self.hard_patch_num)
        med_ent_idc = sorted_entropy[:,self.hard_patch_num : self.hard_patch_num + self.medium_patch_num].flatten() + batch_idc_helper.repeat_interleave(self.medium_patch_num)
        low_ent_idc = sorted_entropy[:,self.hard_patch_num + self.medium_patch_num :].flatten() + batch_idc_helper.repeat_interleave(self.easy_patch_num)
        # groups of patches
        hard_patches = image_patches[max_ent_idc, :, :, :].reshape((batch_size, -1, self.psize, self.psize))
        medium_patches = image_patches[med_ent_idc, :, :, :].reshape((batch_size, -1, self.psize, self.psize))
        easy_patches = image_patches[low_ent_idc, :, :, :].reshape((batch_size, -1, self.psize, self.psize))

        # saving indices for further patches-to-image
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


class Conv_Block_Patches(nn.Sequential):
    def __init__(self, in_features, out_features, kernel_size, stride, padding, group=1):
        super(Conv_Block_Patches, self).__init__()
        self.conv = nn.Conv2d(in_features, in_features, kernel_size, stride, padding=0, groups=group)
        self.padd = torch.nn.ReflectionPad2d(padding)

        self.inst_norm = nn.InstanceNorm2d(out_features)
        self.relu = nn.ReLU(True)

    def forward(self, input: Tensor) -> Tensor:
        x = self.conv(input)
        x = self.padd(x)
        x = self.inst_norm(x)
        x = self.relu(x)

        return x


class Conv_Bn_Relu(nn.Sequential):
    def __init__(self, in_features, out_features, kernel_size, stride, padding, group=1, depthwise=False):
        super(Conv_Bn_Relu, self).__init__()
        if group > 1:
            conv_block = [nn.Conv2d(in_features, in_features, kernel_size, stride, padding, groups=group),
                         nn.Conv2d(in_features, out_features, kernel_size=1)]
            self.conv = nn.Sequential(*conv_block)
        elif depthwise ==True:
            self.conv = nn.Conv2d(in_features, in_features, kernel_size, stride, padding, groups=group)
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
    def __init__(self, in_features, bilinear=False):
        super(Upsample2x, self).__init__()
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.conv = nn.Conv2d(in_features, in_features * 4, kernel_size=3, padding=1)
            self.ps = nn.PixelShuffle(2)
            self.up = nn.Sequential(self.conv, self.ps)

    def forward(self, input):
        x = self.up(input)

        return x


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


# class of real-time efficient net for semantic segmentation
class RTEffNet(nn.Sequential):
    def __init__(self, patch_size, image_width, image_height, num_classes, cuda=True):
        super(RTEffNet, self).__init__()
        self.width = image_width
        self.height = image_height
        self.psize = patch_size
        self.patch_generator = Entropy(patch_size=self.psize, image_width=self.width, image_height=self.height, cuda=cuda)
        n_feat_hard = 32
        n_feat_med = 16
        n_feat_easy = 8
        # patch encoders
        self.hardpatch_encoder = PatchEncoderModule(in_features=self.patch_generator.hard_patch_num * 3, n_features=n_feat_hard, n_res_blocks=6)
        self.mediumpatch_encoder = PatchEncoderModule(in_features=self.patch_generator.medium_patch_num * 3, n_features=n_feat_med, n_res_blocks=6)
        self.easypatch_encoder = PatchEncoderModule(in_features=self.patch_generator.easy_patch_num * 3, n_features=n_feat_easy, n_res_blocks=6)
        self.dec_lf_ext = Conv_Bn_Relu(in_features=3, out_features=32, kernel_size=3, stride=1, padding=1 )

        # additional convs for the MSE calculation (between the input and output of EPE module)
        self.dec_lf1 = Conv_Bn_Relu(in_features=3, out_features=32, kernel_size=3, stride=1, padding=1 )
        self.dec_lf2 = Conv_Bn_Relu(in_features=32, out_features=64, kernel_size=3, stride=1, padding=1 )
        self.dec_lf3 = Conv_Bn_Relu(in_features=64, out_features=3, kernel_size=3, stride=1, padding=1 )

        # define encoder module
        self.encoder = EfficientNet.from_pretrained('efficientnet-b6')

        # define decoder module
        self.dec_block2 = Upsample2x(in_features=200)
        self.dec_conv2 = Conv_Bn_Relu(in_features=200, out_features=72, kernel_size=3, stride=1, padding=1)

        self.dec_block3 = Upsample2x(in_features=72 )
        self.dec_conv3 = Conv_Bn_Relu(in_features=72, out_features= 40, kernel_size=3, stride=1, padding=1)

        self.dec_block4 = Upsample2x(in_features=40 )
        self.dec_conv4 = Conv_Bn_Relu(in_features=40, out_features= 32, kernel_size=3, stride=1, padding=1)

        # final layers
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.common_bn = nn.BatchNorm2d(num_features=64)
        self.final = Conv_Bn_Relu(in_features=64, out_features= num_classes, kernel_size=3, stride=1, padding=1)
        self.tail = Upsample2x(num_classes)

    def patch_to_image(self, image_patches,  indices, num_channels=3) -> Tuple[Tensor, Tensor, Tensor]:
        b_size = image_patches.shape[0]
        sorted_patches = image_patches.reshape((-1, num_channels, self.psize, self.psize))[indices, :, :, :]

        reconstructed = sorted_patches.reshape(b_size, int(self.height / self.psize),int(self.width / self.psize), num_channels, self.psize,self.psize).\
            permute(0, 3, 1, 4, 2, 5).\
            reshape(b_size, num_channels, int(self.height), int(self.width))

        return reconstructed

    def forward(self, input: Tensor) -> (Tensor, Tensor, Tensor):

        hard_p, medium_p, easy_p, indices = self.patch_generator(input)

        hard_dict = self.hardpatch_encoder(hard_p)
        medium_dict = self.mediumpatch_encoder(medium_p)
        easy_dict = self.easypatch_encoder(easy_p)
        feature_ext = torch.cat((hard_dict, medium_dict, easy_dict ), dim=1)

        local_descriptors = self.patch_to_image(feature_ext, indices)
        desc_output = self.dec_lf1(local_descriptors)
        desc_output = self.dec_lf2(desc_output)
        desc_output = self.dec_lf3(desc_output)

        local_descriptors_ext1 = self.dec_lf_ext(local_descriptors)
        # ##################################

        endpoints = self.encoder.extract_endpoints(input)
        enc4, enc3, enc2, enc1 = endpoints['reduction_4'], endpoints['reduction_3'], endpoints['reduction_2'], endpoints['reduction_1']

        dec2 = self.dec_block2(enc4)
        dec2 = self.dec_conv2(dec2)
        dec_add2 = torch.add(dec2, enc3)

        dec3 = self.dec_block3(dec_add2)
        dec3 = self.dec_conv3(dec3)
        dec_add3 = torch.add(dec3, enc2)

        dec4 = self.dec_block4(dec_add3)
        dec4 = self.dec_conv4(dec4)
        dec_cat4 = torch.add(dec4, enc1)

        feature_fuse = self.common_bn(torch.cat((dec_cat4, self.avg_pool(local_descriptors_ext1)), dim=1))
        output = self.final(feature_fuse)
        output = self.tail(output)

        return output, local_descriptors, desc_output

