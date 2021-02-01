import natsort
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import glob
import random
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

weights_ = '/home/diegushko/checkpoint/cezanne2photo/'
output_ = '/home/diegushko/github/MUNIT_Lite_Pytorch/out_images/'
test_path = '/home/diegushko/github/MUNIT_Lite_Pytorch/test_images/'

class AdaptiveInstanceNorm2d(nn.Module):
    '''
    AdaptiveInstanceNorm2d Class
    Values:
        channels: the number of channels the image has, a scalar
        s_dim: the dimension of the style tensor (s), a scalar
        h_dim: the hidden dimension of the MLP, a scalar
    '''

    def __init__(self, channels, s_dim=8, h_dim=256):
        super().__init__()

        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_scale_transform = self.mlp(s_dim, h_dim, channels)
        self.style_shift_transform = self.mlp(s_dim, h_dim, channels)

    @staticmethod
    def mlp(self, in_dim, h_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, out_dim),
        )

    def forward(self, image, w):
        '''
        Function for completing a forward pass of AdaIN: Given an image and a style, 
        returns the normalized image that has been scaled and shifted by the style.
        Parameters:
          image: the feature map of shape (n_samples, channels, width, height)
          w: the intermediate noise vector w to be made into the style (y)
        '''
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image

class LayerNorm2d(nn.Module):
    '''
    LayerNorm2d Class
    Values:
        channels: number of channels in input, a scalar
        affine: whether to apply affine denormalization, a bool
    '''

    def __init__(self, channels, eps=1e-5, affine=True):
        super().__init__()
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.rand(channels))
            self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.flatten(1).mean(1).reshape(-1, 1, 1, 1)
        std = x.flatten(1).std(1).reshape(-1, 1, 1, 1)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            x = x * self.gamma.reshape(1, -1, 1, 1) + self.beta.reshape(1, -1, 1, 1)

        return x

class ResidualBlock(nn.Module):
    '''
    ResidualBlock Class
    Values:
        channels: number of channels throughout residual block, a scalar
        s_dim: the dimension of the style tensor (s), a scalar
        h_dim: the hidden dimension of the MLP, a scalar
    '''

    def __init__(self, channels, s_dim=None, h_dim=None):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(channels, channels, kernel_size=3)
            ),
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(channels, channels, kernel_size=3)
            ),
        )
        self.use_style = s_dim is not None and h_dim is not None
        if self.use_style:
            self.norm1 = AdaptiveInstanceNorm2d(channels, s_dim, h_dim)
            self.norm2 = AdaptiveInstanceNorm2d(channels, s_dim, h_dim)
        else:
            self.norm1 = nn.InstanceNorm2d(channels)
            self.norm2 = nn.InstanceNorm2d(channels)

        self.activation = nn.ReLU()

    def forward(self, x, s=None):
        x_id = x
        x = self.conv1(x)
        x = self.norm1(x, s) if self.use_style else self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x, s) if self.use_style else self.norm2(x)
        return x + x_id

class ContentEncoder(nn.Module):
    '''
    ContentEncoder Class
    Values:
        base_channels: number of channels in first convolutional layer, a scalar
        n_downsample: number of downsampling layers, a scalar
        n_res_blocks: number of residual blocks, a scalar
    '''

    def __init__(self, base_channels=64, n_downsample=2, n_res_blocks=4):
        super().__init__()

        channels = base_channels

        # Input convolutional layer
        layers = [
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(
                nn.Conv2d(3, channels, kernel_size=7)
            ),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        ]

        # Downsampling layers
        for i in range(n_downsample):
            layers += [
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2)
                ),
                nn.InstanceNorm2d(2 * channels),
                nn.ReLU(inplace=True),
            ]
            channels *= 2

        # Residual blocks
        layers += [
            ResidualBlock(channels) for _ in range(n_res_blocks)
        ]
        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.layers(x)

    @property
    def channels(self):
        return self.out_channels

class StyleEncoder(nn.Module):
    '''
    StyleEncoder Class
    Values:
        base_channels: number of channels in first convolutional layer, a scalar
        n_downsample: number of downsampling layers, a scalar
        s_dim: the dimension of the style tensor (s), a scalar
    '''

    n_deepen_layers = 2

    def __init__(self, base_channels=64, n_downsample=4, s_dim=8):
        super().__init__()

        channels = base_channels

        # Input convolutional layer
        layers = [
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(
                nn.Conv2d(3, channels, kernel_size=7, padding=0)
            ),
            nn.ReLU(inplace=True),
        ]

        # Downsampling layers
        for i in range(self.n_deepen_layers):
            layers += [
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2)
                ),
                nn.ReLU(inplace=True),
            ]
            channels *= 2
        for i in range(n_downsample - self.n_deepen_layers):
            layers += [
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(channels, channels, kernel_size=4, stride=2)
                ),
                nn.ReLU(inplace=True),
            ]

        # Apply global pooling and pointwise convolution to style_channels
        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, s_dim, kernel_size=1),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    '''
    Decoder Class
    Values:
        in_channels: number of channels from encoder output, a scalar
        n_upsample: number of upsampling layers, a scalar
        n_res_blocks: number of residual blocks, a scalar
        s_dim: the dimension of the style tensor (s), a scalar
        h_dim: the hidden dimension of the MLP, a scalar
    '''

    def __init__(self, in_channels, n_upsample=2, n_res_blocks=4, s_dim=8, h_dim=256):
        super().__init__()

        channels = in_channels

        # Residual blocks with AdaIN
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels, s_dim) for _ in range(n_res_blocks)
        ])

        # Upsampling blocks
        layers = []
        for i in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2),
                nn.ReflectionPad2d(2),
                nn.utils.spectral_norm(
                    nn.Conv2d(channels, channels // 2, kernel_size=5)
                ),
                LayerNorm2d(channels // 2),
            ]
            channels //= 2
        
        layers += [
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(
                nn.Conv2d(channels, 3, kernel_size=7)
            ),
            nn.Tanh(),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, s):
        for res_block in self.res_blocks:
            x = res_block(x, s=s)
        x = self.layers(x)
        return x

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        base_channels: number of channels in first convolutional layer, a scalar
        n_downsample: number of downsampling layers, a scalar
        n_res_blocks: number of residual blocks, a scalar
        s_dim: the dimension of the style tensor (s), a scalar
        h_dim: the hidden dimension of the MLP, a scalar
    '''

    def __init__(
        self,
        base_channels: int = 64,
        n_c_downsample: int = 2,
        n_s_downsample: int = 4,
        n_res_blocks: int = 4,
        s_dim: int = 8,
        h_dim: int = 256,
    ):
        super().__init__()
        self.c_enc = ContentEncoder(
            base_channels=base_channels, n_downsample=n_c_downsample, n_res_blocks=n_res_blocks,
        )
        self.s_enc = StyleEncoder(
            base_channels=base_channels, n_downsample=n_s_downsample, s_dim=s_dim,
        )
        self.dec = Decoder(
            self.c_enc.channels, n_upsample=n_c_downsample, n_res_blocks=n_res_blocks, s_dim=s_dim, h_dim=h_dim,
        )

    def encode(self, x):
        content = self.c_enc(x)
        style = self.s_enc(x)
        return (content, style)
    
    def decode(self, content, style):
        return self.dec(content, style)

class Discriminator(nn.Module):
    '''
    Generator Class
    Values:
        base_channels: number of channels in first convolutional layer, a scalar
        n_layers: number of downsampling layers, a scalar
        n_discriminators: number of discriminators (all at different scales), a scalar
    '''

    def __init__(
        self,
        base_channels: int = 64,
        n_layers: int = 3,
        n_discriminators: int = 3,
    ):
        super().__init__()

        self.discriminators = nn.ModuleList([
            self.patchgan_discriminator(base_channels, n_layers) for _ in range(n_discriminators)
        ])

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    @staticmethod
    def patchgan_discriminator(base_channels, n_layers):
        '''
        Function that constructs and returns one PatchGAN discriminator module.
        '''
        channels = base_channels
        # Input convolutional layer
        layers = [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(3, channels, kernel_size=4, stride=2),
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Hidden convolutional layers
        for _ in range(n_layers):
            layers += [
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2)
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels *= 2

        # Output projection layer
        layers += [
            nn.utils.spectral_norm(
                nn.Conv2d(channels, 1, kernel_size=1)
            ),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        for discriminator in self.discriminators:
            outputs.append(discriminator(x))
            x = self.downsample(x)
        return outputs

class GinormousCompositeLoss(nn.Module):
    '''
    GinormousCompositeLoss Class: implements all losses for MUNIT
    '''

    @staticmethod
    def image_recon_loss(x, gen):
        c, s = gen.encode(x)
        recon = gen.decode(c, s)
        return F.l1_loss(recon, x), c, s

    @staticmethod
    def latent_recon_loss(c, s, gen):
        x_fake = gen.decode(c, s)
        recon = gen.encode(x_fake)
        return F.l1_loss(recon[0], c), F.l1_loss(recon[1], s), x_fake

    @staticmethod
    def adversarial_loss(x, dis, is_real):
        preds = dis(x)
        target = torch.ones_like if is_real else torch.zeros_like
        loss = 0.0
        for pred in preds:
            loss += F.mse_loss(pred, target(pred))
        return loss

class MUNIT(nn.Module):
    def __init__(
        self,
        gen_channels: int = 64,
        n_c_downsample: int = 2,
        n_s_downsample: int = 4,
        n_res_blocks: int = 4,
        s_dim: int = 8,
        h_dim: int = 256,
        dis_channels: int = 64,
        n_layers: int = 3,
        n_discriminators: int = 3,
    ):
        super().__init__()

        self.gen_a = Generator(
            base_channels=gen_channels, n_c_downsample=n_c_downsample, n_s_downsample=n_s_downsample, n_res_blocks=n_res_blocks, s_dim=s_dim, h_dim=h_dim,
        )
        self.gen_b = Generator(
            base_channels=gen_channels, n_c_downsample=n_c_downsample, n_s_downsample=n_s_downsample, n_res_blocks=n_res_blocks, s_dim=s_dim, h_dim=h_dim,
        )
        self.dis_a = Discriminator(
            base_channels=dis_channels, n_layers=n_layers, n_discriminators=n_discriminators,
        )
        self.dis_b = Discriminator(
            base_channels=dis_channels, n_layers=n_layers, n_discriminators=n_discriminators,
        )
        self.s_dim = s_dim
        self.loss = GinormousCompositeLoss

    def forward(self, x_a, x_b):
        s_a = torch.randn(x_a.size(0), self.s_dim, 1, 1, device=x_a.device).to(x_a.dtype)
        s_b = torch.randn(x_b.size(0), self.s_dim, 1, 1, device=x_b.device).to(x_b.dtype)

        # Encode real x and compute image reconstruction loss
        x_a_loss, c_a, s_a_fake = self.loss.image_recon_loss(x_a, self.gen_a)
        x_b_loss, c_b, s_b_fake = self.loss.image_recon_loss(x_b, self.gen_b)

        # Decode real (c, s) and compute latent reconstruction loss
        c_b_loss, s_a_loss, x_ba = self.loss.latent_recon_loss(c_b, s_a, self.gen_a)
        c_a_loss, s_b_loss, x_ab = self.loss.latent_recon_loss(c_a, s_b, self.gen_b)

        # Compute adversarial losses
        gen_a_adv_loss = self.loss.adversarial_loss(x_ba, self.dis_a, True)
        gen_b_adv_loss = self.loss.adversarial_loss(x_ab, self.dis_b, True)

        # Sum up losses for gen
        gen_loss = (
            10 * x_a_loss + c_b_loss + s_a_loss + gen_a_adv_loss + \
            10 * x_b_loss + c_a_loss + s_b_loss + gen_b_adv_loss
        )

        # Sum up losses for dis
        dis_loss = (
            self.loss.adversarial_loss(x_ba.detach(), self.dis_a, False) + \
            self.loss.adversarial_loss(x_a.detach(), self.dis_a, True) + \
            self.loss.adversarial_loss(x_ab.detach(), self.dis_b, False) + \
            self.loss.adversarial_loss(x_b.detach(), self.dis_b, True)
        )

        return gen_loss, dis_loss, x_ab, x_ba

weight_path = 'drive/MyDrive/checkpoints/munit/cezanne2photo/'

# Initialize model
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')

munit_config = {
    'gen_channels': 64,
    'n_c_downsample': 2,
    'n_s_downsample': 4,
    'n_res_blocks': 4,
    's_dim': 8,
    'h_dim': 256,
    'dis_channels': 64,
    'n_layers': 3,
    'n_discriminators': 3,
}

munit = MUNIT(**munit_config).to(device)

# Initialize optimizers
gen_params = list(munit.gen_a.parameters()) + list(munit.gen_b.parameters())
dis_params = list(munit.dis_a.parameters()) + list(munit.dis_b.parameters())
gen_optimizer = torch.optim.Adam(gen_params, lr=1e-4, betas=(0.5, 0.999))
dis_optimizer = torch.optim.Adam(dis_params, lr=1e-4, betas=(0.5, 0.999))

pretrained = True
if pretrained:
    pre_dict = torch.load(weights_ + 'MUNIT_base.pth')
    munit.gen_a.load_state_dict(pre_dict['gen_a'])
    munit.gen_b.load_state_dict(pre_dict['gen_b'])
    gen_optimizer.load_state_dict(pre_dict['gen_opt'])
    munit.dis_a.load_state_dict(pre_dict['disc_a'])
    munit.dis_b.load_state_dict(pre_dict['disc_b'])
    dis_optimizer.load_state_dict(pre_dict['dis_opt'])
else:
    gen_AB = munit.gen_a.to(device).apply(weights_init)
    gen_BA = munit.gen_b.to(device).apply(weights_init)
    disc_A = munit.dis_a.to(device).apply(weights_init)
    disc_B = munit.dis_b.to(device).apply(weights_init)

batch_no = 1

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return (tensor_image - 0.5) * 2

dim_A = 3
target_shape = 256

transform_test = transforms.Compose([transforms.Resize((256, 256)),
                               transforms.ToTensor()
                               ])

test_set = CustomDataSet(test_path, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_no, shuffle=False)

with torch.no_grad():
    for idx, img in enumerate(test_loader):
        cuda_img = img.to(device)
        outputs = munit(cuda_img, cuda_img)
        losses, x_ab, x_ba = outputs[:-2], outputs[-2], outputs[-1]
        real_b = nn.functional.interpolate(cuda_img, size=target_shape)
        catenation = torch.cat([real_b[:1, ...], x_ba[:1, ...]])
        image_tensor = (catenation + 1) / 2
        image_shifted = image_tensor
        image_unflat = image_shifted.detach().cpu().view(-1, *(dim_A, target_shape, target_shape))
        image_grid = make_grid(image_unflat[:25], nrow=5)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.savefig(output_ + f'figure_{idx}_rev.png')
        #plt.show()