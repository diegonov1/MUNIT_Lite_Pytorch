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
torch.manual_seed(42)

PATH = '/home/diegushko/dataset/cezanne2photo'
Weights = '/home/diegushko/checkpoint/cezanne2photo/'
output = '/home/diegushko/output/cezanne2photo/'

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        super().__init__()
        self.transform = transform
        self.files_A = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*')) #This is switched for art2photo dataset
        self.files_B = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*')) #This is switched for art2photo dataset
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()
        assert len(self.files_A) > 0, "Make sure you downloaded the images!"

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[self.randperm[index]]))
        if item_A.shape[0] != 3: 
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3: 
            item_B = item_B.repeat(3, 1, 1)
        if index == len(self) - 1:
            self.new_perm()
        return item_A, item_B

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))

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
        base_channels: int = 32, #Changed from 64 to 32 <=====
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

munit = MUNIT(**munit_config)

# Initialize optimizers
gen_params = list(munit.gen_a.parameters()) + list(munit.gen_b.parameters())
dis_params = list(munit.dis_a.parameters()) + list(munit.dis_b.parameters())
gen_optimizer = torch.optim.Adam(gen_params, lr=1e-4, betas=(0.5, 0.999))
dis_optimizer = torch.optim.Adam(dis_params, lr=1e-4, betas=(0.5, 0.999))

pretrained = False
if pretrained:
    pre_dict = torch.load(Weights + 'MUNIT_base.pth')
    munit.gen_a.load_state_dict(pre_dict['gen_a'])
    munit.gen_b.load_state_dict(pre_dict['gen_b'])
    gen_optimizer.load_state_dict(pre_dict['gen_opt'])
    munit.dis_a.load_state_dict(pre_dict['dis_a'])
    munit.dis_b.load_state_dict(pre_dict['dis_b'])
    dis_optimizer.load_state_dict(pre_dict['dis_opt'])
else:
    gen_AB = munit.gen_a.to(device).apply(weights_init)
    gen_BA = munit.gen_b.to(device).apply(weights_init)
    disc_A = munit.dis_a.to(device).apply(weights_init)
    disc_B = munit.dis_b.to(device).apply(weights_init)

# Initialize dataloader
transform = transforms.Compose([
    transforms.Resize(286),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
dataloader = DataLoader(
    ImageDataset(PATH, transform),
    batch_size=1, pin_memory=True, shuffle=True,
)

# Parse torch version for autocast
# ######################################################
version = torch.__version__
version = tuple(int(n) for n in version.split('.')[:-1])
has_autocast = version >= (1, 6)
# ######################################################

def train(munit, dataloader, optimizers, device):

    save_model = True
    max_iters = 1000000
    decay_every = 100000
    cur_iter = 0
    epochs = 300

    display_every = 500
    mean_losses = [0., 0.]

    for epoch in range(epochs):

        if cur_iter == max_iters:
            break

        for (x_a, x_b) in tqdm(dataloader):
            x_a = x_a.to(device)
            x_b = x_b.to(device)

            # Enable autocast to FP16 tensors (new feature since torch==1.6.0)
            # If you're running older versions of torch, comment this out
            # and use NVIDIA apex for mixed/half precision training
            if has_autocast:
                with torch.cuda.amp.autocast(enabled=(device=='cuda:1')):
                    outputs = munit(x_a, x_b)
            else:
                outputs = munit(x_a, x_b)
            
            losses, x_ab, x_ba = outputs[:-2], outputs[-2], outputs[-1]
            munit.zero_grad()

            for i, (optimizer, loss) in enumerate(zip(optimizers, losses)):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_losses[i] += loss.item() / display_every

            cur_iter += 1

            if cur_iter % display_every == 0:
                print('Step {}: [G loss: {:.5f}][D loss: {:.5f}]'
                      .format(cur_iter, *mean_losses))
                #show_tensor_images(x_ab, x_a)
                #show_tensor_images(x_ba, x_b)
                mean_losses = [0., 0.]

            # Schedule learning rate by 0.5
            if cur_iter % decay_every == 0:
                for optimizer in optimizers:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5

        if save_model:
            if epoch > 0:
                numeral = epoch - 1
                os.remove(Weights + f"MUNIT_{numeral}_c32s8.pth")

            torch.save({
                'gen_a': munit.gen_a.state_dict(),
                'gen_b': munit.gen_b.state_dict(),
                'gen_opt': gen_optimizer.state_dict(),
                'disc_a': munit.dis_a.state_dict(),
                'disc_b': munit.dis_b.state_dict(),
                'dis_opt': dis_optimizer.state_dict()
            }, Weights + f"MUNIT_{epoch}_c32s8.pth")
            print('Saved weights for iteration {}'.format(epoch))

train(
    munit, dataloader,
    [gen_optimizer, dis_optimizer],
    device,
)


