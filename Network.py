import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.utils import spectral_norm

class Vgg19(nn.Module):
    def __init__(self,pretrained=True):
        super(Vgg19,self).__init__()
        self.base_model=models.vgg19(pretrained=pretrained)

    def forward(self,x):
        # get conv4_4
        feature=self.base_model.features
        for i in range(26):
            x=feature[i](x)
        return x

class resnet_block(nn.Module):
    def __init__(self):
        super(resnet_block,self).__init__()
        self.conv1=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        self.conv1_norm=nn.InstanceNorm2d(256)
        self.conv2=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        self.conv2_norm=nn.InstanceNorm2d(256)


    def forward(self,input):
        x=F.relu(self.conv1_norm(self.conv1(input)),True)
        x=self.conv1_norm(self.conv2(x))

        return x+input

class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.down_convs=nn.Sequential(
            nn.Conv2d(3,64,7,1,3,padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,128,3,2,1,padding_mode='reflect'),
            nn.Conv2d(128,128,3,1,1,padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128,256,3,2,1,padding_mode='reflect'),
            nn.Conv2d(256,256,3,1,1,padding_mode='reflect'),
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )
        self.resnet_blocks=[]
        for i in range(8):
            self.resnet_blocks.append(resnet_block())
        self.resnet_blocks=nn.Sequential(*self.resnet_blocks)
        self.up_convs=nn.Sequential(
            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.Conv2d(128,128,3,1,1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.Conv2d(64,64,3,1,1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,3,7,1,3),
            nn.Tanh()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')


    def forward(self,input):
        x=self.down_convs(input)
        x=self.resnet_blocks(x)
        output=self.up_convs(x)

        return output


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.convs=nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(32,64,3,2,1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(128,128,3,2,1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256,1,3,1,1),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,input):
        output=self.convs(input)
        return output

class AnimeDiscriminator(nn.Module):
  def __init__(self, channel: int = 64, nblocks: int = 3) -> None:
    super().__init__()
    channel = channel // 2
    last_channel = channel
    f = [
        spectral_norm(nn.Conv2d(3, channel, 3, stride=1, padding=1, bias=False)),
        nn.LeakyReLU(0.2, inplace=True)
    ]
    in_h = 256
    for i in range(1, nblocks):
      f.extend([
          spectral_norm(nn.Conv2d(last_channel, channel * 2,
                                  3, stride=2, padding=1, bias=False)),
          nn.LeakyReLU(0.2, inplace=True),
          spectral_norm(nn.Conv2d(channel * 2, channel * 4,
                                  3, stride=1, padding=1, bias=False)),
          nn.GroupNorm(1, channel * 4, affine=True),
          nn.LeakyReLU(0.2, inplace=True)
      ])
      last_channel = channel * 4
      channel = channel * 2
      in_h = in_h // 2

    self.body = nn.Sequential(*f)

    self.head = nn.Sequential(*[
        spectral_norm(nn.Conv2d(last_channel, channel * 2, 3,
                                stride=1, padding=1, bias=False)),
        nn.GroupNorm(1, channel * 2, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(channel * 2, 1, 3, stride=1, padding=1, bias=False))])

  def forward(self, x):
    x = self.body(x)
    x = self.head(x)
    return x





class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError

        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch * expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))

        # dw
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        # pw
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out


class AnimeGenerator(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.block_a = nn.Sequential(
            ConvNormLReLU(3, 32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(64, 64)
        )

        self.block_b = nn.Sequential(
            ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(128, 128)
        )

        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )

        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64, 64),
            ConvNormLReLU(64, 32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input, align_corners=False):
        out = self.block_a(input)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out = self.block_c(out)

        if align_corners:
            out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out


class LSGanLoss(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    # NOTE c=b a=0

  def _d_loss(self, real_logit, fake_logit):
    # 1/2 * [(real-b)^2 + (fake-a)^2]
    return 0.5 * (torch.mean((real_logit - 1)**2) + torch.mean(fake_logit**2))

  def _g_loss(self, fake_logit):
    # 1/2 * (fake-c)^2
    return torch.mean((fake_logit - 1)**2)

  def forward(self, real_logit, fake_logit):
    g_loss = self._g_loss(fake_logit)
    d_loss = self._d_loss(real_logit, fake_logit)
    return d_loss, g_loss


# a=torch.randn((3,3,256,256))
# net=discriminator()
# b=net(a)
# loss=nn.BCELoss()
# print(loss(b,c))
# net1=generator()
# d=net1(a)
# print(d.shape)
# net2=Vgg19(pretrained=False)
# e=net2(a)
# a = torch.randn((3,3,256,256))
# net=AnimeGenerator()
# b=net(a)

