import functools
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size=512):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.layer4 = nn.Linear(hidden_size // 4, hidden_size // 8)
        self.layer5 = nn.Linear(hidden_size // 8, input_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, -1)  # 展平
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = x.view(b, c, h, w)
        return x
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

class CNN(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=in_channels, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )
        self.cnn1 = CNN(64)
        self.cnn2 = CNN(128)
        self.cnn3 = CNN(256)
        self.cnn4 = CNN(512)
        self.mlp1 = MLP(512*1*1)
        self.mlp2 = MLP(512*2*2)
        self.mlp3 = MLP(512*4*4)
        self.mlp4 = MLP(512*8*8)


    def forward(self, x):
        features1, features2 = [], []
        d1 = self.down1(x); features1.append(d1)
        x1 = self.cnn1(d1); features2.append(x1)
        d2 = self.down2(d1); features1.append(d2)
        x2 = self.cnn2(d2); features2.append(x2)
        d3 = self.down3(d2); features1.append(d3)
        x3 = self.cnn3(d3); features2.append(x3)
        d4 = self.down4(d3); features1.append(d4)
        x4 = self.cnn4(d4); features2.append(x4)
        d5 = self.down5(d4); features1.append(d5)
        x5 = self.mlp4(d5); features2.append(x5)
        d6 = self.down6(d5); features1.append(d6)
        x6 = self.mlp3(d6); features2.append(x6)
        d7 = self.down7(d6); features1.append(d7)
        x7 = self.mlp2(d7); features2.append(x7)
        d8 = self.down8(d7); features1.append(d8)
        x8 = self.mlp1(d8); features2.append(x8)
        u1 = self.up1(self.mlp1(d8), self.mlp2(d7))
        u2 = self.up2(u1, self.mlp3(d6))
        u3 = self.up3(u2, self.mlp4(d5))
        u4 = self.up4(u3, self.cnn4(d4))
        u5 = self.up5(u4, self.cnn3(d3))
        u6 = self.up6(u5, self.cnn2(d2))
        u7 = self.up7(u6, self.cnn1(d1))

        return self.final(u7), features1, features2

    def freeze_cm(self):
        self.mlp1.freeze()
        self.mlp2.freeze()
        self.mlp3.freeze()
        self.mlp4.freeze()
        self.cnn1.freeze()
        self.cnn2.freeze()
        self.cnn3.freeze()
        self.cnn4.freeze()
    def eval_cm(self):
        self.mlp1.eval()
        self.mlp2.eval()
        self.mlp3.eval()
        self.mlp4.eval()
        self.cnn1.eval()
        self.cnn2.eval()
        self.cnn3.eval()
        self.cnn4.eval()

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
class CBAMDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(CBAMDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            CBAM(64),  # Add CBAM after each block
            *discriminator_block(64, 128),
            CBAM(128),
            *discriminator_block(128, 256),
            CBAM(256),
            *discriminator_block(256, 512),
            CBAM(512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, feature1, feature2):
        similarity = self.cos_sim(feature1, feature2)
        loss = 1 - similarity.mean()  # 目标是最大化相似度，因此损失为 1 - 相似度
        return loss