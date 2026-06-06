import torch
import torch.nn as nn
from models import register


@register('cnn')
class CNN(nn.Module):
    def __init__(self, input_channels, output_channels, init_range=1):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        # self.conv1 = nn.ConvTranspose2d(input_channels, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.convtrans = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        self._initialize_weights(init_range)

    def _calculate_flatten_size(self, input_size):
        h, w = input_size
        h, w = h // 8, w // 8
        self.flatten_size = 128 * h * w

    def _initialize_weights(self, init_range):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -init_range, init_range)

    def forward(self, x):
        x1 = self.leaky_relu(self.conv1(x)) 
        x2 = self.leaky_relu(self.conv2(x1)) 
        x3 = self.leaky_relu(self.conv3(x2)) 
        x4 = self.leaky_relu(self.conv4(x3))
        x5 = self.leaky_relu(self.conv5(x4))
        x5 = x3 + x5
        x6 = self.leaky_relu(self.conv6(x5))
        x6 = x2 + x6
        x7 = self.leaky_relu(self.conv7(x6))
        x7 = x1 + x7
        x7 = self.leaky_relu(self.convtrans(x7))
        x8 = self.conv8(x7)
        
        return x8, x7

