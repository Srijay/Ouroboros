import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn.functional as F

from generators import tissue_image_generator
from resnet import ResNet

class GenerativeModel(nn.Module):

    def __init__(self, mode='train',
                 normalization='instance', activation='leakyrelu-0.2', generator_name='dcgan',
                 **kwargs):
        super(GenerativeModel, self).__init__()
        self.mode = mode
        input_channels = 38
        self.image_generator = tissue_image_generator(input_dim=input_channels,
                                                      output_nc=3,
                                                      generator_name=generator_name,
                                                      n_blocks_global=2,
                                                      n_downsample_global=3,
                                                      ngf=64,
                                                      norm='instance')
        self.image_generator.cuda()

    def forward(self, hyperion_features):
        generated_image = self.image_generator(hyperion_features)
        return generated_image