import torch
import torch.nn as nn
import torch.nn.functional as F
import biggan_layers
import functools
from torch.nn import init
import torch.optim as optim
from layers import GlobalAvgPool, Flatten, get_activation, build_cnn


class PatchDiscriminator(nn.Module):
  def __init__(self, arch, normalization='batch', activation='leakyrelu-0.2',
               padding='same', pooling='avg', input_size=(256,256),
               layout_dim=0):
    super(PatchDiscriminator, self).__init__()
    #input_dim = 1 + layout_dim
    input_dim = 3

    arch = 'I%d,%s' % (input_dim, arch)

    cnn_kwargs = {
      'arch': arch,
      'normalization': normalization,
      'activation': activation,
      'pooling': pooling,
      'padding': padding,
    }
    self.cnn, output_dim = build_cnn(**cnn_kwargs)
    self.classifier = nn.Conv2d(output_dim, 1, kernel_size=1, stride=1)

  def forward(self, x, layout=None):
    if layout is not None:
      x = torch.cat([x, layout], dim=1)
    return self.cnn(x)


class Pix2PixDiscriminator(nn.Module):

  def __init__(self, in_channels=3):
    super(Pix2PixDiscriminator, self).__init__()

    def discriminator_block(in_filters, out_filters, stride=2, normalization=True):
      """Returns downsampling layers of each discriminator block"""
      layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
      if normalization:
        layers.append(nn.InstanceNorm2d(out_filters))
      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers

    self.model = nn.Sequential(

      #If model loading failed, try this discriminator and keep kernel size as 5
      # *discriminator_block(in_channels, 16, normalization=False),
      # *discriminator_block(16, 32),
      # *discriminator_block(32, 64),
      # *discriminator_block(64, 128),
      # *discriminator_block(128, 256),
      # #nn.ZeroPad2d((1, 0, 1, 0)),
      # nn.Conv2d(256, 1, 5, padding=1, bias=False),
      # #nn.ReLU()

      *discriminator_block(in_channels, 64, normalization=False),
      *discriminator_block(64, 128),
      *discriminator_block(128, 256),
      # *discriminator_block(256, 512), #Added to safronize framework
      *discriminator_block(256, 512, stride=1),
      # nn.ZeroPad2d((1, 0, 1, 0)),
      nn.Conv2d(512, 1, 4, stride=1, padding=1, bias=False),
      # nn.Sigmoid() #It was not there when trained with residual generator so turn it off while its inference
    )

  def forward(self, img):
    # Concatenate image and condition image by channels to produce input
    # img_input = torch.cat((mask, img), 1)
    # output = checkpoint_sequential(self.model, 1, img)
    output = self.model(img)
    return output


class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, which_conv=biggan_layers.SNConv2d, wide=True,
               preactivation=True, activation=None, downsample=None,
               channel_ratio=4):
    super(DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels // channel_ratio
    self.which_conv = which_conv
    self.preactivation = preactivation
    self.activation = activation
    self.downsample = downsample

    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels,
                                 kernel_size=1, padding=0)
    self.conv2 = self.which_conv(self.hidden_channels, self.hidden_channels)
    self.conv3 = self.which_conv(self.hidden_channels, self.hidden_channels)
    self.conv4 = self.which_conv(self.hidden_channels, self.out_channels,
                                 kernel_size=1, padding=0)

    self.learnable_sc = True if (in_channels != out_channels) else False
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels - in_channels,
                                     kernel_size=1, padding=0)

  def shortcut(self, x):
    if self.downsample:
      x = self.downsample(x)
    if self.learnable_sc:
      x = torch.cat([x, self.conv_sc(x)], 1)
    return x

  def forward(self, x):
    # 1x1 bottleneck conv
    h = self.conv1(F.relu(x))
    # 3x3 convs
    h = self.conv2(self.activation(h))
    h = self.conv3(self.activation(h))
    # relu before downsample
    h = self.activation(h)
    # downsample
    if self.downsample:
      h = self.downsample(h)
      # final 1x1 conv
    h = self.conv4(h)
    return h + self.shortcut(x)


class BigGanDiscriminator(nn.Module):
  def __init__(self, D_ch=64, D_wide=True, D_depth=2, resolution=128,
               D_kernel_size=3, D_attn='64', n_classes=1000,
               num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
               D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
               SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
               D_init='ortho', skip_init=False, D_param='SN', **kwargs):
    super(BigGanDiscriminator, self).__init__()
    # Width multiplier
    self.ch = D_ch
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.D_wide = D_wide
    # How many resblocks per stage?
    self.D_depth = D_depth
    # Resolution
    self.resolution = resolution
    # Kernel size
    self.kernel_size = D_kernel_size
    # Attention?
    self.attention = D_attn
    # Number of classes
    self.n_classes = n_classes
    # Activation
    self.activation = D_activation
    # Initialization style
    self.init = D_init
    # Parameterization style
    self.D_param = D_param
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # Fp16?
    self.fp16 = D_fp16

    # Architecture
    ch = 64
    attention = '64'
    self.arch = {'in_channels' :  [item * ch for item in [1, 2, 4, 8, 8, 16]],
               'out_channels' : [item * ch for item in [2, 4, 8, 8, 16, 16]],
               'downsample' : [True] * 6 + [False],
               'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}

    # Which convs, batchnorms, and linear layers to use
    # No option to turn off SN in D right now
    if self.D_param == 'SN':
      self.which_conv = functools.partial(biggan_layers.SNConv2d,
                                          kernel_size=3, padding=1,
                                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                          eps=self.SN_eps)
      self.which_linear = functools.partial(biggan_layers.SNLinear,
                                            num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                            eps=self.SN_eps)
      self.which_embedding = functools.partial(biggan_layers.SNEmbedding,
                                               num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                               eps=self.SN_eps)

    # Prepare model
    # Stem convolution
    self.input_conv = self.which_conv(3, self.arch['in_channels'][0])
    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [
        [DBlock(in_channels=self.arch['in_channels'][index] if d_index == 0 else self.arch['out_channels'][index],
                out_channels=self.arch['out_channels'][index],
                which_conv=self.which_conv,
                wide=self.D_wide,
                activation=self.activation,
                preactivation=True,
                downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] and d_index == 0 else None))
         for d_index in range(self.D_depth)]]
      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [biggan_layers.Attention(self.arch['out_channels'][index],
                                             self.which_conv)]
    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    # Linear output layer. The output dimension is typically 1, but may be
    # larger if we're e.g. turning this into a VAE with an inference output

    self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)

    self.discrim_prediction = self.which_linear(self.arch['out_channels'][-1], 38)

    # Embedding for projection discrimination
    self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

    # Initialize weights
    if not skip_init:
      self.init_weights()

    # Set up optimizer
    self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
    if D_mixed_precision:
      print('Using fp16 adam in D...')
      import utils
      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                              betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
      # LR scheduling, left here for forward compatibility
      # self.lr_sched = {'itr' : 0}# if self.progressive else {}
      # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
          or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for D''s initialized parameters: %d' % self.param_count)

  def forward(self, x, y=None):
    # Run input conv
    h = self.input_conv(x)
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)

    # Apply global sum pooling as in SN-GAN
    h = torch.sum(self.activation(h), [2, 3])

    # Get initial class-unconditional output
    out = self.linear(h)

    prediction = self.discrim_prediction(h)

    # Get projection of final featureset onto class vectors and add to evidence
    # out = out #+ torch.sum(self.linear_biomarkersvec(y) * h, 1, keepdim=True) #Srijay changes here
    return out, prediction


