from torch import nn
import numpy as np
from models.ZClassUtils import G_Block, ToRGB


class Generator(nn.Module):
    def __init__(self, latent_size, out_res, noise_dim=50):
        super().__init__()
        self.depth = 1
        self.alpha = 1
        self.fade_iters = 0
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.current_net = nn.ModuleList([G_Block(latent_size + noise_dim, latent_size, initial_block=True)])
        self.toRGBs = nn.ModuleList([ToRGB(latent_size, 3)])
        # __add_layers(out_res)
        for d in range(2, int(np.log2(out_res))):
            if d < 6:
                ## low res blocks 8x8, 16x16, 32x32 with 512 channels
                in_ch, out_ch = 512, 512
            else:
                ## from 64x64(5th block), the number of channels halved for each block
                in_ch, out_ch = int(512 / 2**(d - 6)), int(512 / 2**(d - 5))
            self.current_net.append(G_Block(in_ch, out_ch))
            self.toRGBs.append(ToRGB(out_ch, 3))


    def forward(self, x):
        for block in self.current_net[:self.depth-1]:
            x = block(x)
        out = self.current_net[self.depth-1](x)
        x_rgb = self.toRGBs[self.depth-1](out)
        if self.alpha < 1:
            x_old = self.upsample(x)
            old_rgb = self.toRGBs[self.depth-2](x_old)
            x_rgb = (1-self.alpha)* old_rgb + self.alpha * x_rgb

            self.alpha += self.fade_iters

        return x_rgb
        

    def growing_net(self, num_iters):
        
        self.fade_iters = 1/num_iters
        self.alpha = 1/num_iters

        self.depth += 1
