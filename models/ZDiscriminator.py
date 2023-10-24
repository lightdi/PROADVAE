from torch import nn
import numpy as np
from models.ZClassUtils import D_Block, FromRGB

class Discriminator(nn.Module):
    def __init__(self, latent_size, out_res, Nd):
        super().__init__()
        self.depth = 1
        self.alpha = 1
        self.fade_iters = 0
        flatten_result = 1
        
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.t = 1

        self.reshape = nn.Flatten()

        self.project = nn.Linear(flatten_result, Nd+3)


        self.current_net = nn.ModuleList([D_Block(latent_size, latent_size, initial_block=True)])
        self.fromRGBs = nn.ModuleList([FromRGB(3, latent_size)])
        for d in range(2, int(np.log2(out_res))):
            if d < 6:
                in_ch, out_ch = 512, 512
            else:
                in_ch, out_ch = int(512 / 2**(d - 5)), int(512 / 2**(d - 6))
            self.current_net.append(D_Block(in_ch, out_ch))
            self.fromRGBs.append(FromRGB(3, in_ch))


        
    
    def forward(self, x_rgb):
        x = self.fromRGBs[self.depth-1](x_rgb)

        x = self.current_net[self.depth-1](x)
        if self.alpha < 1:

            x_rgb = self.downsample(x_rgb)
            x_old = self.fromRGBs[self.depth-2](x_rgb)
            x = (1-self.alpha)* x_old + self.alpha * x
            self.alpha += self.fade_iters
        for block in reversed(self.current_net[:self.depth-1]):
            x = block(x)

        x = self.reshape(x)
        #print(x.shape)

        x = self.project(x)
      
        
        return x
        
    def growing_net(self, num_iters):

        self.fade_iters = 1/num_iters
        self.alpha = 1/num_iters

        self.depth += 1
