import os
import yaml
import argparse
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from experiment import PROGADVAE
from dataset import GANDataModule

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


if __name__ == '__main__':  

    tb_logger =  TensorBoardLogger(save_dir='log_dir/',
                                name='GAN',)

    max_epochs = 5000
    model = PROGADVAE(3, 96, 96, 320, 64,300 ,max_epochs=max_epochs)

    data = GANDataModule('')
    data.setup()
    runner = Trainer(accelerator='gpu', devices=1, logger=tb_logger,
                    callbacks = [
                                LearningRateMonitor(),
                                ModelCheckpoint(save_top_k=10,
                                                dirpath='checkpoints/',
                                                monitor='val_loss',
                                                save_last=True),
                                ],
                    strategy=DDPPlugin(find_unused_parameters=True),
                    max_epochs=max_epochs)


    print(f"======= Training GAN =======")
    runner.fit(model, datamodule=data)
   