import os
from readerDS import get_batch, FaceIdExpDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from typing import List, Optional, Sequence, Union, Any, Callable

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 16 if torch.cuda.is_available() else 16
NUM_WORKERS = int(os.cpu_count() / 2)


    

class GANDataModule( LightningDataModule):

    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dims = (3, 96, 96)
        self.num_classes = 178
        self.pin_memory = 1

    def setup(self, stage: Optional[str] = None):
        super().setup()

        a = False
        if os.name == 'nt':
            self.rootV = 'C:/Temp/CAS-PEAL.VDGAN/'
            self.rootG = 'C:/Temp/CAS-PEAL.VDGAN/'
            self.train_path = self.rootV + 'LoadPEAL200.txt'
            self.valid_path = self.rootV + 'LoadPEAL200.txt'
            self.test_path = 'C:/Temp/CAS-PEAL.VDGAN/LoadPEAL100.txt'
        else:
            self.rootV = '/media/lightdi/CRUCIAL/Datasets/AR-Cropped/'
            self.rootG = '/media/lightdi/CRUCIAL/Datasets/AR-Cropped/'
            self.train_path = 'dataset_file/Load_AR_training_50_4.txt'
            self.valid_path = 'dataset_file/Load_AR_test_50_4.txt'
            self.test_path = 'dataset_file/Load_AR_test_50_4.txt'

        self.train_dataset =  FaceIdExpDataset(self.rootV, self.train_path,
                                 transform=transforms.Compose([
                                     transforms.Resize((64,64)),
#                                     transforms.CenterCrop((96,96)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                 ]))
        

        self.val_dataset =  FaceIdExpDataset(self.rootV, self.valid_path,
                                 transform=transforms.Compose([
                                     #transforms.Resize((64,64)),
#                                     transforms.CenterCrop((96,96)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                 ]))
        

        self.test_dataset =  FaceIdExpDataset(self.rootG, self.test_path,
                                 transform=transforms.Compose([
                                     #transforms.Resize((64,64)),
#                                     transforms.CenterCrop((96,96)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                 ]))
        a = True



    def train_dataloader(self) -> DataLoader:

        return DataLoader(self.train_dataset,batch_size=self.batch_size,
                            shuffle=True,drop_last=True, num_workers=6)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        
        return DataLoader(self.val_dataset,batch_size=self.batch_size,
                            shuffle=True,drop_last=True, num_workers=6)
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        
        return DataLoader(self.test_dataset,batch_size=self.batch_size,
                            shuffle=False,drop_last=False, num_workers=6)
