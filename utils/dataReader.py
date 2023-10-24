# -*- coding: utf-8 -*-
import os
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label, disguise, proPath = line.strip().split(' ')
            if int(disguise) != 0:
                disguise = 1 
            imgList.append((imgPath, int(label), int(disguise), proPath))
    return imgList

class FaceIdExpDataset(Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_list_reader):
        self.root      = root
        self.imgList   = list_reader(fileList)
        self.transform = transform
        
    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        imgPath, id_label, disguise_label, proPath = self.imgList[idx]
        img = Image.open(os.path.join(self.root, imgPath)+'.bmp')
        img = img.convert('L').convert('RGB')
        img= self.transform(img)
        pro = Image.open(os.path.join(self.root, proPath)+'.bmp')
        pro = pro.convert('L').convert('RGB')
        pro= self.transform(pro)
        return [img.float(), id_label, disguise_label, pro.float()]
#        return [os.path.join(self.root, imgPath), id_label, exp_label, pro.float()]



def get_batch(root, fileList, batch_size, shuffle=True, drop_last= True):
    data_set = FaceIdExpDataset(root, fileList,
                                 transform=transforms.Compose([
#                                     transforms.Resize((110,110)),
                                     transforms.CenterCrop((96,96)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                 ]))
    dataloader = DataLoader(data_set,batch_size=batch_size,
                            shuffle=shuffle,drop_last=drop_last)  #drop_last is necessary,because last iteration may fail 
    return dataloader


def test_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label, disguise, proPath = line.strip().split(' ')
            if int(disguise) != 0:
                disguise = 1 
            else:
                continue
            imgList.append((imgPath, int(label), int(disguise), proPath))
    return imgList

class FaceIdTestDataset(Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=test_list_reader):
        self.root      = root
        self.imgList   = list_reader(fileList)
        self.transform = transform
        
    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        imgPath, id_label, disguise_label, proPath = self.imgList[idx]
        img = Image.open(os.path.join(self.root, imgPath)+'.bmp')
        img = img.convert('L').convert('RGB')
        img= self.transform(img)
        pro = Image.open(os.path.join(self.root, proPath)+'.bmp')
        pro = pro.convert('L').convert('RGB')
        pro= self.transform(pro)
        return [img.float(), id_label, disguise_label, pro.float()]
#        return [os.path.join(self.root, imgPath), id_label, exp_label, pro.float()]


def get_test_batch(root, fileList, batch_size, shuffle=True, drop_last= True):
    data_set = FaceIdTestDataset(root, fileList,
                                 transform=transforms.Compose([
                                     transforms.Resize((64,64)),
                                     #transforms.CenterCrop((96,96)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                 ]))
    dataloader = DataLoader(data_set,batch_size=batch_size,
                            shuffle=shuffle,drop_last=drop_last)  #drop_last is necessary,because last iteration may fail 
    return dataloader




#a = test_list_reader('/media/lightdi/CRUCIAL/Datasets/CAS-PEAL.VDGAN/LoadPEAL100.txt')
#b = default_list_reader('/media/lightdi/CRUCIAL/Datasets/CAS-PEAL.VDGAN/LoadPEAL100.txt')
#print(a,'\n,',b)