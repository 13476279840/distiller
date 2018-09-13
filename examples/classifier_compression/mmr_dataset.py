#coding:utf-8
from PIL import Image
import torch
# import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class mmr_dataset(Dataset):

    def __init__(self,root, datatxt, transform=None, target_transform=None):
        fh = open(root + datatxt, 'r') 
        imgs = []          
        for line in fh:            
            line = line.rstrip()       
            words = line.split('\t')  
            imgs.append((words[2],int(words[1]))) # words[2]是路径，words[1]是lable
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):    #这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]
        # img = Image.open(self.root+fn).convert('RGB')
        img = pil_loader(self.root+fn)
        if self.transform is not None:
            img = self.transform(img) #是否进行transform
        return img,label  #训练时循环读取每个batch时，需要获得的内容
    
    def __len__(self): #返回的是数据集的长度，必须实现
        return len(self.imgs)
    
# #根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
# root = '/opt/data/calib_mmr/'
# filename = 'test_calib_mmr.txt'
# train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
#                                     transforms.RandomHorizontalFlip(), 
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 

# train_data=MyDataset(root, filename, transform=train_transform)
# # test_data=MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())

# train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
# # test_loader = DataLoader(dataset=test_data, batch_size=64)