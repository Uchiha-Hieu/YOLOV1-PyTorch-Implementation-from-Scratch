import os
import numpy as np
import json
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

classes = {
    'triangle':0,
    'rectangle':1,
    'circle':2
}

class GeoShape(Dataset):
    def __init__(self,img_root,idx_root,json_path,mode="train",cell_size=7,num_classes=3):
        """
        img_root : directory contains images
        idx_root : directory contain train/val/test idxs (np.array .npy file)
        json_path : path to json labels
        mode : "train"/"val"/"test"
        cell_size : grid_size
        num_classes : num_classes to predict for each object detected
        """
        super(GeoShape,self).__init__()
        self.img_root = img_root

        if mode != "train" and mode != "val" and mode != "test":
            raise ValueError("Dataset mode shoulde be 'train' , 'test' or 'val'")
        
        self.idxs = np.load(os.path.join(idx_root,mode+".npy"))
        json_labels = json.load(open(json_path))
        self.labels = np.array(json_labels)[self.idxs.astype(int)]
        self.S = cell_size
        self.C = num_classes
    
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self,idx):
        img_name = str(self.idxs[idx])+".png"
        img = cv2.imread(os.path.join(self.img_root,img_name),cv2.IMREAD_GRAYSCALE)
        if (img.shape[0] != 224) and (img.shape[1] != 224):
            img = cv2.resize(img,(224,224))
        img = np.reshape(img,(img.shape[0],img.shape[1],1))
        img = np.array(img,dtype=np.float32)
        img = img/255.
        label = self.labels[idx]
        label_matrix = np.zeros((self.S,self.S,5+self.C))
        for box in label['boxes']:
            x1 = box['x1']
            y1 = box['y1']
            x2 = box['x2']
            y2 = box['y2']
            class_name = box['class']
            one_hot_list = [0]*len(classes)
            one_hot_list[classes[class_name]] = 1 
            x_center = (x1+x2)/2.
            y_center = (y1+y2)/2.
            w = x2-x1
            h = y2-y1
            x_idx,y_idx = int(x_center/img.shape[1]*self.S),int(y_center/img.shape[0]*self.S)
            label_matrix[y_idx,x_idx] = 1,float(x_center),float(y_center),float(w),float(h),*one_hot_list
            
        
        img = transforms.ToTensor()(img)
        return img,torch.FloatTensor(label_matrix)

# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#     img_root = "D:\\datasets\\geo_shapes\\train"
#     idx_root = "../train_val_test_idxs"
#     json_path = "D:\\datasets\\geo_shapes\\labels.json"
#     ds = GeoShape(img_root,idx_root,json_path,"train")
#     res = ds[0]
#     print(res[0].shape)
#     print(res[1].shape)
#     test_loader = DataLoader(ds,batch_size=64)
#     for data,target in test_loader:
#         print(data.shape)
#         print(target.shape)
#         break