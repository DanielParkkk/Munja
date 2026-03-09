import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from ema import EMA
from torch.utils.data import Dataset
import cv2
from utils.plots import colors

class mnistsimple_Dataset(Dataset):
    def __init__(self, image, detect_bbox_list, class_list, transforms=None):
        self.transforms = transforms
        self.data = []
        self.labels = []
        self.bbox_coord = []
        
        for i in range(len(detect_bbox_list)): # 이미지 크롭
            xmin = detect_bbox_list[i][0]
            ymin = detect_bbox_list[i][1]
            xmax = detect_bbox_list[i][2]
            ymax = detect_bbox_list[i][3]
            
            self.data.append(cv2.resize(image[ymin:ymax, xmin:xmax], dsize=(28, 28), interpolation=cv2.INTER_CUBIC))
            
            if len(class_list):
                self.labels.append(class_list[i])
            else:
                self.labels.append(-1)
                
            self.bbox_coord.append(detect_bbox_list[i][0:4])
                
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        if self.transforms:
            img = self.transforms(img)
        img = torch.tensor(np.array(img))
        label = torch.tensor(np.array(int(label))) 
        
        return img, label
    
    def __len__(self):
        return len(self.data)
    
class mnistsimple_Classifier_Model(nn.Module):
    def __init__(self, class_num):
        super(mnistsimple_Classifier_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 7, bias=False)    # output becomes 22x22
        self.conv1_bn = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, 7, bias=False)   # output becomes 16x16
        self.conv2_bn = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 144, 7, bias=False)  # output becomes 10x10
        self.conv3_bn = nn.BatchNorm2d(144)
        self.conv4 = nn.Conv2d(144, 192, 7, bias=False) # output becomes 4x4
        self.conv4_bn = nn.BatchNorm2d(192)
        self.fc1 = nn.Linear(3072, class_num, bias=False)
        self.fc1_bn = nn.BatchNorm1d(class_num)
    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        flat1 = torch.flatten(conv4.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat1))
        return logits
    def forward(self, x):
        logits = self.get_logits(x)
        return F.log_softmax(logits, dim=1)
    
def get_predictions(model, device, iterator, eval_flag = False):
    model.eval()
    ema = EMA(model, decay=0.999)
    ema.assign(model)

    tp_num = 0
    return_pred = []
    with torch.no_grad():

        if eval_flag:        
                    
            for data, target in iterator:
                data = data.to(device)
                output = model(data)
                
                pred = output.argmax(dim=1, keepdim=True)[0].item()
                
                return_pred.append(pred)        
                            
                if pred == target.item():
                    tp_num += 1
                    
            return tp_num, return_pred
        
        else:
            for data, target in iterator:
                data = data.to(device)
                output = model(data)
                
                pred = output.argmax(dim=1, keepdim=True)[0].item()
                return_pred.append(pred)        
            
            return return_pred

def get_f1_score(tp_num, gt_len, pred_len):
    
    if tp_num == 0:
        return 0, 0, 0
    
    p = tp_num / pred_len
    r = tp_num / gt_len
    
    f1 = 2 * p * r / (p + r)
    return p, r, f1

def get_results_image(annotator, t_data, ocrdataset, pred_class_list, tp_fp_list, save_img, save_crop):
    
    for i in range(len(pred_class_list)):
        if save_img or save_crop:  # Add bbox to image

            class_str = ''
            if pred_class_list[i] == -1:
                class_str = 'none'
            else:
                class_str = t_data[1][pred_class_list[i]]

            if len(tp_fp_list) == 0:
                annotator.box_label(ocrdataset.bbox_coord[i], class_str, color=colors(0, True))
            elif tp_fp_list[i] == 0: # tp : 0
                annotator.box_label(ocrdataset.bbox_coord[i], class_str, color=colors(0, True))
            
