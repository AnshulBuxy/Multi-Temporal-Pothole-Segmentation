import argparse
import os
import sys
# sys.path.append('/content/drive/My Drive/Colab Notebooks/prediction/')
from natsort import natsorted
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from libtest import Pre_dataset
from  network import  Seq2seqGRU, SASTANGen
from config import ParseGRU
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode

import numpy as np
import argparse
import json
from PIL import Image
from os.path import join

import time

parse  = ParseGRU()
opt    = parse.args

save_path = "/home/jatinsahu/jatin/tempdata/btp_file/result_2_F_100_500img"

IGNORE_LABEL = 100
num_classes = 3
restore_from ="/home/jatinsahu/jatin/tempdata/btp_file/2_full_frame_MIOU:50.62-epoch:100.pth"

test_size = 500


RGB_TO_CLASS = {
    (0, 0, 0): 0,      # Background (Black)
    (255, 0, 0): 1,    # Class 1 (Red)
    (0, 255, 0): 2     # Class 2 (Green)
}
CLASS_TO_RGB = {
    0: (0, 0, 0),       # Background class (Black)
    1: (255, 0, 0),     # Class 1 (Red)
    2: (0, 255, 0)      # Class 2 (Green)
}
def rgb_to_class_indices(ground_truth_rgb):
    frames, height, width, channels = ground_truth_rgb.shape
    ground_truth_classes = np.zeros((frames, height, width), dtype=int)
    ground_truth_rgb=ground_truth_rgb.cpu().numpy()
    for f in range(frames):
        for h in range(height):
            for w in range(width):
                rgb_value = tuple(ground_truth_rgb[f][h][w])
                #print(rgb_value)  # Get the RGB value
                ground_truth_classes[f][h][w] = RGB_TO_CLASS.get(rgb_value, -1)  # Default to -1 if not found

    return ground_truth_classes

# Convert class indices to one-hot encoding without batch size
def one_hot_encode(ground_truth_classes, num_classes):
    frames, height, width = ground_truth_classes.shape
    one_hot = np.zeros((frames, height, width, num_classes), dtype=np.float32)

    for f in range(frames):
        for h in range(height):
            for w in range(width):
                one_hot[f, h, w, ground_truth_classes[f, h, w]] = 1.0

    return one_hot

def one_hot_to_rgb(one_hot_predictions):
    # Convert one-hot to class indices
    class_indices = np.asarray(np.argmax(one_hot_predictions.cpu().detach(), axis=3), dtype=np.uint8)
    #print(class_indices[0])
    frames, height, width = class_indices.shape
    rgb_image = np.zeros((frames, height, width, 3), dtype=np.uint8)

    for f in range(frames):
        for h in range(height):
            for w in range(width):
                class_idx = class_indices[f, h, w]
                rgb_image[f, h, w] = CLASS_TO_RGB.get(class_idx, (0, 0, 0))
    #print(f"rgb  {rgb_image[0]}")
    return rgb_image

transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Grayscale(1),
    transforms.Resize((opt.image_size[0], opt.image_size[1]),InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

palette = [
    0, 0, 0,      # Background (class 0 - black)
    255, 0, 0,    # Potholes (class 1 - red)
    0, 255, 0     # Cracks (class 2 - green)
]
zero_pad = 256 * 3 - len(palette)

for i in range(zero_pad):
    palette.append(0)
    
def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask
        
def main():
    """Create the model and start the evaluation process."""
    
    with torch.no_grad():

        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        autoencoder = SASTANGen(opt)
        saved_state_dict = torch.load(restore_from)
        ### for running different versions of pytorch
        model_dict = autoencoder.state_dict()
        saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
        #for k, v in saved_state_dict.items():
         #   saved_state_dict[k] = v.cpu()
        model_dict.update(saved_state_dict)
        ###
        autoencoder.load_state_dict(saved_state_dict)
    
        autoencoder.eval()
        autoencoder.cuda()
    
        datat_ = Pre_dataset(opt, opt.testset, extension=opt.img_extension, transforms=transform)  # b,label?
        test_loader = DataLoader(datat_, batch_size=opt.batch_size, shuffle=False)  # if shuffle
        for i, (data, labels) in enumerate(test_loader):
          print(f"Batch {i + 1}")
          print(f"Data shape: {data.shape}")
          print(f"Labels shape: {labels.shape}")
          if(i==1):
             break
        hist = np.zeros((num_classes, num_classes))
        print(f"hist={hist.shape}")
        t = []
        
        sp = torch.empty(test_size-opt.T+1, opt.T, opt.image_size[0], opt.image_size[1], opt.n_class)
        st = torch.empty(test_size-opt.T+1, opt.T, opt.image_size[0], opt.image_size[1])
        print(f"sp={sp.shape}")
        print(f"st={st.shape}")
        idx = 0
        m = nn.Softmax(dim=3)
    
        for data, ydata in test_loader:
    
            if data.size(0) != opt.batch_size:
                break
                
            start = time.time()
    
            x = data.reshape(-1, opt.T, opt.n_channels, opt.image_size[0], opt.image_size[1])
            y = ydata.reshape(-1,opt.n_channels,opt.image_size[0], opt.image_size[1])
            yper=y.permute(0,2,3,1)
           
            #shape=2,544,960,3
            ground_truth_classes = rgb_to_class_indices(yper)
            
            
            labels = one_hot_encode(ground_truth_classes, len(RGB_TO_CLASS))
    
            labels=torch.from_numpy(labels)
            
            y=labels.permute(0,3,1,2)
    
            if opt.cuda:
                x = Variable(x).cuda()
                y = Variable(y).cuda()
            else:
                x = Variable(x)
    
            yhat = autoencoder(x) # [1, T, C, H, W]
            
            yhat = yhat.reshape(-1, opt.n_class, opt.image_size[0], opt.image_size[1]) # [T, C, H, W]
            
            y=np.asarray(np.argmax(y.cpu().detach(), axis=1), dtype=np.uint8)    
            yhat = np.squeeze(yhat)
            yhat = yhat.permute(0,2,3,1) # [T, H, W, C]
            print(f"yhat={yhat.shape}")
            print(f"y={y.shape}")
            sp[idx] = m(yhat).cpu().detach()
            st[idx] = torch.tensor(y).detach()
            idx = idx + 1
            
            # yhat = np.asarray(np.argmax(yhat.cpu().detach(), axis=3), dtype=np.uint8)
            
            end = time.time()
            t.append(end - start)
            
            # y = np.asarray(y.cpu().detach(), dtype=np.uint8)
            
            #hist += fast_hist(y.flatten(), yhat.flatten(), num_classes)
        
        #print(sum(t)/len(t))
        sp1 = torch.zeros(test_size, opt.image_size[0], opt.image_size[1], opt.n_class)
        st1 = torch.ones(test_size, opt.image_size[0], opt.image_size[1])
        m = nn.Softmax(dim=2)
        for i in range(sp.shape[0]):
            for j in range(opt.T):
                sp1[i+j,:,:,:] = sp1[i+j,:,:,:] + sp[i,j,:,:,:]
                st1[i+j,:,:] = st[i,j,:,:]
                
        for i in range(sp1.shape[0]):
            yhat = np.asarray(np.argmax(sp1[i,:,:,:], axis=2), dtype=np.uint8)
            yhat_col = colorize_mask(yhat); #print(yhat_col.shape)
            yhat_ = Image.fromarray(yhat)
            name = str(i+1) + '.png'; #print(name)
            yhat_.save('%s/%s' % (save_path, name.replace('jpg','png')))
            yhat_col.save('%s/%s_color.png' % (save_path, name.split('.')[0]))
            y = np.asarray(st1[i,:,:], dtype=np.uint8)
            label = np.asarray(np.squeeze(y), dtype=np.uint8)        
            label_col = colorize_mask(label)
            label_col.save('%s/%s_gt_color.png' % (save_path, name.split('.')[0]))
            print(f"yhat={yhat.shape}")
            print(f"y={y.shape}")
            hist += fast_hist(y.flatten(), yhat.flatten(), num_classes)
        mIoUs = per_class_iu(hist)
        print(mIoUs)
        for ind_class in range(num_classes):
            print(str(round(mIoUs[ind_class] * 100, 2)))
        print(str(round(np.nanmean(mIoUs) * 100, 2)))
        return mIoUs
        
if __name__ == '__main__':
    main()