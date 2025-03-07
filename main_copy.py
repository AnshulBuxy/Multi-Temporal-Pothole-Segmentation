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
from lib import Pre_dataset
from  network import  Seq2seqGRU, SASTANGen
from config import ParseGRU
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
import torch.profiler
import time

parse  = ParseGRU()
opt    = parse.args

autoencoder = SASTANGen(opt)
autoencoder.train()
#mse_loss = nn.MSELoss()
#ce_loss = torch.nn.CrossEntropyLoss(ignore_index=100)
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(autoencoder.parameters(),
                             lr=opt.learning_rate,
                             weight_decay=1e-5)
transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Grayscale(1),
    transforms.Resize((opt.image_size[0], opt.image_size[1]),InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
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
# has all image shape,[data,label]
datat_ = Pre_dataset(opt, opt.trainset, extension=opt.img_extension, transforms=transform)  # b,label?
train_loader = DataLoader(datat_, batch_size=opt.batch_size, shuffle=True)  # if shuffle
# for i, (data, labels) in enumerate(train_loader):
#     ground_truth_rgb_reshaped = labels.permute(0, 1, 3, 4, 2)
#     print(ground_truth_rgb_reshaped.shape)
#     print(ground_truth_rgb_reshaped[0][0])
#     ground_truth_classes = rgb_to_class_indices(ground_truth_rgb_reshaped)
#     num_classes = len(RGB_TO_CLASS)
#     labels = one_hot_encode(ground_truth_classes, num_classes)
#     labels=torch.from_numpy(labels)
#     #labels=labels.permute(0,1,4,2,3)
#     print(labels.shape)
#     print(labels[0][0])
#     non_zero_indices = torch.nonzero(labels[0][0][0])
#     non_zero_values = labels[0][0][0][non_zero_indices[:, 0], non_zero_indices[:, 1]]
#     print(non_zero_values)
for i, (data, labels) in enumerate(train_loader):
    print(f"Batch {i + 1}")
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    if(i==1):
       break
# has all image shape,[data,label]
datatest_ = Pre_dataset(opt, opt.valset, extension=opt.img_extension, transforms=transform)  # b,label?,T
test_loader = DataLoader(datatest_, batch_size=1, shuffle=False)  # if shu
# for data, labels in train_loader:
#   print(labels.shape)
#   label=labels.permute(0, 1, 3, 4, 2)
#   print(f"after rezing{label.shape}")
#   print(labels[0][0][0])
#   print(label[0][0])
#   non_zero_indices = torch.nonzero(labels[0][0])
#   t=torch.nonzero(label[0][0])
#   print(f"normal{non_zero_indices}")
#   print(f"permute{non_zero_indices}")

# Step 2: Extract non-zero values using the indices
  # non_zero_values = labels[0][0][0][non_zero_indices[:, 0], non_zero_indices[:, 1]]
  # print(non_zero_values)
  # break;


# for i, (data, labels) in enumerate(test_loader):
#   y = labels.reshape(-1,opt.n_channels,opt.image_size[0], opt.image_size[1])

#   tests = y[:opt.n_test].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])                
#   os.makedirs(opt.log_folder, exist_ok=True)
#   for itr in range(opt.n_test):
#                     save_image((tests[itr] / 2 + 0.5), os.path.join(opt.log_folder, "real_itr{}_no{}.png".format(55, i)))
#                     print("image_saved")
if opt.cuda:
    autoencoder.cuda()

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    
losses = np.zeros(opt.num_epochs)
mIoU_best = 0
num_batches_for_estimate = 5  # Number of initial batches to time for estimation
total_batches = len(train_loader)
mIoU_best = 0  
    
for itr in range(opt.num_epochs):
    autoencoder.train()
    
    i=0
   
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/jatinsahu/jatin/tempdata/btp_file/log')
    ) as profiler:
      start_batch_time = time.time()
      for data, ydata in train_loader:
          
          if data.size(0) != opt.batch_size:
              break
  
          x = data.reshape(-1,opt.T, opt.n_channels, opt.image_size[0], opt.image_size[1])
          y = ydata.reshape(-1,opt.n_channels,opt.image_size[0], opt.image_size[1])
          #yper=y.permute(0,2,3,1)
         
          #shape=2,544,960,3
          #ground_truth_classes = rgb_to_class_indices(yper)
          #print(ground_truth_classes)
          #num_classes = len(RGB_TO_CLASS)
          #labels = one_hot_encode(ground_truth_classes, num_classes)
  
          #labels=torch.from_numpy(labels)
          
          #y=labels.permute(0,3,1,2)
          #print("complete preprocessing . . .")    
          if opt.cuda:
              x = Variable(x).cuda()
              y = Variable(y).cuda()
          else:
              x = Variable(x)
          #print(y.shape)
          #with torch.cuda.amp.autocast(): # for fater computation
          yhat = autoencoder(x)
          yhat = yhat.reshape(-1, opt.n_class, opt.image_size[0], opt.image_size[1])
          
          # ????(?????)????????loss???
          #loss = mse_loss(yhat, y)
          # print(f"yhat={yhat.shape}")
          # print(f"y={y.shape}")
          
          # print(f"labels={y.shape}")
          loss = ce_loss(yhat.float(), y.float())
          losses[itr] = losses[itr] * (itr / (itr + 1.)) + loss.data * (1. / (itr + 1.))
  
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          #batch_times.append(batch_time)

        
            # if i==4:
          #    ans=yhat.permute(0,2,3,1)
          #    print(f"yper={yper}")
          #    y=y.permute(0,2,3,1)
          #    print(f"y={y}")
          #    print(f"grouth={ground_truth_classes}")
          #    print(f"labels={labels}")
          #    print(f"yhat={ans}")
          #    break
          
          # i=i+1
      batch_time = time.time() - start_batch_time   
      print('epoch [{}/{}], loss: {:.4f}'.format(itr + 1, opt.num_epochs, losses[itr]))
      print(f"time-{batch_time}")
    #if itr==25 or itr==50 or itr ==1 or itr==100: 
      #torch.save(autoencoder.state_dict(), os.path.join("/home/jatinsahu/jatin/tempdata/btp_file", '4frame_MIOU{:04f}{:04d}.pth'.format(mIoU_best,itr)))
      #with open(f'/home/jatinsahu/jatin/tempdata/btp_file/losses3{itr}.csv', 'w', newline='') as file:
        #writer = csv.writer(file)
        #writer.writerow(losses)
    hist = np.zeros((opt.n_class, opt.n_class))
    if (itr) % opt.check_point == 0:
        autoencoder.eval()
        with torch.no_grad(): 
          for data, ydata in test_loader:

              if data.size(0) != opt.batch_size:
                  break

              x = data.reshape(-1, opt.T, opt.n_channels, opt.image_size[0], opt.image_size[1])
              y = ydata.reshape(-1,opt.n_channels, opt.image_size[0], opt.image_size[1])

              if opt.cuda:
                  x = Variable(x).cuda()
                  y = Variable(y).cuda()
              else:
                  x = Variable(x)

              yhat = autoencoder(x)
              yhat = yhat.reshape(-1, opt.n_class, opt.image_size[0], opt.image_size[1])
              
              yhat = yhat.permute(0,2,3,1)
              label=one_hot_to_rgb(yhat)
              label=torch.from_numpy(label)
              label=label.permute(0,3,1,2)
              tests = y[:opt.n_test].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])
              recon = label[:opt.n_test].reshape(-1, opt.n_channels, opt.image_size[0], opt.image_size[1])

                
              os.makedirs(opt.log_folder, exist_ok=True)
              for i in range(1):
                    save_image((tests[-1]/2+0.5),
                               os.path.join(opt.log_folder, "real_itr{}_no{}.png".format(itr, i)))
                    save_image((recon[-1]/2+0.5),
                               os.path.join(opt.log_folder, "recon_itr{}_no{}.png".format(itr, i)))
                    #print((ce_loss(tests, recon)))
         
              
              
              
#               #print(yhat.shape)
#               #yhat = yhat.reshape(-1, opt.n_class, opt.image_size[0], opt.image_size[1])
              
              
#               yhat = np.squeeze(yhat)
              #yhat = yhat.permute(0,2,3,1)
                      
              yhat = np.asarray(np.argmax(yhat.cpu().detach(), axis=3), dtype=np.uint8)
              #yhat = np.asarray(np.argmax(yhat.cpu().detach(), axis=2), dtype=np.uint8) 
              #print(yhat.shape)
              #print(y.shape)
              y=np.asarray(np.argmax(y.cpu().detach(), axis=1), dtype=np.uint8)
              #print(yhat.shape)
              #print(y.shape)
              # print(len(y.flatten()))
              # print(len(yhat.flatten()))
              
              hist += fast_hist(y.flatten(), yhat.flatten(), opt.n_class)
          print("image_saved")
          mIoUs = per_class_iu(hist)
          mIoU = round(np.nanmean(mIoUs) * 100, 2)
          print('===> mIoU: ' + str(mIoU))
          if mIoU > mIoU_best:
              mIoU_best = mIoU
              print('===> Best mIoU: ' + str(mIoU_best))
    filename = '1_full_frame_MIOU:{:.2f}-epoch:{}.pth'.format(mIoU, itr)
    filepath = os.path.join("/home/jatinsahu/jatin/tempdata/btp_file", filename)
    if itr==25 or itr==50 or itr ==1 or itr==100: 
      torch.save(autoencoder.state_dict(), filepath)     
              
          
    

