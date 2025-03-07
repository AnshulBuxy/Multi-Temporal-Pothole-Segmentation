import os

import torch
import cv2
import numpy as np
from natsort import natsorted
import glob
from PIL import Image
from config import ParseGRU

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
    height, width, channels = ground_truth_rgb.shape
    ground_truth_classes = np.zeros((height, width), dtype=int)
    ground_truth_rgb=ground_truth_rgb.cpu().numpy()
    #for f in range(frames):
    for h in range(height):
      for w in range(width):
        rgb_value = tuple(ground_truth_rgb[h][w])
                #print(rgb_value)  # Get the RGB value
        ground_truth_classes[h][w] = RGB_TO_CLASS.get(rgb_value, -1)  # Default to -1 if not found

    return ground_truth_classes

# Convert class indices to one-hot encoding without batch size
def one_hot_encode(ground_truth_classes, num_classes):
    height, width = ground_truth_classes.shape
    one_hot = np.zeros((height, width, num_classes), dtype=np.float32)

    #for f in range(frames):
    for h in range(height):
      for w in range(width):
        one_hot[h, w, ground_truth_classes[ h, w]] = 1.0

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
parse  = ParseGRU()
opt    = parse.args
#folder='C:/Users/lenovo/Desktop/machine_learning/BTP/cracks-and-potholes-in-road/cracks-and-potholes-in-road/images'
#ame = natsorted(glob.glob(folder + '/*.' + "png"))
class Pre_dataset(torch.utils.data.Dataset):
    def __init__(self,
                 opt,
                 video_folder,
                 extension='jpg',
                 transforms=None):
        #video_dir = natsorted(os.listdir(video_folder))
        self.videos = []
        self.futures = []
        self.T = opt.T
        self.transforms = transforms
        #for i in range(len(video_dir)):
        frame_list = natsorted(glob.glob(video_folder + '/*.' + extension))
        folder_path=video_folder.split('/')[-1]
        label_list=[]
        if folder_path=="images":
          label_list = natsorted(glob.glob(video_folder.replace("images","masks") + '/*.' + "png"))
          #frame_list=frame_list[0:20]
          #label_list=label_list[0:20]
        elif folder_path=="image":
          label_list = natsorted(glob.glob(video_folder.replace("image","mask") + '/*.' + "png"))
        
        print(len(label_list))
        print(len(frame_list))
        for j in range(len(frame_list) - opt.T + 1):
            # print(frames[0])
            # print(frame_list[j + opt.T+1])
            video = [frame_list[j:j + opt.T][k] for k in range(opt.T)]
            #future = [frame_list[j + opt.T]]
            future = [label_list[j:j + opt.T][k] for k in range(opt.T)]

            # print(video)ramerame
            self.videos.append(video)
            self.futures.append(future)

        # print(len(self.videos))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_list = self.videos[idx]
        future_list = self.futures[idx]
        # print(future_list)

        # video = np.array([self.transforms(cv2.imread(video_list[i])).numpy() for i in range(len(video_list))])
        video = np.array([self.transforms(cv2.cvtColor(cv2.imread(video_list[i]), cv2.COLOR_BGR2RGB)).numpy() for i in
                          range(len(video_list))])
        #future = self.transforms(cv2.cvtColor(cv2.imread(future_list[0]), cv2.COLOR_BGR2RGB)).numpy()
        futures = np.empty((len(future_list),video.shape[1],video.shape[2],video.shape[3]),dtype=np.longlong)
        #print(futures.shape)
        #print(futures[0].shape)
        for i in range(len(future_list)):
            future = Image.open(future_list[i]) 
            
            future = future.resize((opt.image_size[1], opt.image_size[0]), resample=Image.Resampling.NEAREST)

            future = np.asarray(future, np.longlong)
            future = np.transpose(future, (2, 0, 1))
            future= torch.from_numpy(future)
            #print(future.shape)
            # break
            yper=future.permute(1,2,0)
         
          #shape=2,544,960,3
            ground_truth_classes = rgb_to_class_indices(yper)
            #print(ground_truth_classes)
            num_classes = len(RGB_TO_CLASS)
            labels = one_hot_encode(ground_truth_classes, num_classes)
    
            labels=torch.from_numpy(labels)
            
            y=labels.permute(2,0,1)
            
            futures[i] = y.numpy()

        # for i in range(len(video_list)):
        #     self.transforms(cv2.imread(video_list[i])).cat([video],dim=0)
        #
        # for j in range(len(future_video_list)):
        #     self.transforms(cv2.imread(future_video_list[j])).append(future_video)

        return torch.from_numpy(video), torch.from_numpy(futures)


#class VideoDataloader(torch.utils.data.Dataset):
#    def __init__(self,
#                 opt,
#                 video_folder,
#                 transforms=None):
#
#
#        video_dir = natsorted(os.listdir(video_folder))
#        self.videos = []
#        self.T = opt.T
#        self.transforms = transforms
#        for i in range(len(video_dir)):
#            frame_list = natsorted(glob.glob(video_folder + video_dir[i] + '/*.jpg'))
#
#            for j in range(len(frame_list)- opt.T*2 + 1):
#
#                #print(frames[0])
#                video = [frame_list[j:j+opt.T*2][k] for k in range (opt.T*2)]
#                #print(video)
#                self.videos.append(video)
#        #print(len(self.videos))
#
#
#
#    def __len__(self):
#        return len(self.videos)
#
#    def __getitem__(self, idx):
#        video_list = self.videos[idx][:self.T]
#        future_video_list = self.videos[idx][self.T:]
#
#        video = np.array([self.transforms(cv2.cvtColor(cv2.imread(video_list[i]), cv2.COLOR_BGR2RGB)).numpy() for i in range(len(video_list))])
#        future_video = np.array([self.transforms(cv2.cvtColor(cv2.imread(future_video_list[i]), cv2.COLOR_BGR2RGB)).numpy() for i in range(len(future_video_list))])
#
#        # for i in range(len(video_list)):
#        #     self.transforms(cv2.imread(video_list[i])).cat([video],dim=0)
#        #
#        # for j in range(len(future_video_list)):
#        #     self.transforms(cv2.imread(future_video_list[j])).append(future_video)
#
#        return torch.from_numpy(video),torch.from_numpy(future_video)





# Process video frames
        for i in range(len(video_list)):
            # Read and convert video frame to RGB
            video_frame = cv2.imread(video_list[i], cv2.IMREAD_UNCHANGED)
            if video_frame is not None:
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                video_frame = self.transforms(video_frame)  # Apply your transformations
                video_frames.append(video_frame.numpy())  # Append as numpy array
        
        video = np.array(video_frames)

# Process future frames
        for i in range(len(future_list)):
            future = cv2.imread(future_list[i], cv2.IMREAD_UNCHANGED)  # Load the image with PIL
            if future is not None:
                future = cv2.cvtColor(future, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                future = self.transforms(future)  # Apply your transformations
                futures.append(future.numpy())  # Append as numpy array
        label=np.array(futures)








  