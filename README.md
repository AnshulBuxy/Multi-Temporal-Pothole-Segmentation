# Deep Learning-Based Spatiotemporal Segmentation of Road Damage

This project leverages the power of deep learning to automatically segment road damages—specifically potholes and cracks—by combining spatial and temporal information. By integrating a convolutional encoder, ConvLSTM layers, and a convolutional decoder, the model captures both the texture details and the evolution of damage over consecutive frames.

---

## 1. About the Project

The objective of this project is to develop an automated, robust system for road damage detection. Traditional segmentation models that operate on single images can miss the temporal evolution of damage. This project instead utilizes a ConvLSTM-based architecture to:
- **Extract spatial features:** Use a CNN encoder to capture details of road surfaces.
- **Learn temporal dependencies:** Employ ConvLSTM layers to understand the evolution of damage over multiple frames.
- **Generate accurate segmentations:** Use a CNN decoder to output precise segmentation maps classifying each pixel as a crack, pothole, or normal road surface.

---

## 2. Problem Statement

Manual inspection of roads for damages is time-consuming, expensive, and prone to errors. Moreover, conventional image segmentation models are limited by their single-frame approach, which does not consider temporal changes. This project addresses these challenges by:
- **Reducing manual effort:** Automating the detection process.
- **Increasing accuracy:** Capturing temporal correlations to better recognize evolving damages.
- **Handling real-world variations:** Improving robustness against issues like motion blur and occlusions.

---

## 3. Advantages Over Other Segmentation Architectures

The proposed ConvLSTM architecture outperforms traditional segmentation methods in several key areas:

- **Spatiotemporal Learning:** Unlike single-frame CNNs, the model integrates temporal information from multiple frames, leading to a better understanding of damage progression.
- **Enhanced Robustness:** The temporal fusion makes the model less sensitive to issues like motion blur and changing perspectives.
- **Improved Accuracy:** By leveraging the temporal context, the model achieves higher Intersection over Union (IoU) scores, resulting in more reliable segmentation.

> ![Comparison Diagram](./images/comparison_architecture.png)  
> *Figure: A schematic comparison between traditional CNN segmentation and the ConvLSTM-based approach.*

---

## 4. Model Architecture

The model is designed with three main components:

1. **CNN Encoder:**  
   Extracts spatial features from each input frame using several convolutional layers. This helps in identifying textures and edges that are characteristic of road damages.

2. **ConvLSTM Layers:**  
   These layers capture the temporal dependencies between consecutive frames, enabling the model to learn the evolution and progression of road damage over time.

3. **CNN Decoder:**  
   Converts the fused spatiotemporal features back into segmentation maps. It classifies each pixel into one of the target categories (crack, pothole, or normal surface).

> ![Model Architecture](./images/model_architecture.png)  
> *Figure: Diagram of the ConvLSTM-based spatiotemporal segmentation model.*

---

## 5. Results

The model was trained and evaluated on a dataset of 2000 road image sequences (with 1750 images for training and 250 images for testing), where each image is of size 512 x 320 pixels. The performance of the model was measured using the Intersection over Union (IoU) metric.

- **Multi-frame Sequences:** Using 2-frame and 3-frame sequences significantly improved segmentation accuracy compared to single-frame inputs.
- **Consistent Predictions:** The model produces smoother and more consistent segmentation maps, effectively highlighting the regions affected by cracks and potholes.
> ![Results Comparison](https://github.com/AnshulBuxy/Multi-Temporal-Pothole-Segmentation/blob/main/Screenshot%202025-03-07%20104408.png)
> *Figure: Visual comparison of segmentation outputs from 1-frame, 2-frame, and 3-frame sequences.*
> | ![Image 1](https://github.com/AnshulBuxy/Multi-Temporal-Pothole-Segmentation/blob/main/Picture1.png) | ![Image 2](https://github.com/AnshulBuxy/Multi-Temporal-Pothole-Segmentation/blob/main/Picture1.png) |

---

## 6. Files Information

Below is a brief description of the files included in this project:

- **main_copy.py:**  
  The main Python script to run the model.

- **network.py:**  
  Contains the definition of the model architecture including the encoder, ConvLSTM layers, and decoder.

- **lib.py:**  
  Handles the preprocessing of the training and validation data.

- **config.py:**  
  A configuration file that defines the hyperparameters used in the project.

- **libtest.py:**  
  Responsible for preprocessing the test data.

- **evaluate.py:**  
  Contains the evaluation routines to assess the performance of the trained model.

---

## Acknowledgments

This project is developed under the supervision of Dr. Tarutal Ghosh Mondal. Special thanks to the Department of Civil Engineering, School of Infrastructure, IIT Bhubaneswar for their guidance and support.

---

*For more detailed information, please refer to the project presentation provided.*
