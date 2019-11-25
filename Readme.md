# Semantic Image Segmentation

The project implements modified u-net Model that has no skip connections and uses Transposed Convolutional Blocks for upsampling instead of basic upsampling.
## Requirement
```
Python 3.7
Tensorflow 2.0
```
## Link to U-net Model
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## Training
The Model does binary segmentation i.e into object of interest and background
Preprocess your dataset into numpy arrays
```
Train_X  Nx224X224x3 
Train_Y  Nx224x224x1
```

where N is the number of images in your dataset. 
Loss function used is Binary Cross Entropy Function.

Pickle dump variable Train_X,Train_Y into DATA.TXT
Place DATA.TXT in Vessel Folder.

To train the model:
```
python Train.py
```

## Test
The Model is already trained on a dataset of Chemical Vessels
[Vessel Data Set](https://drive.google.com/file/d/0B6njwynsu2hXRFpmY1pOV1A4SFE/view)
The dataset is discussed in [Setting an attention region for convolutional neural networks using region selective features, for recognition of materials within glass vessels](https://arxiv.org/abs/1708.08711)

```
python Load_and_Test.py
```
