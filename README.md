# Multi-Modal Brain Tumor Segmentation (MMBTS_2020)

## 🧠 Problem Statement
The goal of this project is to segment brain tumors from MRI scans using deep learning. Each scan is multimodal, offering multiple views of the same brain region, each with unique imaging characteristics. The objective is to identify and classify different tumor regions accurately using a 3D U-Net architecture.

## 📊 Dataset Information
- **Source**: [BraTS2020 Training Data](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
- **Format**: NIfTI (.nii) files in 3D format
- **Multimodality**: 4 different views from the same region
- **Total Cases**: 369 (344 valid after preprocessing)

### Modalities Used
- **FLAIR**: Removes cerebrospinal fluid to make abnormalities in brain visible
- **T1CE**: Shows where the blood-brain barrier might be damaged (tumors/inflammation)
- **T2**: Highlights cerebrospinal fluid and identifies areas of abnormal fluid accumulations (edema)
- *Note: T1 modality was excluded as T1CE provides similar information with additional features*

### Annotations (Segmentation Labels)
- **Label 0**: Background (unlabeled volume)
- **Label 1**: Necrotic & non-enhancing tumor
- **Label 2**: Peritumoral edema
- **Label 4**: GD-enhancing tumor (aggressive part)
- *Note: Label 3 is not used in the dataset*

## 🔧 Data Preprocessing Pipeline

1. **NIfTI File Handling**:
   - Load 3D medical images (.nii format)
   - Extract metadata and image orientation information

2. **Normalization**:
   - Convert 3D images to 2D for normalization using MinMaxScaler
   - Restore normalized data back to 3D shape

3. **Modality Combination**:
   - Stack FLAIR, T1CE, and T2 modalities into a single numpy array

4. **Brain Region Extraction**:
   - Crop images to remove non-informative black areas
   - Keep only slices with at least 1% valid brain information

5. **Custom Data Generator**:
   - Created to handle numpy arrays for model training
   - Yields batches in format [(batch_size, height, width, channels), (batch_size, height, width, channels)]
   - Ensures aligned input images and segmentation masks

## 🧪 Model Architecture

### 3D U-Net Architecture
- **Encoder**: Down-sampling using convolutional layers to detect patterns
- **Decoder**: Up-sampling using transposed convolutional layers
- **Skip Connections**: Connect encoder layers to decoder layers to preserve spatial information
- **Activation**: ReLU to avoid vanishing gradient problem
- **Weight Initialization**: he_uniform for better performance with ReLU
- **Pooling**: MaxPooling3D to reduce spatial dimensions while preserving important features

### Loss Functions & Metrics
- **Categorical Focal Loss**: Focuses on hard-to-classify areas
- **Dice Loss**: Measures similarity between predicted segmentation and ground truth
- **IOU Score**: Measures overlap between predicted and actual regions
- **Optimizer**: Adam (combines momentum and RMSprop)

## 📈 Performance Metrics
- **Accuracy**: 98.50%
- **IOU Score**: 0.6946
- **Dice Loss**: 0.8059
- **Mean IOU**: 0.5183

## 🚀 Usage

### Web Application
The project includes a Streamlit web application for tumor segmentation:

```bash
streamlit run app.py
```

### Requirements
- TensorFlow
- Keras
- Nibabel
- Streamlit
- Scikit-learn
- Segmentation-models
- NumPy
- Matplotlib

### Model Files
- Trained model saved in `.h5` format at `D:\MMBTS_2020\Trained_Model\Trained_Model_With_Optimizer.h5`

## 👥 Contributors
- [Dharshini] (https://github.com/DharshiniRaji)
- [Mahima] (https://github.com/Caeruleaphile08)


