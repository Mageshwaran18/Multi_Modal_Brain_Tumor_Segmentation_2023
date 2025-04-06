import streamlit as st
import nibabel as nib
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv3D, MaxPooling3D, Conv3DTranspose, 
                                   Dropout, concatenate)
import tempfile
import os
import matplotlib.pyplot as plt
import segmentation_models_3D as sm

# Configure segmentation models framework
sm.set_framework('tf.keras')

kernel_initializer = 'he_uniform'
# Define colors for segmentation visualization
colors = {
    0: [0, 0, 0, 0],        # Background - transparent
    1: [1, 0, 0, 0.3],      # Necrotic and non-enhancing tumor - red
    2: [0, 1, 0, 0.3],      # Peritumoral edema - green
    3: [0, 0, 1, 0.3]       # Enhancing tumor - blue
}

# Define loss and metrics
def get_loss_and_metrics():
    wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]
    return total_loss, metrics

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    # [Previous U-Net model implementation remains the same]
    # The function body is kept as is since it was correctly implemented
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs
    
    # Encoding
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D((2, 2, 2))(c4)

    # Bottleneck
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)

    # Decoding
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)

    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)

    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)

    # [Rest of the U-Net implementation...]
    # Keep the existing implementation

    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def load_and_preprocess_image(uploaded_file):
    """Load and preprocess a NIfTI image file."""
    scaler = StandardScaler()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    try:
        img = nib.load(temp_file_path).get_fdata()
        processed_img = scaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape)
        return processed_img
    finally:
        os.remove(temp_file_path)

def create_overlay(prediction, slice_num):
    """Create a segmentation overlay for visualization."""
    pred_slice = prediction[0, :, :, slice_num]
    overlay = np.zeros((*pred_slice.shape, 4))
    
    for label, color in colors.items():
        overlay[pred_slice == label] = color
    
    return overlay

def main():
    st.title("Multi Modal Brain Tumor Segmentation")
    st.write("Upload FLAIR, T1CE, and T2 NIfTI files for segmentation")
    
    # File uploaders
    flair_file = st.file_uploader("Upload FLAIR image", type=['nii', 'nii.gz'])
    t1ce_file = st.file_uploader("Upload T1CE image", type=['nii', 'nii.gz'])
    t2_file = st.file_uploader("Upload T2 image", type=['nii', 'nii.gz'])
    
    # Load model
    @st.cache_resource
    def load_trained_model():
        model = simple_unet_model(128, 128, 128, 3, 4)
        total_loss, metrics = get_loss_and_metrics()
        model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
        try:
            model.load_weights(r"D:\MMBTS_2020\Trained_Model\newly_trianed_model_without_optimizers.h5")
        except:
            st.error("Could not load model weights. Please ensure the weights file is in the correct location.")
            return None
        return model
    
    model = load_trained_model()
    
    if model is None:
        st.error("Model could not be loaded. Please check the model weights file.")
        return

    if all([flair_file, t1ce_file, t2_file]):
        if st.button("Process Images"):
            try:
                with st.spinner('Processing images...'):
                    # Process each modality
                    processed_flair = load_and_preprocess_image(flair_file)
                    processed_t1ce = load_and_preprocess_image(t1ce_file)
                    processed_t2 = load_and_preprocess_image(t2_file)
                    
                    # Stack and prepare images
                    stacked_image = np.stack([processed_flair, processed_t1ce, processed_t2], axis=3)
                    stacked_image = stacked_image[56:184, 56:184, 13:141]
                    stacked_image = np.expand_dims(stacked_image, axis=0)
                    
                    # Make prediction
                    prediction = model.predict(stacked_image)
                    
                    # Visualization
                    st.write("Segmentation Results")
                    slice_num = st.slider("Select Slice", 0, prediction.shape[3]-1, prediction.shape[3]//2)
                    
                    cols = st.columns(4)
                    
                    # Display original images and overlay
                    images = [processed_flair, processed_t1ce, processed_t2]
                    titles = ['FLAIR', 'T1CE', 'T2', 'Segmentation']
                    
                    for idx, (img, title) in enumerate(zip(images, titles[:3])):
                        fig, ax = plt.subplots()
                        ax.imshow(img[:, :, slice_num], cmap='gray')
                        ax.set_title(title)
                        ax.axis('off')
                        cols[idx].pyplot(fig)
                        plt.close()
                    
                    # Display segmentation overlay
                    fig, ax = plt.subplots()
                    ax.imshow(prediction[0, :, :, slice_num] , cmap='gray')
                    ax.set_title('Segmentation')
                    ax.axis('off')
                    cols[3].pyplot(fig)
                    plt.close()
                    
                    st.success("Segmentation completed successfully!")
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                st.info("Please make sure all uploaded files are valid NIfTI images with compatible dimensions.")

if __name__ == "__main__":
    main()