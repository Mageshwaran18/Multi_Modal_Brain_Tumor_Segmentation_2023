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
import segmentation_models_3D as sm

# Ensure segmentation_models uses TensorFlow's Keras
sm.set_framework('tf.keras')

# Define loss and metrics
wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

# Define U-Net model
kernel_initializer = 'he_uniform'

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs

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
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
     
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
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
     
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    
    return model
# Streamlit app
st.title("3D U-Net Brain Tumor Segmentation")

@st.cache_resource
def load_trained_model():
    model = simple_unet_model(128, 128, 128, 3, 4)
    model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
    model.load_weights("D:/MMBTS_2020/Trained_Model/Trained_Model.h5")
    return model

model = load_trained_model()

def load_and_preprocess_image(uploaded_file, scaler):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as temp_file:
        temp_file.write(uploaded_file.read())  # Write the uploaded file to temp
        temp_file_path = temp_file.name  # Get the temporary file path
    
    # Load the file with nibabel
    img = nib.load(temp_file_path).get_fdata()
    
    # Remove the temporary file after loading
    os.remove(temp_file_path)
    
    # Preprocess the image
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)

    return img


def main():
    st.write("Upload FLAIR, T1CE, and T2 NIfTI files for segmentation")
    
    # File uploaders
    flair_file = st.file_uploader("Upload FLAIR image", type=['nii', 'nii.gz'])
    t1ce_file = st.file_uploader("Upload T1CE image", type=['nii', 'nii.gz'])
    t2_file = st.file_uploader("Upload T2 image", type=['nii', 'nii.gz'])
    
    if flair_file and t1ce_file and t2_file:
        if st.button("Process Images"):
            try:
                # Initialize scaler
                scaler = StandardScaler()
                
                # Process each modality directly from uploaded files
                with st.spinner('Processing FLAIR image...'):
                    processed_flair = load_and_preprocess_image(flair_file, scaler)
                with st.spinner('Processing T1CE image...'):
                    processed_t1ce = load_and_preprocess_image(t1ce_file, scaler)
                with st.spinner('Processing T2 image...'):
                    processed_t2 = load_and_preprocess_image(t2_file, scaler)
                
                # Stack the processed images
                stacked_image = np.stack([processed_flair, processed_t1ce, processed_t2], axis=3)
                stacked_image = stacked_image[56:184, 56:184, 13:141]
                
                # Add batch dimension
                stacked_image = np.expand_dims(stacked_image, axis=0)
                
                # Load the model
                with st.spinner('Loading model and making prediction...'):
                    prediction = model.predict(stacked_image)
                
                # Display results
                st.success("Segmentation completed!")
                st.image(prediction[0, :, :, 64], caption='Segmentation Result (Middle Slice)', clamp=True)
            
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
