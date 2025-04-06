import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from keras.metrics import MeanIoU

# Load the U-Net model
def simple_unet_model():
    # Define your U-Net model structure here
    # For now, we're assuming the model is pre-saved and loaded directly
    model = load_model(r"D:\MMBTS_2020\Trained_Model\newly_trianed_model_without_optimizers.h5", compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = simple_unet_model()

# Function to preprocess images (for .nii format)
def load_and_preprocess_image(filepath):
    img = nib.load(filepath).get_fdata()
    scaler = StandardScaler()
    img_scaled = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    stacked_image = np.stack([img_scaled] * 3, axis=-1)  # Create a 3D channel dimension
    return stacked_image

# Function to calculate IoU
def calculate_iou(predictions, ground_truth, num_classes=4):
    iou = MeanIoU(num_classes=num_classes)
    pred_argmax = np.argmax(predictions, axis=4)
    gt_argmax = np.argmax(ground_truth, axis=4)
    iou.update_state(pred_argmax, gt_argmax)
    return iou.result().numpy()

# Streamlit app
def main():
    st.title("U-Net Segmentation Viewer")

    # File upload
    uploaded_file = st.file_uploader("Upload a .nii file for segmentation", type=["nii"])
    if uploaded_file is not None:
        with st.spinner("Loading and preprocessing image..."):
            stacked_image = load_and_preprocess_image(uploaded_file)
            st.success("Image loaded and preprocessed successfully.")

        # Display some slices from the uploaded file
        slice_idx = st.slider("Select a slice index", 0, stacked_image.shape[2] - 1, 0)
        plt.figure(figsize=(5, 5))
        plt.imshow(stacked_image[:, :, slice_idx, 0], cmap="gray")
        plt.title(f"Slice {slice_idx}")
        plt.axis("off")
        st.pyplot(plt)

        # Prediction
        with st.spinner("Performing segmentation..."):
            stacked_image_expanded = np.expand_dims(stacked_image, axis=0)  # Add batch dimension
            prediction = model.predict(stacked_image_expanded)
            st.success("Segmentation completed.")

        # IoU calculation (if ground truth is available)
        if st.checkbox("Calculate IoU (requires ground truth)"):
            uploaded_mask = st.file_uploader("Upload the ground truth mask", type=["nii"])
            if uploaded_mask is not None:
                ground_truth = load_and_preprocess_image(uploaded_mask)
                iou_score = calculate_iou(prediction, np.expand_dims(ground_truth, axis=0))
                st.write(f"Mean IoU: {iou_score:.4f}")

        # Visualization of segmentation
        st.header("Segmentation Results")
        predicted_slice = np.argmax(prediction[0], axis=-1)[:, :, slice_idx]

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(stacked_image[:, :, slice_idx, 0], cmap="gray")
        ax[0].set_title(f"Original Slice {slice_idx}")
        ax[0].axis("off")

        ax[1].imshow(predicted_slice, cmap="viridis")
        ax[1].set_title(f"Predicted Segmentation Slice {slice_idx}")
        ax[1].axis("off")

        st.pyplot(fig)

if __name__ == "__main__":
    main()
