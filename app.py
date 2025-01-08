import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Function to load and extract a slice from a NIfTI file
def load_nii_slice(file, slice_num=50):
    # Load NIfTI from UploadedFile
    file_bytes = file.read()  # Read file into bytes
    nii_img = nib.Nifti1Image.from_bytes(file_bytes)  # Load NIfTI file from bytes
    data = nii_img.get_fdata()
    # Assuming we're dealing with 3D volumes (x, y, z)
    slice_2d = data[:, :, slice_num]
    return slice_2d

# Streamlit app
def main():
    st.title("Multi Modal Brain Tumor Segmentation")

    # Upload multiple NIfTI files
    st.header("Upload the required NIfTI Files")
    uploaded_files = st.file_uploader("Choose four .nii files", type="nii", accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) == 4:
        st.write("Input Images with Segmentation Overlay")

        slice_num = st.slider("Select Slice Number", 0, 100, 50)  # Slider to choose which slice to display
        columns = st.columns(4)  # Create 4 columns for horizontal layout

        # Load segmentation first
        segmented_path = r"D:/MMBTS_2020/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii"
        segmented_slice = load_nii_slice(open(segmented_path, 'rb'), slice_num=slice_num)

        # Create segmentation mask with colors
        colors = {
            0: [0, 0, 0, 0],      # background (transparent)
            1: [1, 0, 0, 0.5],    # red with 50% transparency
            2: [0, 1, 0, 0.5],    # green with 50% transparency
            4: [0, 0, 1, 0.5]     # blue with 50% transparency
        }

        # Create RGBA segmentation overlay
        seg_overlay = np.zeros((*segmented_slice.T.shape, 4))
        for label, color in colors.items():
            seg_overlay[segmented_slice.T == label] = color

        for i, file in enumerate(uploaded_files):
            img_slice = load_nii_slice(file, slice_num=slice_num)
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Display the original brain image
            ax.imshow(img_slice.T, cmap="gray", origin="lower")
            
            # Overlay the segmentation
            ax.imshow(seg_overlay, origin="lower")
            
            ax.axis('off')  # Turn off axis to make it cleaner
            columns[i].pyplot(fig)  # Display each image in its own column

    elif len(uploaded_files) != 4:
        st.warning("Please upload exactly 4 NIfTI files.")

if __name__ == "__main__":
    main()