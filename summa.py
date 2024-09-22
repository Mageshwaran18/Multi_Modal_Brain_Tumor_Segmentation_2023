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
        st.write("Input Images")

        # Display each uploaded NIfTI file horizontally
        slice_num = st.slider("Select Slice Number", 0, 100, 50)  # Slider to choose which slice to display
        columns = st.columns(4)  # Create 4 columns for horizontal layout

        for i, file in enumerate(uploaded_files):
            img_slice = load_nii_slice(file, slice_num=slice_num)
            fig, ax = plt.subplots()
            ax.imshow(img_slice.T, cmap="gray", origin="lower")
            ax.axis('off')  # Turn off axis to make it cleaner
            columns[i].pyplot(fig)  # Display each image in its own column

        # Display the preloaded segmented file
        st.header("Segmented Image")
        segmented_path = segmented_path = r"D:/MMBTS_2020/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii"
        
        if segmented_path:
            segmented_slice = load_nii_slice(open(segmented_path, 'rb'), slice_num=slice_num)
            fig, ax = plt.subplots(figsize=(4, 4))  # Reduced size of the figure for segmented image
            ax.imshow(segmented_slice.T, cmap="gray", origin="lower")
            ax.axis('off')  # Hide the axis
            st.pyplot(fig)

    elif len(uploaded_files) != 4:
        st.warning("Please upload exactly 4 NIfTI files.")

if __name__ == "__main__":
    main()
