import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf
import segmentation_models as sm

# ─────────────────────────────────────────────────────────────
# 🔧 CONFIGURATION
st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
scaler = MinMaxScaler()
sm.set_framework('tf.keras')

# Register custom losses and metrics
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + focal_loss
get_custom_objects().update({
    'categorical_focal_loss': focal_loss,
    'dice_loss': dice_loss,
    'total_loss': total_loss,
    'iou_score': sm.metrics.IOUScore(threshold=0.5),
})

# ─────────────────────────────────────────────────────────────
# 🧠 TITLE
st.title("🧠 Brain Tumor Segmentation from NIfTI MRI Scans")
st.markdown("Upload **three NIfTI files** (`FLAIR`, `T1ce`, `T2`) of the same shape for segmentation using a pre-trained 3D U-Net model.")

# ─────────────────────────────────────────────────────────────
# 📤 FILE UPLOAD
flair_file = st.file_uploader("Upload FLAIR image (.nii or .nii.gz)", type=["nii", "nii.gz"])
t1ce_file = st.file_uploader("Upload T1ce image (.nii or .nii.gz)", type=["nii", "nii.gz"])
t2_file   = st.file_uploader("Upload T2 image (.nii or .nii.gz)", type=["nii", "nii.gz"])

# ─────────────────────────────────────────────────────────────
# 📦 MODEL LOADING
@st.cache_resource
def load_trained_model(path):
    return load_model(path, compile=False)

model_path = r"D:\MMBTS_2020\Trained_Model\Trained_Model_With_Optimizer.h5"  # Update if needed
model = load_trained_model(model_path)

# ─────────────────────────────────────────────────────────────
# 🧪 UTILS

def load_and_normalize_nifti(nifti_file):
    nifti_img = nib.load(BytesIO(nifti_file.read()))
    data = nifti_img.get_fdata()
    data = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
    return data

def preprocess_inputs(flair, t1ce, t2):
    assert flair.shape == t1ce.shape == t2.shape, "All modalities must have the same shape"
    stacked = np.stack([flair, t1ce, t2], axis=-1)  # shape: (128, 128, 128, 3)
    return stacked

# ─────────────────────────────────────────────────────────────
# 🚀 INFERENCE PIPELINE

if flair_file and t1ce_file and t2_file:
    try:
        flair = load_and_normalize_nifti(flair_file)
        t1ce = load_and_normalize_nifti(t1ce_file)
        t2 = load_and_normalize_nifti(t2_file)

        if flair.shape != (128, 128, 128):
            st.warning(f"Input shape is {flair.shape}. Resizing to (128, 128, 128) is required. ❌ Auto-resizing not enabled.")
            st.stop()

        input_data = preprocess_inputs(flair, t1ce, t2)
        input_tensor = np.expand_dims(input_data, axis=0).astype(np.float32)  # (1, 128, 128, 128, 3)

        st.info("Running model inference...")
        prediction = model.predict(input_tensor)
        pred_mask = np.argmax(prediction[0], axis=-1)  # (128, 128, 128)

        # ───── Visualization ─────
        st.success("Segmentation completed ✅. Visualizing middle slice.")

        slice_idx = 64
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(input_data[:, :, slice_idx, 0], cmap='gray')
        axes[0].set_title("FLAIR")
        axes[1].imshow(input_data[:, :, slice_idx, 1], cmap='gray')
        axes[1].set_title("T1ce")
        axes[2].imshow(input_data[:, :, slice_idx, 2], cmap='gray')
        axes[2].set_title("T2")
        axes[3].imshow(pred_mask[:, :, slice_idx], cmap='viridis')
        axes[3].set_title("Predicted Mask")
        for ax in axes: ax.axis('off')
        st.pyplot(fig)

        # ───── Download ─────
        st.download_button(
            label="📥 Download Predicted Mask (.npy)",
            data=pred_mask.astype(np.uint8).tobytes(),
            file_name="segmented_mask.npy",
            mime="application/octet-stream"
        )

    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
else:
    st.info("Upload all three required NIfTI files to start segmentation.")
