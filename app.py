import streamlit as st
from PIL import Image
import os
import atexit
import shutil
import json
import cv2
import numpy as np
from utils import predict_segformer, predict_mask2former, predict_with_threshold, visualize_image_and_segments, visualize_image_and_segments_thresh
from colors import find_unique_colors, unpack_legend, get_user_defined_labels

st.title("Semantic Segmentation")

# Path to save the uploaded images
save_folder = "current_image"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

def cleanup():
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
        print(f"Deleted {save_folder}")
    if os.path.exists("Output"):
        shutil.rmtree("Output")
        print("Deleted Output")
    if os.path.exists("seg_map"):
        shutil.rmtree("seg_map")
        print("Deleted seg_map")
    if os.path.exists("true"):
        shutil.rmtree("true")
        print("Deleted true")

atexit.register(cleanup)

# Uploading the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")
# Dropdown menu to select segmentation method
method = st.selectbox('Choose a segmentation method:', 
                    ['Segformer', 'Mask2Former', 'Thresholding'], key="method_selectbox")



checkbox = st.checkbox("Upload the mask", key="mask_checkbox")
if checkbox:
    st.write("Please upload the mask image")
    true_value = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="mask_uploader")
    if uploaded_file is not None and true_value is not None:
        true_img = Image.open(true_value)
        img = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption='Original Image', use_column_width=True)
        with col2:
            st.image(true_img, caption='True Segmentation Map', use_column_width=True)

        if not os.path.exists("Output"):
            os.makedirs("Output")
        
        if not os.path.exists("seg_map"):
            os.makedirs("seg_map")
        
        if not os.path.exists("true"):
            os.makedirs("true")

        if not os.path.exists("true_legend"):
            os.makedirs("true_legend")
        
        save_path = os.path.join(save_folder, uploaded_file.name)
        true_path = os.path.join("true", uploaded_file.name)
        img.save(save_path)
        true_img.save(true_path)
        if method!="Thresholding":
            colors = find_unique_colors(true_path)
            l = unpack_legend(method)
            labels = get_user_defined_labels(colors, l)
            with open(f'true_legend\{method.lower()}.json', 'w') as json_file:
                json.dump(labels, json_file, indent=4)

        
        st.success(f"Image saved to {save_path}")
        st.success(f"Segmentation Map saved to {true_path}")

        if st.button('Run Segmentation', key="run_segmentation_button"):

            if method == 'Segformer':
                predicted_segmentation_map = predict_segformer(save_path)
                visualize_image_and_segments(predicted_segmentation_map, save_path, "segformer")
            elif method == 'Mask2Former':
                predicted_segmentation_map = predict_mask2former(save_path)
                visualize_image_and_segments(predicted_segmentation_map, save_path, "mask2former")
            elif method == 'Thresholding':
                predicted_segmentation_map = predict_with_threshold(save_path)
                visualize_image_and_segments_thresh(predicted_segmentation_map, save_path, "thresholding")

            st.success(f"Segmentation method '{method}' completed")
            
            # Display the segmented images
            true_image = Image.open(true_value).resize((128, 128))
            image = Image.open(uploaded_file).resize((128, 128))
            image_seg = Image.open(f"seg_map\\{method.lower()}\{uploaded_file.name}").resize((128, 128))
            image_combined = Image.open(f"Output\\{method.lower()}\{uploaded_file.name}").resize((128, 128))

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.image(image, caption='Original Image', use_column_width=True)
            with col2:
                st.image(true_image, caption='True Segmentation Map', use_column_width=True)
            with col3:
                st.image(image_seg, caption='Segmentation Map', use_column_width=True)
            with col4:
                st.image(image_combined, caption='Combined Predicted Image', use_column_width=True)


else:
    if uploaded_file is not None:
        # To read image file buffer as a PIL Image:
        img = Image.open(uploaded_file)
        
        # Display image
        st.image(img, caption='Uploaded Image.', use_column_width=True)

        if not os.path.exists("Output"):
            os.makedirs("Output")
        
        if not os.path.exists("seg_map"):
            os.makedirs("seg_map")

        if st.button('Run Segmentation', key="run_segmentation_button_no_mask"):
            save_path = os.path.join(save_folder, uploaded_file.name)
            img.save(save_path)
            st.success(f"Image saved to {save_path}")

            if method == 'Segformer':
                predicted_segmentation_map = predict_segformer(save_path)
                visualize_image_and_segments(predicted_segmentation_map, save_path, "segformer")
            elif method == 'Mask2Former':
                predicted_segmentation_map = predict_mask2former(save_path)
                visualize_image_and_segments(predicted_segmentation_map, save_path, "mask2former")
            elif method == 'Thresholding':
                predicted_segmentation_map = predict_with_threshold(save_path)
                visualize_image_and_segments_thresh(predicted_segmentation_map, save_path, "thresholding")

            st.success(f"Segmentation method '{method}' completed")
            
            # Display the segmented images
            image_normal = Image.open(save_path).resize((128, 128))
            image_seg = Image.open(f"seg_map\\{method.lower()}\{uploaded_file.name}").resize((128, 128))
            image_combined = Image.open(f"Output\\{method.lower()}\{uploaded_file.name}").resize((128, 128))

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(image_normal, caption='Original Image', use_column_width=True)
            with col2:
                st.image(image_seg, caption='Segmentation Map', use_column_width=True)
            with col3:
                st.image(image_combined, caption='Combined Image', use_column_width=True)
