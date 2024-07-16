import cv2
import numpy as np
import json
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

def find_unique_colors(true_path):
    image = cv2.imread(true_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image_rgb.reshape((-1, 3))
    unique_colors = np.unique(pixel_values, axis=0)
    return unique_colors

def unpack_legend(model_name):
    with open(f"legends\{model_name}.json", 'r') as f:
        l = json.load(f)
    return l

def get_user_defined_labels(colors, l):
    labels = {}
    for color in colors:
        col1, col2 = st.columns(2)
        
        with col1:
            color_preview = np.zeros((50, 50, 3), dtype=np.uint8)  # Create a blank image
            color_preview[:, :] = color  # Fill with the current color
            st.image(color_preview, caption=f"RGB({color[0]}, {color[1]}, {color[2]})")
        with col2:
            label = st.selectbox(
                f"Select label for RGB({color[0]}, {color[1]}, {color[2]})",
                list(l.values()) + ["background"]
            )
            labels[str(color)] = label  # Store label with string representation of color tuple

    if st.button('Select colors', key='color_selection'):
        # Display the selected labels
        st.write("Selected labels:", labels)
        return labels
