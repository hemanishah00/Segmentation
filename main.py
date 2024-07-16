import cv2
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import predict_segformer, predict_deeplab, predict_mask2former, predict_with_threshold, visualize_image_and_segments, visualize_image_and_segments_thresh

directory_path = r"Val"

for image in os.listdir(directory_path):
    image_path = f"{directory_path}\{image}"
    predicted_segmentation_map = predict_segformer(image_path)
    visualize_image_and_segments(predicted_segmentation_map, image_path, "segformer")

    predicted_segmentation_map = predict_deeplab(image_path)
    visualize_image_and_segments(predicted_segmentation_map, image_path, "deeplab")

    predicted_segmentation_map = predict_mask2former(image_path)
    visualize_image_and_segments(predicted_segmentation_map, image_path, "mask2former")

    predicted_segmentation_map = predict_with_threshold(image_path)
    visualize_image_and_segments_thresh(predicted_segmentation_map, image_path, "threshold")