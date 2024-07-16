import cv2
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras

def load_image_and_get_shape(image_path):
    img = cv2.imread(image_path)
    H, W, C = img.shape
    return img, H, W, C

def palette():
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

def predict_segformer(image_path):
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    image_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    image = Image.open(image_path).resize((128,128))
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs)[0].numpy()
    x = image_path.split('\\')[-1]
    fname = f"seg_map\segformer\{x}"
    if os.path.exists("seg_map\segformer"):
        plt.imsave(fname,predicted_segmentation_map)
    else:
        os.mkdir("seg_map\segformer")
        plt.imsave(fname,predicted_segmentation_map)
    return predicted_segmentation_map


def predict_mask2former(image_path):
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
    image = Image.open(image_path).resize((384, 384))
    inputs = processor(images=image, return_tensors="pt", size=(384,384))
    outputs = model(**inputs)
    predicted_segmentation_map = processor.post_process_semantic_segmentation(outputs)[0].numpy()
    x = image_path.split('\\')[-1]
    fname = f"seg_map\mask2former\{x}"
    if os.path.exists("seg_map\mask2former"):
        plt.imsave(fname,predicted_segmentation_map)
    else:
        os.mkdir("seg_map\mask2former")
        plt.imsave(fname,predicted_segmentation_map)
    return predicted_segmentation_map

def predict_with_threshold(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x = image_path.split('\\')[-1]
    fname = f"seg_map\\thresholding\{x}"
    if os.path.exists(r"seg_map\\thresholding"):
        cv2.imwrite(fname,binary_image)
    else:
        os.mkdir(r"seg_map\\thresholding")
        cv2.imwrite(fname,binary_image)
    return binary_image


def visualize_image_and_segments(predicted_segmentation_map, image_path, model_used):
    import matplotlib.pyplot as plt

    image = Image.open(image_path).resize((predicted_segmentation_map.shape[0],predicted_segmentation_map.shape[1]))

    color_seg = np.zeros((predicted_segmentation_map.shape[0],
                        predicted_segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
    
    pa = np.array(palette())
    for label, color in enumerate(pa):
        color_seg[predicted_segmentation_map == label, :] = color

    color_seg = color_seg[..., ::-1]

    img = np.asarray(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    x = image_path.split('\\')[-1]
    fname = f"Output\{model_used}\{x}"
    if os.path.exists(f"Output\{model_used}"):
        plt.imsave(fname,img)
    else:
        os.mkdir(f"Output\{model_used}")
        plt.imsave(fname,img)

def visualize_image_and_segments_thresh(predicted_segmentation_map, image_path, model_used):
    import matplotlib.pyplot as plt

    image = Image.open(image_path).resize((predicted_segmentation_map.shape[1],predicted_segmentation_map.shape[0]))

    color_seg = np.zeros((predicted_segmentation_map.shape[0],
                        predicted_segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3

    pa = np.array(palette())
    for label, color in enumerate(pa):
        color_seg[predicted_segmentation_map == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.asarray(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    x = image_path.split('\\')[-1]
    fname = f"Output\{model_used}\{x}"
    if os.path.exists(f"Output\{model_used}"):
        plt.imsave(fname,img)
    else:
        os.mkdir(f"Output\{model_used}")
        plt.imsave(fname,img)