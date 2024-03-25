import cv2

import numpy as np

import GraphOperator as go

def segment_image(sigma, k, min_size, img):
    float_img = np.asarray(img, dtype=float) # Calculate the image array in a float format

    gaussian_img = cv2.GaussianBlur(float_img, (5, 5), sigma) # Performs Gaussian Blur
    b, g, r = cv2.split(gaussian_img) # Splits into RGB images
    smooth_img = (r, g, b)
    # smooth_img=gaussian_img

    height, width, channel = img.shape
    graph = go.build_graph(smooth_img, width, height) # Constructs the graph

    weight = lambda edge: edge[2] # Distance between the 2 points in each edge.
    sorted_graph = sorted(graph, key=weight) # Sorts based on the distance between 2 points

    ufset = go.segment_graph(sorted_graph, width * height, k) # Based on the segment distance and the area, divides the graph into segments. 
    ufset = go.remove_small_component(ufset, sorted_graph, min_size) # Merge 2 points if they are very small. 

    return ufset
