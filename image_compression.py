# import libraries
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Defining the required functions for segmentation
def initialize_K_centroids(X, K):
    m = len(X)
    return X[np.random.choice(m, K, replace=False), :]

def find_closest_centroids(X, centroids):
    m = len(X)
    c = np.zeros(m)
    for i in range(m):
        # Find distances
        distances = np.linalg.norm(X[i] - centroids, axis=1)

        # Assign closest cluster to c[i]
        c[i] = np.argmin(distances)

    return c

# function for computing means
def compute_means(X, idx, K):
    _, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        examples = X[np.where(idx == k)]
        mean = [np.mean(column) for column in examples.T]
        centroids[k] = mean
    return centroids

# function for find k-means
def find_k_means(X, K, max_iters=10):
    centroids = initialize_K_centroids(X, K)
    previous_centroids = centroids
    for _ in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_means(X, idx, K)
        if (centroids == previous_centroids).all():
            # The centroids aren't moving anymore.
            return centroids
        else:
            previous_centroids = centroids

    return centroids, idx

# loading image from local machine 
def Load_image(path):
    image = Image.open(path)
    return np.asarray(image) / 255

# Importing the image 
image_path = '/Users/pranav/Desktop/Image compress k Means/River/image.png'
image = Load_image(image_path)
w, h, d = image.shape

# Reshaping the image to feed into kmeans algorithm
X = image.reshape((w * h, d))
# the desired number of colors in the compressed image
K = 20 

# obtaining clusters
colors, _ = find_k_means(X, K, max_iters=10)
idx = find_closest_centroids(X, colors)

idx = np.array(idx, dtype=np.uint8)
X_reconstructed = np.array(colors[idx, :] * 255, dtype=np.uint8).reshape((w, h, d))

compressed_image = Image.fromarray(X_reconstructed)

# saving the compressed image 
compressed_image.save('/Users/pranav/Desktop/Image compress k Means/River/out.png')

# displaying the image
plt.figure(figsize=(10,10))
plt.imshow(compressed_image)
plt.show()