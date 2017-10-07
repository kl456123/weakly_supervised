import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform, io, img_as_float


def outlier_removal(points, diag):
    neighbors = np.zeros((points.shape[0]))
    selPoints = np.empty((1, 2))
    for i in range(points.shape[0]):
        diff = np.sqrt(np.sum(np.square(points-points[i]), axis=1))
        neighbors[i] = np.sum(diff < diag)
    for i in range(points.shape[0]):
        if neighbors[i] > 0.05*points.shape[0]:
            selPoints = np.append(selPoints, points[i:i+1, :], axis=0)
    selPoints = selPoints[1:, :]
    selPoints = selPoints.astype(int)
    return selPoints


def heatmap(img, points, sigma=20):
    k = (np.min(img.shape[:2])) if (
        np.min(img.shape[:2]) % 2 == 1) else (np.min(img.shape[:2])-1)
    mask = np.zeros(img.shape[:2])
    shape = mask.shape
    for i in range(points.shape[0]):
        # Check if inside the image
        if points[i, 0] < shape[0] and points[i, 1] < shape[1]:
            mask[points[i, 0], points[i, 1]] += 1
    # Gaussian blur the points to get a nice heatmap
    blur = cv2.GaussianBlur(mask, (k, k), sigma)
    blur = blur*255/np.max(blur)
    return blur


def visualize(img_path, points, diag_percent, image_label):
    img = cv2.imread(img_path)
    b, g, r = cv2.split(img)
    img = cv2.merge((r, g, b))
    diag = math.sqrt(img.shape[0]**2 + img.shape[1]**2)*diag_percent
    values = np.asarray(points)
    selPoints = outlier_removal(values, diag)
    # Make heatmap and show images
    hm = heatmap(np.copy(img), selPoints)
    _, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img), ax[0].axis('off'), ax[0].set_title(image_label)
    ax[1].imshow(img), ax[1].axis('off'),
    ax[1].scatter(selPoints[:, 1], selPoints[:, 0]),
    ax[1].set_title('CNN Fixations')
    ax[2].imshow(img), ax[2].imshow(hm, 'jet', alpha=0.6),
    ax[2].axis('off'), ax[2].set_title('Heatmap')
    plt.show()
