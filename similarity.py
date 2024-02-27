import cv2
import numpy as np

def check_similarity(image1, image2):
    # Load images
    img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # Initialize feature detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Initialize feature matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Set a threshold for the number of matches
    threshold = 10

    # Check if the number of matches is above the threshold
    if len(matches) >= threshold:
        return True
    else:
        return False

# Example usage
image1_path = 'image1.jpg'
image2_path = 'image2.jpg'

if check_similarity(image1_path, image2_path):
    print("The images have similar objects.")
else:
    print("The images do not have similar objects.")
