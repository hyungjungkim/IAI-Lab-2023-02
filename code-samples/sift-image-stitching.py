import cv2 as cv
import numpy as np

def keypoints(img1, img2, ratio=0.5, show_match=False):
    sift = cv.SIFT_create()

    img1_ = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_ = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(img1_, None)
    kp2, des2 = sift.detectAndCompute(img2_, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m,n in matches:
        if (m.distance/n.distance) < ratio:
            good_matches.append([m])

    if show_match:
        img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv.imshow('matching.jpg', img3)
        
    return kp1, kp2, good_matches

def find_homography(kp1, kp2, matches):
    src_points = []
    dst_points = []

    for match in matches:
        src_points.append(kp1[match[0].queryIdx].pt)
        dst_points.append(kp2[match[0].trainIdx].pt)
    
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)

    H, status = cv.findHomography(dst_points, src_points, cv.RANSAC, 4.0)

    return H

def create_mask(img1, img2, left=True):
    img1_height = img1.shape[0]
    img1_width = img1.shape[1]
    img2_width = img2.shape[1]
    height_panorama = img1_height
    width_panorama = img1_width + img2_width
    offset = int(img1_width * 0.25)
    barrier = img1_width
    mask = np.zeros((height_panorama, width_panorama))
    if left == True:
        mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
        mask[:, :barrier - offset] = 1
    else:
        mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
        mask[:, barrier + offset:] = 1
    return cv.merge([mask, mask, mask])

def blending(img1, img2):
    kp1, kp2, good_matches = keypoints(img1, img2)
    H = find_homography(kp1, kp2, good_matches)

    img1_height = img1.shape[0]
    img1_width = img1.shape[1]
    img2_width = img2.shape[1]
    height_panorama = img1_height
    width_panorama = img1_width + img2_width

    panorama1 = np.zeros((height_panorama, width_panorama, 3))
    mask1 = create_mask(img1, img2)
    panorama1[0:img1_height, 0:img1_width, :] = img1
    panorama1 *= mask1
    mask2 = create_mask(img1, img2, left=False)
    panorama2 = cv.warpPerspective(img2, H, (width_panorama, height_panorama)) * mask2
    result = panorama1 + panorama2

    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    return result[min_row:max_row, min_col:max_col, :]

if __name__ == '__main__':
    img1 = cv.imread('sample-desk-left.jpg')
    img2 = cv.imread('sample-desk-right.jpg')

    dst = blending(img1, img2)
    cv.imshow('Stitching result.', dst/256.)
    k = cv.waitKey(0) # Wait for a keystroke in the window