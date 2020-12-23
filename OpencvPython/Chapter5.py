# Chapter 5 - Warp Rrespective
import cv2
import numpy as np

img = cv2.imread("Resources/cards.jpg")

width, height = 250, 350        # common size of the cards

pts1 = np.float32([[111, 219], [287, 188], [154, 482], [352, 440]])     # the edge point of the cards
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])   # output for the cards
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow("Image", img)
cv2.imshow("Image Output", imgOutput)
cv2.waitKey(0)