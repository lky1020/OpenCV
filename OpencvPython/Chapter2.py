#Chapter 2 - Basic Fucntions
import cv2
import numpy as np

img = cv2.imread("Resources/lena.jpg")
kernel = np.ones((3, 3), np.uint8)      #matrix 5 x 5 && unit8 = unsigned interger (0 - 255)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #cvtColor - convert color
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)  #ksize must be odd number
imgCanny = cv2.Canny(img, 100, 100)
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)   #iterations = tickness
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

cv2.imshow("Gray Image", imgGray)
cv2.imshow("Blur Image", imgBlur)
cv2.imshow("Canny Image", imgCanny)
cv2.imshow("Dialation Image", imgDialation)
cv2.imshow("Eroded Image", imgEroded)
cv2.waitKey(0)