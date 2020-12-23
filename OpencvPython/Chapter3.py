#Chapter 3 - Resizing & Cropping
import cv2

img = cv2.imread("Resources/lambo.png")
print(img.shape)    #get the shape of the image height, width, bgr

imgResize = cv2.resize(img, (300, 200)) #width = 300, height = 200
print(imgResize.shape)    #get the shape of the image height, width, bgr

imgCropped = img[0:200, 200:500]  #height first then width

cv2.imshow("Image", img)
cv2.imshow("Image Resize", imgResize)
cv2.imshow("Image Cropped", imgCropped)
cv2.waitKey(0)