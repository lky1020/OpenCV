# Project 2 - Document Scanner
import cv2
import numpy as np

widthImg = 480
heightImg = 640


cap = cv2.VideoCapture(0)
cap.set(3, widthImg)
cap.set(4, heightImg)
cap.set(10, 150)


def screenResize(img):
    # define the screen resolution
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)

    # resized window width and height
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    # cv2.WINDOW_NORMAL makes the output window resizealbe
    cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)

    # resize the window according to the screen resolution
    cv2.resizeWindow('Resized Window', window_width, window_height)

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)

    kernel = np.ones((5, 5))
    imgDialation = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDialation, kernel, iterations=1)

    return imgThres


def getContours(img):
    biggest = np.array([])
    maxArea = 0

    # Get the detail of contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Handle the contours
    for cnt in contours:
        area = cv2.contourArea(cnt)  # Calculate the area

        if area > 5000:
            peri = cv2.arcLength(cnt, True)  # Find arc length of cnt

            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # detect the corner of the cnt

            if area > maxArea and len(approx) == 4:
                biggest = approx  # store the corner point to the biggest array
                maxArea = area

    cv2.drawContours(imgContours, biggest, -1, (255, 0, 0), 15)  # Draw the contours to the imgContours
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    print("add", add)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def getWarp(img, biggest):
    biggest = reorder(biggest)

    pts1 = np.float32(biggest)  # the edge point of the cards (biggest contain the edge point in array
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # output for the cards
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))

    return imgCropped

"""
# In this project, i will use image, since i doesn't has camera that can move easily
img = cv2.imread("Resources/paper.jpg")

imgContours = img.copy()
imgThres = preProcessing(img)
biggest = getContours(imgThres)

if biggest.size != 0:
    imgWrapped = getWarp(img, biggest)
    imgArray = ([img, imgThres], [imgContours, imgWrapped])
    cv2.imshow("Result", imgWrapped)

else:
    imgArray = ([img, imgThres], [img, img])

stackedImages = stackImages(0.5, imgArray)

screenResize(img)
cv2.imshow('Resized Window', stackedImages)

cv2.waitKey(0)
"""

while True:
    success, img = cap.read()
    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()

    imgThres = preProcessing(img)
    biggest = getContours(imgThres)

    if biggest.size != 0:
        imgWrapped = getWarp(img, biggest)
        imgArray = ([imgContours, imgWrapped])
        cv2.imshow("Result", imgWrapped)

    else:
        imgArray = ([img, imgContours])

    stackedImages = stackImages(0.5, imgArray)

    cv2.imshow('Workflow', stackedImages)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break