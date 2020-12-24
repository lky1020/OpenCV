# Project 3 - Number Plate Detection
import cv2
import numpy as np

numberPlateCascade = cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")
minArea = 500
userChoice = 1

while True:

    img = cv2.imread("Resources/p" + str(userChoice) + ".jpg")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = numberPlateCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in numberPlates:
        area = w * h

        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, (y + h) + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            imgRoi = img[y:y+h, x:x+w]      # Region of number plate
            cv2.imshow("ROI", imgRoi)

    cv2.imshow("Result", img)
    cv2.waitKey(0)

    if cv2.waitKey(0) & 0xFF == ord('1'):
        userChoice = 1
        print("1")

    elif cv2.waitKey(0) & 0xFF == ord('2'):
        userChoice = 2
        print("2")

    elif cv2.waitKey(0) & 0xFF == ord('3'):
        userChoice = 3
        print("3")

    elif cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
