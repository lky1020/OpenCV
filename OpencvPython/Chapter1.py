# Chapter 1 - import img && video(mp4 and default webcam)
import cv2

print("Package Imported")

# img
"""img = cv2.imread("Resources/lena.jpg")

cv2.imshow("Output", img)
cv2.waitKey(0)"""

# video
faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)  # id = 0 == Default webcam
cap.set(3, 640)  # set width (id = 3)
cap.set(4, 480)  # set width (id = 4)

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 5, minSize=(20, 20))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = imgGray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
