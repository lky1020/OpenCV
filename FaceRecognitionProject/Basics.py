import cv2
import face_recognition

imgElon = face_recognition.load_image_file("ImageBasics/Elon Musk.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgElonTest = face_recognition.load_image_file("ImageBasics/Elon Test.jpg")
imgElonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2RGB)

# input image to get the location of the image (Plot)
faceLocation = face_recognition.face_locations(imgElon)[0]  # return 4 element
# input image to encode the image
encodeElon = face_recognition.face_encodings(imgElon)[0]
# draw rectangle around the face
cv2.rectangle(imgElon, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (255, 0, 255), 2)

# input image to get the location of the image (Plot)
faceLocationTest = face_recognition.face_locations(imgElonTest)[0]  # return 4 element
# input image to encode the image
encodeElonTest = face_recognition.face_encodings(imgElonTest)[0]
# draw rectangle around the face
cv2.rectangle(imgElonTest, (faceLocationTest[3], faceLocationTest[0]),
              (faceLocationTest[1], faceLocationTest[2]), (255, 0, 255), 2)

result = face_recognition.compare_faces([encodeElon], encodeElonTest)
faceDistance = face_recognition.face_distance([encodeElon], encodeElonTest)
cv2.putText(imgElonTest, f"{result} {round(faceDistance[0], 2)}", (faceLocationTest[3], faceLocationTest[2] + 25),
            cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
print(result, faceDistance)

cv2.imshow("Elon Musk", imgElon)
cv2.imshow("Elon Test", imgElonTest)
cv2.waitKey(0)