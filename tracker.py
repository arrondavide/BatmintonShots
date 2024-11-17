import cv2
from cvzone.PoseModule import PoseDetector
import socket


cap = cv2.VideoCapture(0)
cap.set(3, 450)
cap.set(4, 450)
success, img = cap.read()
h, w, _ = img.shape
detector = PoseDetector()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5056)


while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)
    posList = []
    if bboxInfo:
        lmString = ''
        for lm in lmList:
            posList.extend([lm[1], img.shape[0] - lm[2], lm[3]])
            # lmString += f'{lm[1]},{img.shape[0] - lm[2]},{lm[3]},'

        sock.sendto(str.encode(str(posList)), serverAddressPort)
    print(posList)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
