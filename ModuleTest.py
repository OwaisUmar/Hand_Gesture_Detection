import HandTrackingModule as HTM
import cv2
import time

cap = cv2.VideoCapture(1)
pTime = 0
cTime = 0
detector = HTM.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPositions(img, myID=[4, 8, 12, 16, 20, 0])
    if len(lmList):
        print(lmList[8])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 4)
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
