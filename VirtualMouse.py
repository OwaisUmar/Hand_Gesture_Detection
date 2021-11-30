import numpy as np
from cv2 import cv2
import HandTrackingModule as htm
import mediapipe as mp
import time
import autopy

wScr, hScr = autopy.screen.size()
tRange = 75  # Touching range
smoothening = 5
prevLocX, prevLocY = 0, 0

cap = cv2.VideoCapture(0)
cTime = 0
pTime = 0
detector = htm.handDetector(maxHands=1)

while True:
    # 1. Find hand landmarks
    success, img = cap.read()
    hCam, wCam = img.shape[0:2]
    img = detector.findHands(img)
    lmlist, bbox = detector.findPositions(img, draw=False)
    cv2.rectangle(img, (tRange, tRange), (wCam - tRange, hCam - tRange), (0, 255, 0), thickness=2)

    # 2. Get the tip of index and middle fingers
    if len(lmlist):
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]
        # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. Only index fingers up: Moving mode

        if fingers[1] and not fingers[2]:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (tRange, wCam - tRange), (0, wScr))
            y3 = np.interp(y1, (tRange, hCam - tRange), (0, hScr))
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            # 6. Smoothen Values
            currLocX = prevLocX+(x3-prevLocX)/smoothening
            currLocY = prevLocY+(y3-prevLocY)/smoothening
            # 7. Move mouse
            autopy.mouse.move(currLocX, currLocY)
            prevLocX, prevLocY = currLocX, currLocY

        # 8. Both index and middle fingers up: Clicking mode
        if fingers[1] and fingers[2]:
            # 9. Find distance between fingers
            l, _ = detector.getDistance(img, 8, 12, draw=False)
            print(l)
            if l < 30:
                cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
                # 10. Click mouse if distance is short
                autopy.mouse.click()

    # 11. Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "FPS: " + str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
