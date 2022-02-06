import numpy as np
from cv2 import cv2
import HandTrackingModule as htm
from autopy import mouse, screen
import pyautogui
from cvzone import cornerRect

wScr, hScr = screen.size()
tRange = 75  # Touching range
smoothening = 5
prevLocX, prevLocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, 1230)
cap.set(4, 600)
mouseClicked = False
detector = htm.HandDetector(maxHands=1)

while True:
    # 1. Find hand landmarks
    success, img = cap.read()
    hCam, wCam = img.shape[0:2]
    img = detector.findHands(img, draw=False)
    LMList, bbox = detector.findPositions(img, draw=False)
    # cv2.rectangle(img, (tRange, tRange), (wCam - tRange, hCam - tRange), (0, 255, 0), thickness=2)
    cornerRect(img, (tRange, tRange, wCam - 2*tRange, hCam - 2*tRange))

    # 2. Get the tip of index and middle fingers
    if len(LMList):
        x1, y1 = LMList[8][1:]
        x2, y2 = LMList[12][1:]
        # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. Convert Coordinates
        x3 = np.interp(x1, (tRange, wCam - tRange), (0, wScr))
        y3 = np.interp(y1, (tRange, hCam - tRange), (0, hScr))
        cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)

        # 5. Smoothen Values
        currLocX = prevLocX + (x3 - prevLocX) / smoothening
        currLocY = prevLocY + (y3 - prevLocY) / smoothening
        # currLocX, currLocY = x3, y3

        # 6. Only index fingers up: Moving mode
        if fingers[1] and not fingers[2]:
            # 7. Move mouse
            mouse.move(currLocX, currLocY)
            prevLocX, prevLocY = currLocX, currLocY

        # 8. Both index and middle fingers up: Clicking mode
        if fingers[1] and fingers[2]:
            # 9. Find distance between fingers
            l, _ = detector.getDistance(img, 8, 12, draw=False)
            # print(l)
            if l < 50:
                cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
                # 10. Click mouse if distance is short
                pyautogui.click()

        # 11. Right click when thumb is up
        if fingers[0] and fingers[1]:
            pyautogui.click(button='right')

    detector.showFPS(img, (30, 30), box=True)
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
