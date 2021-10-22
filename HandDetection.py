import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime=0
cTime=0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            for id, lm in enumerate(handLMS.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id == 8:
                    cv2.circle(img, (cx, cy), 10, (255,255,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 4)
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)