import cv2
import numpy as np
import time
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


################################################################################################################
wCam=1280
hCam=720
################################################################################################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4,hCam)

pTime=0
cTime=0

detector=htm.handDetector(detectCon=0.8)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeLevel = volume.GetVolumeRange()
minVol=volumeLevel[0]
maxVol=volumeLevel[1]
vol=0
volBar=239
volpercent=0

while True:
    success, img = cap.read()
    detector.findHands(img)
    lmList = detector.findPositions(img, handNo=0, draw=True, myID=[4, 8])
    if len(lmList):
        # print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.line(img, (x1,y1), (x2,y2), (255,0,255),3)
        cv2.circle(img, (cx,cy), 10, (255, 0, 255), cv2.FILLED)
        dist = math.hypot(x2-x1, y2-y1)
        # print(dist)
        if(dist<50):
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        vol = np.interp(dist, (30, 180), (minVol, maxVol))
        volBar = np.interp(dist, (30, 180), (240, 40))
        volpercent = np.interp(dist, (30,180), (0, 100))
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, 'FPS: '+str(int(fps)), (1100,650), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 3)

    if volpercent<=60:
        cv2.rectangle(img, (40, int(volBar)), (80, 240), (0, 255, 0), thickness=cv2.FILLED)
    else:
        cv2.rectangle(img, (40, int(volBar)), (80, 240), (0, 0, 255), thickness=cv2.FILLED)

    cv2.rectangle(img, (40, 40), (80, 240), (0), 4)
    cv2.putText(img, str(int(volpercent))+'%', (40, 280), cv2.FONT_HERSHEY_TRIPLEX, 1, (0), 2)
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)