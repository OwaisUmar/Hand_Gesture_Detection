from cv2 import cv2
import numpy as np
import time
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

############################################################
wCam = 960
hCam = 540
############################################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
cTime = 0

detector = htm.handDetector(detectCon=0.8, maxHands=1)

# _______________Volume Panel________________
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeLevel = volume.GetVolumeRange()
minVol = volumeLevel[0]
maxVol = volumeLevel[1]
vol = 0
volBar = 239
volPercent = 0

while True:
    success, img = cap.read()
    detector.findHands(img)
    lmList, boxCoord = detector.findPositions(img, handNo=0, draw=True, myID=[4, 8])
    if len(lmList):

        # area = (boxCoord[2]-boxCoord[0])*(boxCoord[3]-boxCoord[1])//1000
        # print(area)
        # if 18 < area < 50:
        #     print("Yes")

        dist, centre = detector.getDistance(img, 4, 8)

        # Set Volume
        # vol = np.interp(dist, (30, 180), (minVol, maxVol))
        volBar = np.interp(dist, (30, 180), (240, 40))
        volPercent = np.interp(dist, (30, 180), (0, 100))
        interval = 10
        volBar = interval * round(volBar/interval)
        volPercent = interval * round(volPercent/interval)

        fingers = detector.fingersUp()
        currVol = volume.GetMasterVolumeLevelScalar()

        if not fingers[4]:
            volume.SetMasterVolumeLevelScalar(volPercent / 100, None)
            cv2.circle(img, (centre[0], centre[1]), 20, (255, 255, 0), cv2.FILLED)
            cv2.putText(img, 'Volume Set: ' + str(int(currVol * 100)), (430, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 3)
        else:
            cv2.putText(img, 'Volume Set: ' + str(int(currVol * 100)), (430, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (200, 0, 0), 3)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, 'FPS: ' + str(int(fps)), (830, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 0, 0), 3)

    if volPercent <= 60:
        cv2.rectangle(img, (40, int(volBar)), (80, 240), (0, 210, 0), thickness=cv2.FILLED)
    elif volPercent <= 85:
        cv2.rectangle(img, (40, int(volBar)), (80, 240), (0, 150, 255), thickness=cv2.FILLED)
    else:
        cv2.rectangle(img, (40, int(volBar)), (80, 240), (0, 0, 255), thickness=cv2.FILLED)

    cv2.rectangle(img, (40, 40), (80, 240), 0, 4)
    cv2.putText(img, str(int(volPercent)) + '%', (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0), 3)
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
