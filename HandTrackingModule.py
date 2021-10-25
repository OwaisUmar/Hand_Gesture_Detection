import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectCon = detectCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPositions(self, img, handNo=0, myID=[-1], draw=True):
        xList = []
        yList = []
        self.lmList = []
        self.boxCoord = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if (-1 in myID or id in myID) and draw:
                    cv2.circle(img, (cx, cy), 20, (0, 255, 0), cv2.FILLED)
            self.boxCoord = min(xList), min(yList), max(xList), max(yList)
        return self.lmList, self.boxCoord

    def getDistance(self, img, a, b, draw=True):
        x1, y1 = self.lmList[a][1], self.lmList[a][2]
        x2, y2 = self.lmList[b][1], self.lmList[b][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (cx, cy), 20, (0, 255, 0), cv2.FILLED)
            cv2.rectangle(img, (self.boxCoord[0] - 20, self.boxCoord[1] - 20),
                          (self.boxCoord[2] + 20, self.boxCoord[3] + 20), (0, 255, 0), 2)

        dist = math.hypot(x2 - x1, y2 - y1)
        return dist, [cx, cy]

    def fingersUp(self):
        fingersStatus = []
        fingerTipIDs = [4, 8, 12, 16, 20]

        if len(self.lmList):
            if self.lmList[fingerTipIDs[0]][1] > self.lmList[fingerTipIDs[0]-1][1]:
                fingersStatus.append(1)
            else:
                fingersStatus.append(0)

            for id in range(1, 5):
                if self.lmList[fingerTipIDs[id]][2] < self.lmList[fingerTipIDs[id]-2][2]:
                    fingersStatus.append(1)
                else:
                    fingersStatus.append(0)
        return fingersStatus

def main():
    cap = cv2.VideoCapture(1)
    pTime = 0
    cTime = 0
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, boxCoord = detector.findPositions(img, draw=False)

        fingersUp = detector.fingersUp()
        if len(fingersUp):
            print(fingersUp)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 4)
        # print(lmList)
        cv2.imshow("Webcam", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
