import cv2
import mediapipe as mp
import time


class Detector:
    def __init__(self, mode=False, maxHands=2, detectCon=0.5, trackCon=0.5):
        self.static_image_mode = mode
        self.max_num_hands = maxHands
        self.min_detection_confidence = detectCon
        self.min_tracking_confidence = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(mode, maxHands, detectCon, trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for handLMS in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPositions(self, img, hand=0, draw=True):
            for id, lm in enumerate(handLMS.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id == 8:
                    cv2.circle(img, (cx, cy), 10, (255,255,255), cv2.FILLED)


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = Detector

    while True:
        success, img = cap.read()

        img = detector.findHands(self=success, img=img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 4)
        cv2.imshow("Webcam", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
