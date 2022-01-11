from cv2 import cv2
from time import sleep
from pynput.keyboard import Controller
from cvzone import cornerRect
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 1230)
cap.set(4, 600)

detector = htm.handDetector(maxHands=1)
keys = [['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']', '\\'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', "'"],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/']]
buttonList = []
textOut = ''


def drawAll(img, buttonList):
    cv2.rectangle(img, (50, 400), (1100, 460), (0, 50, 0), cv2.FILLED)  # Text box
    # cornerRect(img, (50, 400, 1050, 60), 20, 4)
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cornerRect(img, (x, y, w, h), 15, 4)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, button.text, (x + 12, y + 35), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    return img


class Button:
    def __init__(self, pos, text, size=(50, 50)):
        self.pos = pos
        self.size = size
        self.text = text


y = 50
for i in range(len(keys)):
    x = 50 + i*20

    if i == 3:
        buttonList.append(Button((x+10, y), '^ Shift', size=(110, 50)))     # Shift key
        x += 130
    for j, key in enumerate(keys[i]):
        if key == '\\':
            buttonList.append(Button((x, y), key, size=(120, 50)))
        else:
            buttonList.append(Button((x, y), key))
            x += 60
    if i == 0:
        buttonList.append(Button((x, y), '<--', size=(110, 50)))     # Backspace key
    if i == 2:
        buttonList.append(Button((x, y), ' Enter', size=(120, 50)))     # Enter key
    y += 60
    x += 10

buttonList.append(Button((180, y), ' ', size=(560, 50)))     # Space key
cursorPos = 60

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPositions(img)
    img = drawAll(img, buttonList)

    if lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            if x < lmList[8][1] < x+w and y < lmList[8][2] < y+h:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), cv2.FILLED)
                cv2.putText(img, button.text, (x + 15, y + 35), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                dist, _ = detector.getDistance(img, 8, 12, draw=False)
                if dist < 30:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 200), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 15, y + 35), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    if button.text == '<--':
                        textOut = textOut[:-1]
                    else:
                        textOut += button.text
                    cursorPos += 30
                    sleep(0.5)
    cv2.line(img, (cursorPos, 450), (cursorPos + 30, 450), (255, 255, 255), 2)
    cv2.putText(img, textOut, (60, 440), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
