from cv2 import cv2
from time import sleep
from pynput.keyboard import Controller, Key
from cvzone import cornerRect
import numpy as np
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 1230)
cap.set(4, 600)

keyboard = Controller()
detector = htm.HandDetector(maxHands=1)
capsOff = [['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\'],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'"],
        ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/']]

capsOn = [['~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+'],
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '{', '}', '|'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':', '"'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '?']]


caps = False
buttonList = []
textOut = ''


def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    cv2.rectangle(imgNew, (50, 500), (1200, 560), (100, 0, 100), cv2.FILLED)  # Text box
    cornerRect(imgNew, (1150, 50, 80, 50), 18, 4)
    cv2.rectangle(imgNew, (1150, 50), (1230, 100), (100, 0, 200), cv2.FILLED)  # quit button
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cornerRect(imgNew, (x, y, w, h), 15, 4)
        if caps and button.text == "^ Shift":
            cv2.rectangle(img, (x, y), (x + w, y + h), (150, 0, 230), cv2.FILLED)       # indicating caps on
        else:
            cv2.rectangle(imgNew, (x, y), (x + w, y + h), (100, 0, 200), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x + 12, y + 35), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    out = img.copy()
    alpha = 0.2
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1-alpha, 0)[mask]
    return out


class Button:
    def __init__(self, pos, text, size=(50, 50)):
        self.pos = pos
        self.size = size
        self.text = text


def drawButton(myKeys):
    y = 50
    for i in range(len(myKeys)):
        x = 50 + i*20

        for j, key in enumerate(myKeys[i]):
            if key in '\\|':
                buttonList.append(Button((x, y), key, size=(120, 50)))
            else:
                buttonList.append(Button((x, y), key))
                x += 60
        if i == 0:
            buttonList.append(Button((x, y), '<--', size=(110, 50)))     # Backspace key
        elif i == 2:
            buttonList.append(Button((x, y), ' Enter', size=(125, 50)))     # Enter key
        elif i == 3:
            buttonList.append(Button((x, y), '^ Shift', size=(140, 50)))     # Shift key
        y += 60
        x += 10
    buttonList.append(Button((180, y), ' ', size=(560, 50)))  # Space key
    buttonList.append(Button((1150, 50), 'Quit', size=(80, 50)))  # quit button


drawButton(capsOff)
cursorPos = 60

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=False)
    lmList, bbox = detector.findPositions(img, draw=False)
    img = drawAll(img, buttonList)
    close = False

    if lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            if x < lmList[8][1] < x+w and y < lmList[8][2] < y+h:
                cornerRect(img, (x, y, w + 5, h + 5), 15, 4)
                cv2.rectangle(img, (x, y), (x + w + 5, y + h + 5), (200, 0, 200), cv2.FILLED)
                cv2.putText(img, button.text, (x + 12, y + 35), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                dist, _ = detector.getDistance(img, 8, 12, draw=False)
                if dist < 50:
                    if button.text == ' Enter':
                        keyboard.press(Key.enter)
                    elif button.text == '<--':
                        keyboard.press(Key.backspace)
                    elif len(button.text) == 1:
                        keyboard.press(button.text)
                    cv2.rectangle(img, (x, y), (x + w + 5, y + h + 5), (0, 100, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 12, y + 35), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    if button.text == '<--':
                        textOut = textOut[:-1]
                        cursorPos -= 30
                    elif button.text == '^ Shift':
                        buttonList = []
                        if caps:
                            keys = capsOff
                            caps = False
                        else:
                            keys = capsOn
                            caps = True
                        drawButton(keys)

                    elif len(button.text) == 1:
                        textOut += button.text
                        cursorPos += 23
                    sleep(0.5)
                    if button.text == 'Quit':  # close if Quit is clicked
                        close = True

    cv2.line(img, (cursorPos, 555), (cursorPos + 25, 555), (255, 255, 255), 1)      # cursor
    cv2.putText(img, textOut, (60, 540), cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 255, 255), 2)     # text display
    if lmList:
        cv2.circle(img, (lmList[8][1], lmList[8][2]), 3, (0, 255, 0), thickness=cv2.FILLED)     # finger pointer

    detector.showFPS(img, (1150, 630), fontSize=1, box=True)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & close:
        break

