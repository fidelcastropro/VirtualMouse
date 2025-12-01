import cv2 as cv
import numpy as np
import mediapipe as mp
from math import hypot
import pyautogui
from pynput.mouse import Button, Controller
mouse = Controller()

screenWidth,screenHeight = pyautogui.size()

def calculate_Angle(fingerPoints):
    if len(fingerPoints) == 3:
        a = fingerPoints[0]
        b = fingerPoints[1]
        c = fingerPoints[2]

        radians = np.arctan2(c[1] - b[1],c[0] - b[0]) - np.arctan2(a[1] - b[1],a[0] - b[0])
        angle = np.abs(np.degrees(radians))

        return angle
    return


def moveMouse(allLandmarks):
    index_Finger_Tip = allLandmarks[8]

    x = index_Finger_Tip[0] * screenWidth * 1.5
    y = index_Finger_Tip[1] * screenHeight * 1.5

    pyautogui.moveTo(x,y)



def calculate_Distance(landmark_list):
    if len(landmark_list) < 2:
        return
    (x1,y1) = (landmark_list[0][0] * screenWidth , landmark_list[0][1] * screenHeight)
    (x2,y2) = (landmark_list[1][0] * screenWidth , landmark_list[1][1] * screenHeight)

    return hypot((x2-x1),(y2-y1))


def detectGesture(allLandmarks,frame):
    if len(allLandmarks) >= 21:
        stage = 'Up'
        distPoints = [allLandmarks[4],allLandmarks[5]]
        indexFingerPoints = [allLandmarks[5],allLandmarks[6],allLandmarks[8]]
        middleFingerPoints = [allLandmarks[9],allLandmarks[10],allLandmarks[12]]
        if calculate_Distance(distPoints) < 50 and calculate_Angle(indexFingerPoints) > 90 and calculate_Angle(middleFingerPoints) > 90:
            cv.putText(frame,"Mouse movement",(10,10),cv.FONT_HERSHEY_COMPLEX,1.5,(0,255,0),2)
            moveMouse(allLandmarks)
        elif calculate_Distance(distPoints) > 50 and calculate_Angle(indexFingerPoints) < 70 and calculate_Angle(middleFingerPoints) > 90:
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv.putText(frame,"Left Click",(10,10),cv.FONT_HERSHEY_COMPLEX,1.5,(0,255,0),2)
        elif calculate_Distance(distPoints) > 50 and calculate_Angle(indexFingerPoints) > 90 and calculate_Angle(middleFingerPoints) < 70:
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv.putText(frame,"Right Click",(10,10),cv.FONT_HERSHEY_COMPLEX,1.5,(0,255,0),2)
        elif calculate_Distance(distPoints) > 50 and calculate_Angle(indexFingerPoints) < 70 and calculate_Angle(middleFingerPoints) < 70:
            mouse.press(Button.left)
            mouse.release(Button.left)
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv.putText(frame,"Double Click",(10,10),cv.FONT_HERSHEY_COMPLEX,1.5,(0,255,0),2)


mpHands = mp.solutions.hands
hands = mpHands.Hands(
    model_complexity = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7,
    max_num_hands = 1,
    static_image_mode = False)
draw = mp.solutions.drawing_utils


def main():
    cam = cv.VideoCapture(0)
    print("Camera Started")
    stage = 'up'
    try :
        while cam.isOpened():
            returnVal,frame = cam.read()
            if not returnVal:
                break
            frame = cv.flip(frame,1)
            frameRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            if processed.multi_hand_landmarks:
                oneHandLandmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame,oneHandLandmarks,mpHands.HAND_CONNECTIONS)
                allLandmarks = []
    
                for onein21 in oneHandLandmarks.landmark:
                    allLandmarks.append((onein21.x,onein21.y))

                detectGesture(allLandmarks,frame)

            cv.imshow('WebCam',frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()