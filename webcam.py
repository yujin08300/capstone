import cv2
import argparse
import threading
from IPC_Library import IPC_SendPacketWithIPCHeader, IPC_ReceivePacketFromIPCHeader
from IPC_Library import TCC_IPC_CMD_CA72_EDUCATION_CAN_DEMO, IPC_IPC_CMD_CA72_EDUCATION_CAN_DEMO_START
from IPC_Library import parse_hex_data, parse_string_data


def sendtoCAN(channel, canId, sndDataHex):
        sndData = parse_hex_data(sndDataHex)
        uiLength = len(sndData)
        ret = IPC_SendPacketWithIPCHeader("/dev/tcc_ipc_micom", channel, TCC_IPC_CMD_CA72_EDUCATION_CAN_DEMO, IPC_IPC_CMD_CA72_EDUCATION_CAN_DEMO_START, canId, sndData, uiLength)

def face_eyes(image):
    cascade_face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    cascade_eye_detector = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

    image_resized = cv2.resize(image, (755, 500))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    face_detections = cascade_face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in face_detections:
        cv2.rectangle(image_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print("face_detections")
        sendtoCAN(0, 1, "1")

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image_resized[y:y+h, x:x+w]

        eye_detections = cascade_eye_detector.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=6, minSize=(10, 10), maxSize=(30, 30))
        for (ex, ey, ew, eh) in eye_detections:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
            print("eye_detections")
            sendtoCAN(1, 2, "2")

    return image_resized

print("model load")
cap = cv2.VideoCapture(-1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
print("camera connect")

while True:
    ret, img = cap.read()
    if not ret:
        break

    result = face_eyes(img)

    cv2.imshow('face_eyes', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
