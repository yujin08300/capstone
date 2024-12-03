import cv2
import argparse
import threading
from IPC_Library import IPC_SendPacketWithIPCHeader, IPC_ReceivePacketFromIPCHeader
from IPC_Library import TCC_IPC_CMD_CA72_EDUCATION_CAN_DEMO, IPC_IPC_CMD_CA72_EDUCATION_CAN_DEMO_START
from IPC_Library import parse_hex_data, parse_string_data


def sendtoCAN(channel, canId, sndDataHex):
    sndData = parse_hex_data(sndDataHex)
    uiLength = len(sndData)
    ret = IPC_SendPacketWithIPCHeader(
        "/dev/tcc_ipc_micom",
        channel,
        TCC_IPC_CMD_CA72_EDUCATION_CAN_DEMO,
        IPC_IPC_CMD_CA72_EDUCATION_CAN_DEMO_START,
        canId,
        sndData,
        uiLength,
    )


def detect_features(image):
    # Load Haar cascades
    cascade_face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    cascade_eye_detector = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    cascade_smile_detector = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')
    cascade_cat_face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalcatface.xml')

    # Resize and convert to grayscale
    image_resized = cv2.resize(image, (755, 500))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Face detection
    face_detections = cascade_face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in face_detections:
        cv2.rectangle(image_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print("face_detections")
        sendtoCAN(0, 1, "1")

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image_resized[y:y+h, x:x+w]

        # Eye detection
        eye_detections = cascade_eye_detector.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=6, minSize=(10, 10), maxSize=(30, 30))
        for (ex, ey, ew, eh) in eye_detections:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
            print("eye_detections")
            sendtoCAN(1, 2, "2")

        # Smile detection
        smile_detections = cascade_smile_detector.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
        for (sx, sy, sw, sh) in smile_detections:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
            print("smile_detections")
            sendtoCAN(2, 3, "3")

    # Cat face detection
    cat_detections = cascade_cat_face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in cat_detections:
        cv2.rectangle(image_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)
        print("cat_detections")
        sendtoCAN(3, 4, "4")

    return image_resized


print("model load")
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
print("camera connect")

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Detect features
    result = detect_features(img)

    # Display result
    cv2.imshow('Feature Detection', result)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
