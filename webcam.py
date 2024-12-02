import cv2
from IPC_Library import IPC_SendPacketWithIPCHeader, parse_hex_data


def sendtoCAN(channel, canId, sndDataHex):
    sndData = parse_hex_data(sndDataHex)
    uiLength = len(sndData)
    ret = IPC_SendPacketWithIPCHeader("/dev/tcc_ipc_micom", channel, 0, 0, canId, sndData, uiLength)


def detect_face_eyes_smile_cat(image):
    # Load Haar Cascade models
    cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cascade_eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    cascade_smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    cascade_cat = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')

    # Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = cascade_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print("Face detected!")
        sendtoCAN(0, 1, "1")  # Send CAN message for face detection

        # Define ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        # Detect eyes
        eyes = cascade_eye.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=6)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
            print("Eyes detected!")
            sendtoCAN(1, 2, "2")  # Send CAN message for eyes detection

        # Detect smiles
        smiles = cascade_smile.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
            print("Smile detected!")
            sendtoCAN(2, 3, "3")  # Send CAN message for smile detection

    # Detect cats
    cats = cascade_cat.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in cats:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        print("Cat detected!")
        sendtoCAN(3, 4, "4")  # Send CAN message for cat detection

    return image


# Set up the camera and start real-time detection
cap = cv2.VideoCapture(0)  # Connect to the camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera connected successfully")

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to connect to the camera")
        break

    # Detect faces, eyes, smiles, and cats
    result = detect_face_eyes_smile_cat(img)

    # Display the detection results
    cv2.imshow('Detection: Face, Eyes, Smile, Cat', result)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
