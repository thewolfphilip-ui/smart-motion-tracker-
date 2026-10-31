import cv2
import os
import mediapipe as mp
from playsound3 import playsound
import csv
from dotenv import load_dotenv

load_dotenv()
File_path= os.getenv("File_path")
gestures = []

with open("gesture_data.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)

    for row in reader:
        gesture_name = row[0]
        values = [float(x) for x in row[1:]]
        gestures.append((values, gesture_name))

print("Loaded", len(gestures), "gestures")



BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(
    #ADD YOUR OWN FILEPATH HERE
        model_asset_path=r"C:\Users\thewo\PycharmProjects\gesture_recognition.py\gesture_recognizer.task"
    ),
    running_mode=VisionRunningMode.IMAGE
)

recognizer = GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("this is you", frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = recognizer.recognize(mp_image)



    # Your custom gestures
    if result.hand_landmarks:

        webcam_values = []

        for landmark in result.hand_landmarks[0]:
            webcam_values.append(landmark.x)
            webcam_values.append(landmark.y)
            webcam_values.append(landmark.z)

        for csv_values, gesture_name in gestures:

            if len(csv_values) != len(webcam_values):
                continue

            difference = 0

            for i in range(len(webcam_values)):
                difference += abs(
                    webcam_values[i] - csv_values[i]
                )

            difference /= len(webcam_values)

            if difference < 0.05:
                print(gesture_name)
                if gesture_name=="NICE":
                    playsound("nice.mp3")
                if gesture_name=="hell yeah":
                    playsound("yeah.mp3")
                break


    if cv2.waitKey(2) & 0xFF == ord("e"):
        break

cap.release()
cv2.destroyAllWindows()
