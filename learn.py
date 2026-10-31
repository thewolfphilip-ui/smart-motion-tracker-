import csv
import cv2
import mediapipe as mp
from pathlib import Path


DATA_PATH = Path(__file__).parent / "gesture_data.csv"
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

GESTURES = {
    "1": "NICE",
    "2": "PEACE",
    "3": "vex",
}



BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

if not DATA_PATH.exists():

    header = ["gesture"]

    for i in range(21):
        header += [
            f"x{i}",
            f"y{i}",
            f"z{i}"
        ]

    with open(DATA_PATH, "w", newline="") as file:
        writer =csv.writer(file)
        writer.writerow(header)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started.")
print("Press E to quit.")

timestamp = 0

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)


    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp += 1


    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )



    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        print("\n-----------------------------")
        print("GOT YOUR HAND ")
        print("-----------------------------")

        for i, landmark in enumerate(hand):
            print(
                "Point", i,
                ": x =", round(landmark.x, 4),
                "y =", round(landmark.y, 4),
                "z =", round(landmark.z, 4)
            )


            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                5,
                (5, 125, 125),
                -1
            )


        connections = [
            (0,1), (1,2), (2,3), (3,4),
            (0,5), (5,6), (6,7), (7,8),
            (5,9), (9,10), (10,11), (11,12),
            (9,13), (13,14), (14,15), (15,16),
            (13,17), (17,18), (18,19), (19,20),
            (0,17)
        ]

        for start, end in connections:

            x1 = int(hand[start].x * frame.shape[1])
            y1 = int(hand[start].y * frame.shape[0])

            x2 = int(hand[end].x * frame.shape[1])
            y2 = int(hand[end].y * frame.shape[0])

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 250, 0),
                2
            )

    else:

        cv2.putText(
            frame,
            "where's your hand at ",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("Gesture Training", frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("e"):
        break

    if chr(key) in GESTURES:
            current_gesture = GESTURES[chr(key)]
            print(f"Now recording: {current_gesture}")

    elif key == 32:  # SPACE = save sample

            if current_gesture is None:
                print("Select a gesture first!")
                continue

            if not result.hand_landmarks:
                print("No hand detected!")
                continue

            hand = result.hand_landmarks[0]

            row = [current_gesture]

            for landmark in hand:
                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)

            with open(DATA_PATH, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)

            print(f"Saved: {current_gesture}")

            if key == 32:



             if len(result.hand_landmarks) == 0:
                print("No hand detected!")
                continue

            hand = result.hand_landmarks[0]

            row = [current_gesture]

            for landmark in hand:
                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)

            with open(DATA_PATH, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)
                print("Saved:", current_gesture)


    else:

                    print(
                "No hand detected - sample NOT saved."
                    )

    if cv2.waitKey(1) & 0xFF == ord("e"):
        break


cap.release()
cv2.destroyAllWindows()
landmarker.close()
