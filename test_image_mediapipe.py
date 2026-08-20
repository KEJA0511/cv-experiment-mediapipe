import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


image_path = "test_image.jpg"


# model path
model_path = "hand_landmarker.task"


base_options = python.BaseOptions(
    model_asset_path=model_path
)


options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3
)


detector = vision.HandLandmarker.create_from_options(
    options
)


# read image
image = cv2.imread(image_path)

rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb
)


result = detector.detect(mp_image)


if result.hand_landmarks:

    print(
        "Detected hands:",
        len(result.hand_landmarks)
    )


    for hand_id, landmarks in enumerate(
        result.hand_landmarks
    ):

        print("\nHand", hand_id)

        for idx, lm in enumerate(landmarks):

            print(
                idx,
                lm.x,
                lm.y,
                lm.z
            )

else:

    print("No hand detected")