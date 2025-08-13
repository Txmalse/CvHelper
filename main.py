import pyautogui
import cv2
import mediapipe as mp
from mediapipe.tasks import python

gesture_cooldown = 0
gesture_name = None

MODEL_PATH = 'models/gesture_recognizer.task'

def main():
    global gesture_name, gesture_cooldown

    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Callback для обробки жестів
    def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        global gesture_name
        if result.gestures:
            gesture = result.gestures[0][0]
            gesture_name = gesture.category_name

    # Налаштування розпізнавача з прискореними параметрами
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.7
    )

    # Камера в FHD або нижчій роздільності для прискорення
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    with GestureRecognizer.create_from_options(options) as recognizer:
        timestamp = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            recognizer.recognize_async(mp_image, timestamp)
            timestamp += 1

            # Дія по жесту
            if gesture_name == 'Closed_Fist' and gesture_cooldown == 0:
                pyautogui.press('space')
                gesture_cooldown = 60
                gesture_name = None

            if gesture_cooldown > 0:
                gesture_cooldown -= 1

            # Відображення жесту
            # cv2.putText(frame, f"Gesture: {gesture_name}", (10, 40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #
            # cv2.imshow("Gesture Recognition Fast", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
