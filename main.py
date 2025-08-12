import cv2
import mediapipe as mp
import os
import pyautogui

# Прибираємо попередження TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Ініціалізація MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Запуск камери
cap = cv2.VideoCapture(0)

# Лічильник для запобігання повторних натискань
gesture_cooldown = 0

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Перетворення в формат RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # Повертаємо зображення у формат BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Отримуємо координати великого і вказівного пальця
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                # Розрахунок дистанції між пальцями
                distance = ((thumb_tip.x - index_tip.x) ** 2 +
                            (thumb_tip.y - index_tip.y) ** 2) ** 0.5


                # Якщо пальці близько вказівний і великий(жест "щіпка") і cooldown закінчився
                if distance < 0.016 and gesture_cooldown == 0:
                    pyautogui.press('space')
                    #print("Distance:", distance)
                    gesture_cooldown = 60  # 60 кадрів затримка

        # Зменшуємо cooldown
        if gesture_cooldown > 0:
            gesture_cooldown -= 1

        #cv2.imshow('Gesture Control', image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC для виходу
            break

cap.release()
cv2.destroyAllWindows()
