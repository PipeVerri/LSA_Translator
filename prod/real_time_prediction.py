import threading
import cv2
import numpy as np

from utils.video import camera_reader
import mediapipe as mp
from utils.landmarks.landmarks import Landmarks, nn_parser
from utils.tts import speak
import pandas as pd
import torch
from models import SimpleRNN
from collections import deque
import time

# Setup de los datos
signs = pd.read_csv("../data/LSA64/meta.csv")

# Setup de los landmarks
lm = Landmarks()

# Setup threading
capture_finished = threading.Event()

# Setup pytorch
model = SimpleRNN()
model.load_state_dict(torch.load("../Notebooks/inference/best_params.pth"))
model.eval()

# Configuración de sliding window
WINDOW_SIZE_SECONDS = 2.0  # Tamaño de la ventana en segundos
STRIDE_SECONDS = 0.2  # Tiempo entre predicciones


def generator_thread():
    with mp.solutions.holistic.Holistic(model_complexity=2, static_image_mode=False) as holistic:
        for frame in camera_reader(fps=12):
            hol_res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lm.add(hol_res.pose_landmarks, hol_res.left_hand_landmarks, hol_res.right_hand_landmarks)
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

    capture_finished.set()


def parser_thread():
    print("running")

    # Buffer para almacenar (timestamp, landmarks) de la ventana
    window_buffer = deque()
    last_prediction = None
    last_prediction_time = 0

    with torch.no_grad():
        for pose, left, right in lm.get_landmarks(continuous=True):
            current_time = time.time()

            # Parsear y agregar al buffer con timestamp
            x = nn_parser(pose, left, right)
            window_buffer.append((current_time, x))

            # Eliminar elementos antiguos fuera de la ventana
            cutoff_time = current_time - WINDOW_SIZE_SECONDS
            while window_buffer and window_buffer[0][0] < cutoff_time:
                window_buffer.popleft()

            # Solo procesar si han pasado STRIDE_SECONDS desde la última predicción
            # y tenemos suficientes datos en la ventana
            if (current_time - last_prediction_time) >= STRIDE_SECONDS and len(window_buffer) >= 5:
                # Preparar el batch con la secuencia actual
                sequence_data = [landmark for _, landmark in window_buffer]
                sequence = torch.tensor(np.array([sequence_data])).float()  # Shape: (1, seq_len, features)
                lengths = torch.tensor([len(sequence_data)])  # Longitud real de la secuencia

                # Forward pass
                logits = model.forward(sequence, lengths)

                # Obtener predicción
                probs = torch.softmax(logits[0], dim=0)
                max_prob, predicted_class = torch.max(probs, dim=0)

                max_prob = max_prob.item()
                predicted_class = predicted_class.item()

                # Solo hablar si la confianza es alta y es diferente a la última predicción
                if max_prob >= 0.95:
                    predicted_sign = signs.iloc[predicted_class]["Name"]

                    # Evitar repetir la misma predicción consecutivamente
                    if predicted_sign != last_prediction:
                        print(f"Predicción: {predicted_sign} (confianza: {max_prob:.2%})")
                        speak(predicted_sign)
                        last_prediction = predicted_sign
                        last_prediction_time = current_time

            if capture_finished.is_set():
                break

    print("finished")


thread1 = threading.Thread(target=generator_thread)
thread2 = threading.Thread(target=parser_thread)
thread1.start()
thread2.start()
thread1.join()
thread2.join()