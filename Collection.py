import cv2
import mediapipe as mp
import pandas as pd
import time

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Inicializar Face Mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Lista para almacenar los datos
data = []

# Definir emociones
emotions = ["Feliz", "Triste", "Enojado", "Asombrado"]

# Preguntar al usuario qué emoción quiere capturar
for emotion in emotions:
    print(f"Coloca tu cara en la cámara y haz la expresión: {emotion}")
    time.sleep(5)  # Espera 5 segundos para que el usuario se prepare

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Inicializar la lista para la malla facial
                landmark_data = []
                for landmark in face_landmarks.landmark:
                    landmark_data.append([landmark.x, landmark.y, landmark.z])

                # Aplanar la lista de puntos de referencia para que quede en un solo nivel
                flat_landmark_data = [coord for landmark in landmark_data for coord in landmark]

                # Agregar la emoción como etiqueta
                flat_landmark_data.append(emotion)

                # Agregar los datos a la lista
                data.append(flat_landmark_data)

                # Dibujar la malla facial
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Guardar los datos en un DataFrame
num_landmarks = 468  # Número de puntos de referencia de la malla facial
df = pd.DataFrame(data, columns=[f'landmark_{i}_{axis}' for i in range(num_landmarks) for axis in ['x', 'y', 'z']] + [
    'emotion'])
df.to_csv('facial_expressions.csv', index=False)
