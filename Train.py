import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp
import numpy as np

# Cargar los datos
data = pd.read_csv('facial_expressions.csv')
X = data.drop(columns='emotion').values
y = data['emotion'].values

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo Random Forest
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluar el modelo
accuracy = model.score(X_test, y_test)
print(f'Accuracy del modelo: {accuracy * 100:.2f}%')

# Preparar para el reconocimiento
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_data = []
            for landmark in face_landmarks.landmark:
                landmark_data.append([landmark.x, landmark.y, landmark.z])

            # Convertir a array y predecir la emoción
            landmark_array = np.array(landmark_data).flatten().reshape(1, -1)
            prediction = model.predict(landmark_array)
            cv2.putText(frame, f'Predicción: {prediction[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Dibujar la malla facial
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    cv2.imshow('Reconocimiento Facial', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
