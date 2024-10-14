import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)

FACE_CONNECTIONS = mpMallaFacial.FACEMESH_TESSELATION

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = MallaFacial.process(frameRGB)

    if resultados.multi_face_landmarks:
        for rostros in resultados.multi_face_landmarks:
            mpDibujo.draw_landmarks(frame, rostros, FACE_CONNECTIONS, ConfDibu, ConfDibu)

            # Obtener puntos clave para emociones
            x1, y1 = int(rostros.landmark[65].x * frame.shape[1]), int(rostros.landmark[65].y * frame.shape[0])  # Cejas
            x2, y2 = int(rostros.landmark[158].x * frame.shape[1]), int(rostros.landmark[158].y * frame.shape[0])  # Cejas
            x3, y3 = int(rostros.landmark[78].x * frame.shape[1]), int(rostros.landmark[78].y * frame.shape[0])  # Boca
            x4, y4 = int(rostros.landmark[308].x * frame.shape[1]), int(rostros.landmark[308].y * frame.shape[0])  # Boca

            # Calcular distancias
            distancia_cejas = math.hypot(x2 - x1, y2 - y1)
            distancia_boca = math.hypot(x4 - x3, y4 - y3)

            # Reconocer emociones
            if distancia_cejas < 20 and distancia_boca > 80:
                cv2.putText(frame, "Persona Feliz", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif distancia_cejas > 40 and distancia_boca < 50:
                cv2.putText(frame, "Persona Triste", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif distancia_cejas < 20 and distancia_boca < 50:
                cv2.putText(frame, "Persona Enojada", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif distancia_cejas > 40 and distancia_boca > 80:
                cv2.putText(frame, "Persona Asombrada", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    t = cv2.waitKey(1)

    if t == 27:  # Presiona ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
