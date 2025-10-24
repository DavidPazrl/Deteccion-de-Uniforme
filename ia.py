import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Datos generales
MODEL_NAME = "modelo_prendas.h5"
LABELS = ["Camisa", "Polo Azul"]
IMG_SIZE = (224, 224)

if not os.path.exists(MODEL_NAME):
    print(f"No se encontro el modelo '{MODEL_NAME}' en la carpeta actual.")
    exit()
else:
    print(f"Modelo '{MODEL_NAME}' encontrado correctamente.")

# Cargamos el modelo
try:
    model = load_model(MODEL_NAME)
    print("Modelo cargado correctamente.\n")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()



# Cargamos la clasificacion
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Camara
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("No se pudo acceder a la camara.")
    exit()

cv2.namedWindow("Deteccion de Prendas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Deteccion de Prendas", 900, 700)

# Deteccion en vivo
while True:
    ret, frame = cam.read()
    if not ret:
        print("No se pudo leer el frame de la camara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        y_exp = max(0, y - int(h * 0.2)) 
        h_exp = int(h * 2.2)
        y_exp = max(0, y_exp)
        if y_exp + h_exp > frame.shape[0]:
            h_exp = frame.shape[0] - y_exp

        # ROI expandido
        roi_color = frame[y_exp:y_exp + h_exp, x:x + w]
        if roi_color.size == 0:
            continue

        roi_resized = cv2.resize(roi_color, IMG_SIZE)
        roi_resized = roi_resized.astype("float32") / 255.0
        roi_resized = np.expand_dims(roi_resized, axis=0)

        # Prediccion
        pred = model.predict(roi_resized)
        conf_camisa = pred[0][0] * 100
        conf_polo = pred[0][1] * 100
        clase_idx = np.argmax(pred)
        clase = LABELS[clase_idx]

        # Colores del recuadro
        color_recuadro = (0, 255, 0) if clase == "Camisa" else (255, 0, 0)

        # Dibujar el recuadro ampliado
        cv2.rectangle(frame, (x, y_exp), (x + w, y_exp + h_exp), color_recuadro, 3)

        # Mostrar texto con prediccion
        cv2.putText(frame, f"{clase} ({max(conf_camisa, conf_polo):.1f}%)",
                    (x, y_exp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_recuadro, 2)

    # Mostrar ventana
    cv2.imshow("Deteccion de Prendas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar
cam.release()
cv2.destroyAllWindows()
print("Camara cerrada correctamente.")