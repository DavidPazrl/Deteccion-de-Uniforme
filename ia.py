import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === CONFIGURACIÓN ===
MODEL_NAME = "modelo_prendas.h5"
LABELS = ["Camisa", "Polo Azul"]
IMG_SIZE = (224, 224)

# === VERIFICAR MODELO ===
print("📂 Directorio actual:", os.getcwd())
print("📄 Archivos en el directorio:", os.listdir())

if not os.path.exists(MODEL_NAME):
    print(f"❌ No se encontró el modelo '{MODEL_NAME}' en la carpeta actual.")
    exit()
else:
    print(f"✅ Modelo '{MODEL_NAME}' encontrado correctamente.")

# === CARGAR MODELO ===
try:
    model = load_model(MODEL_NAME)
    print("✅ Modelo cargado correctamente.\n")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    exit()

# === CARGAR CLASIFICADOR FACIAL ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === INICIAR CÁMARA ===
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

print("🎥 Cámara iniciada. Presiona 'q' para salir.")

cv2.namedWindow("🧠 Detección de Prendas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("🧠 Detección de Prendas", 900, 700)

# === BUCLE DE DETECCIÓN EN VIVO ===
while True:
    ret, frame = cam.read()
    if not ret:
        print("⚠️ No se pudo leer el frame de la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        # Expandir el cuadro hacia abajo (para incluir torso)
        y_exp = max(0, y - int(h * 0.2))  # un poco hacia arriba
        h_exp = int(h * 2.2)  # más alto para incluir parte del torso

        # Asegurar que no se salga de los límites
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

        # Predicción
        pred = model.predict(roi_resized)
        conf_camisa = pred[0][0] * 100
        conf_polo = pred[0][1] * 100
        clase_idx = np.argmax(pred)
        clase = LABELS[clase_idx]

        # Colores del recuadro
        color_recuadro = (0, 255, 0) if clase == "Camisa" else (255, 0, 0)

        # Dibujar el recuadro ampliado
        cv2.rectangle(frame, (x, y_exp), (x + w, y_exp + h_exp), color_recuadro, 3)

        # Mostrar texto con predicción
        cv2.putText(frame, f"{clase} ({max(conf_camisa, conf_polo):.1f}%)",
                    (x, y_exp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_recuadro, 2)

    # Mostrar ventana
    cv2.imshow("🧠 Detección de Prendas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CERRAR ===
cam.release()
cv2.destroyAllWindows()
print("👋 Cámara cerrada correctamente.")