import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# 1. Cargar el modelo entrenado
model = load_model("mnist_cnn.h5")

# 2. Cargar la imagen
img_path = "Captura de pantalla 2025-08-11 090804.png"  # Cambia por tu ruta si es distinta
img = Image.open(img_path).convert("L")  # L = escala de grises

# 3. Redimensionar a 28x28
img = img.resize((28, 28))

# 4. Convertir a array de numpy
img_array = np.array(img)

# 5. Invertir colores si es necesario (MNIST es fondo negro y número blanco)
if img_array.mean() > 127:  # si es muy claro, invertimos
    img_array = 255 - img_array

# 6. Normalizar (0-1)
img_array = img_array / 255.0

# 7. Ajustar forma para que sea (1, 28, 28, 1)
img_array = img_array.reshape(1, 28, 28, 1)

# 8. Hacer predicción
pred = model.predict(img_array)
predicted_number = np.argmax(pred)

print(f"El modelo predice que es un: {predicted_number}")
print("Probabilidades por clase:", pred)
