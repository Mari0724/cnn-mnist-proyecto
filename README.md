# 🧠 CNN Proyecto - Clasificador de dígitos manuscritos (MNIST)

Este proyecto implementa una **red neuronal convolucional (CNN)** para reconocer dígitos manuscritos utilizando el dataset **MNIST**.  

Incluye:
- **Entrenamiento del modelo** (`main.py`)
- **Predicción de imágenes externas** (`probar_imagen.py`)
- **Modelo entrenado** (`mnist_cnn.h5`)
- **Imágenes de prueba** utilizadas durante el desarrollo

---

## 🚀 Tecnologías utilizadas
- Python 3
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- Matplotlib

---

## 📂 Estructura del proyecto
```

CNN\_PROYECTO/
│── venv/                 # Entorno virtual (no se sube a GitHub normalmente)
│── main.py               # Entrena la CNN y guarda el modelo
│── probar\_imagen.py      # Prueba el modelo con imágenes externas
│── mnist\_cnn.h5          # Modelo entrenado
│── numero\_prueba.png     # Ejemplo de número escrito a mano
│── Captura.png           # Imagen usada para pruebas
│── Captura de pantalla...png # Otra imagen de prueba

````

---

## 🏋️‍♂️ Entrenamiento
Ejecuta el entrenamiento y guarda el modelo:
```bash
python main.py
````

Esto entrenará la CNN en el dataset MNIST y generará el archivo `mnist_cnn.h5`.

---

## 🔍 Probar con una imagen

Ejecuta el script de prueba con una imagen externa:

```bash
python probar_imagen.py
```

Ejemplo de salida:

```
El modelo predice que es un: 7
Probabilidades por clase: [[0.01 0.00 0.03 0.00 0.94 0.00 0.01 0.01 0.00 0.00]]
```

---

## 📊 Resultados

La red alcanza una **precisión >98%** en el dataset de prueba MNIST.
Además, puede predecir imágenes externas de números manuscritos con buena exactitud.

---

## 📌 Notas

* Este proyecto fue creado con fines de práctica para aprender CNNs en TensorFlow/Keras.
* Puedes ampliar el proyecto para reconocer letras (EMNIST) o integrarlo en una app interactiva.
