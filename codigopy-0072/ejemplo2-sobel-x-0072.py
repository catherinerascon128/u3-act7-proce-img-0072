# =====================================================
# EJEMPLO 2: SOBEL X EN EL DOMINIO DE LA FRECUENCIA
# =====================================================
import numpy as np
import matplotlib.pyplot as plt
from funciones_frecuencia_0072 import (
    cargar_imagen,
    obtener_espectro,
    aplicar_filtro_frecuencia,
    normalizar_visualizacion,
    guardar_grafica
)

# 1. Cargar imagen original
imagen = cargar_imagen()

# 2. Crear kernel Sobel X (Detecta cambios horizontales -> resalta bordes verticales)
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# 3. Aplicar Sobel X en el dominio de la frecuencia
resultado_x = aplicar_filtro_frecuencia(imagen, sobel_x)

# 4. Calcular el espectro de la respuesta filtrada
espectro_x = obtener_espectro(resultado_x)

# 5. Normalizar la respuesta para poder visualizarla (0 a 255)
imagen_x_normalizada = normalizar_visualizacion(resultado_x)

# 6. Mostrar resultados
figura, ejes = plt.subplots(1, 3, figsize=(15, 5))

ejes[0].imshow(imagen, cmap="gray")
ejes[0].set_title("Imagen Original")
ejes[0].axis("off")

ejes[1].imshow(espectro_x, cmap="gray")
ejes[1].set_title("Espectro después de Sobel X")
ejes[1].axis("off")

ejes[2].imshow(imagen_x_normalizada, cmap="gray")
ejes[2].set_title("Respuesta Sobel X (Bordes Verticales)")
ejes[2].axis("off")

plt.tight_layout()
guardar_grafica(figura, "02_sobel_x.png")
plt.show()