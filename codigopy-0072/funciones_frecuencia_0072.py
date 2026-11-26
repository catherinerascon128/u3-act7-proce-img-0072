# =====================================================
# ARCHIVO: funciones_frecuencia_0072.py
# PROPÓSITO: Funciones compartidas para procesamiento en frecuencia
# =====================================================
# catherine eileen huerta rason NC = 0072 
import os
import numpy as np
import cv2
from skimage.color import hsv2rgb, rgb2gray

def cargar_imagen():
    """Genera el círculo cromático HSV, lo convierte a RGB y devuelve su versión en escala de grises."""
    x, y = np.ogrid[-1:1:256j, -1:1:256j]
    angulo = np.arctan2(y, x)

    hsv = np.stack([
        (angulo + np.pi) / (2 * np.pi),
        np.ones_like(angulo),
        np.ones_like(angulo)
    ], axis=-1)

    imagen_rgb = hsv2rgb(hsv)
    # Convertimos a escala de grises (2D) para que funcione con FFT 2D y filtros Sobel
    imagen_gris = rgb2gray(imagen_rgb)
    return imagen_gris

def obtener_espectro(imagen):
    """Calcula la Transformada de Fourier y devuelve el espectro visualizable."""
    f_transform = np.fft.fft2(imagen)
    f_shift = np.fft.fftshift(f_transform)
    espectro = 20 * np.log(np.abs(f_shift) + 1)
    return espectro

def aplicar_filtro_frecuencia(imagen, kernel):
    """
    Aplica un kernel en el dominio de la frecuencia.
    Se añade un borde de relleno (padding) para evitar artefactos de convolución circular.
    """
    pad_size = 1
    imagen_padded = np.pad(imagen, pad_size, mode='reflect')
    rows, cols = imagen_padded.shape
    
    kernel_padded = np.zeros((rows, cols), dtype=np.float32)
    k_rows, k_cols = kernel.shape
    start_r = (rows - k_rows) // 2
    start_c = (cols - k_cols) // 2
    kernel_padded[start_r:start_r+k_rows, start_c:start_c+k_cols] = kernel
    
    kernel_padded_shifted = np.fft.ifftshift(kernel_padded)
    
    fft_img = np.fft.fft2(imagen_padded)
    fft_ker = np.fft.fft2(kernel_padded_shifted)
    
    resultado_fft = fft_img * fft_ker
    resultado_complejo = np.fft.ifft2(resultado_fft)
    resultado_real = np.real(resultado_complejo)
    
    return resultado_real[pad_size:-pad_size, pad_size:-pad_size]

def normalizar_visualizacion(imagen):
    """Convierte una respuesta de filtro (con valores negativos/positivos) a 0-255."""
    imagen_abs = np.abs(imagen)
    imagen_normalizada = cv2.normalize(
        imagen_abs, None, 0, 255, cv2.NORM_MINMAX
    )
    return imagen_normalizada.astype(np.uint8)

def guardar_grafica(figura, nombre_archivo):
    """Guarda una figura de matplotlib dentro de la carpeta resultados-img-0072."""
    carpeta = "resultados-img-0072"
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, nombre_archivo)
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"✅ Imagen guardada correctamente en: {ruta}")