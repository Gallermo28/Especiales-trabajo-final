# test_fourier.py

from image_utils.procesamiento import elegir_archivo, cargar_imagen, procesar_imagen_segun_modo_real, separar_canales_rgb, a_senal_1d
from fourier_transform.transform import aplicar_fft, aplicar_ifft
import numpy as np
import matplotlib.pyplot as plt

def reconstruir_imagen_color(img_proc):
    # Separar canales
    r, g, b = separar_canales_rgb(img_proc)

    # Aplicar FFT e IFFT a cada canal
    r_senal = a_senal_1d(r)
    g_senal = a_senal_1d(g)
    b_senal = a_senal_1d(b)

    r_fft = aplicar_fft(r_senal)
    g_fft = aplicar_fft(g_senal)
    b_fft = aplicar_fft(b_senal)

    r_rec = aplicar_ifft(r_fft)
    g_rec = aplicar_ifft(g_fft)
    b_rec = aplicar_ifft(b_fft)

    # Reconstruir imagen tridimensional
    # Normalizar de vuelta a [0,255] para formar imagen
    def normalizar(arr):
        arr = (arr + 1) / 2  # De [-1,1] a [0,1]
        arr = np.clip(arr, 0, 1)
        return (arr * 255).astype(np.uint8)

    img_rgb_rec = np.stack((normalizar(r_rec),
                            normalizar(g_rec),
                            normalizar(b_rec)), axis=-1)
    return img_rgb_rec

def prueba_fft_color():
    ruta = elegir_archivo()
    if not ruta:
        print("No se seleccionó ninguna imagen.")
        return

    img = cargar_imagen(ruta)
    img_proc, es_gris = procesar_imagen_segun_modo_real(img)

    if es_gris:
        print("Imagen en escala de grises, para reconstrucción exacta usa imagen color.")
        return

    r, g, b = separar_canales_rgb(img_proc)
    senal_r = a_senal_1d(r)

    espectro_r = aplicar_fft(senal_r)
    senal_rec_r = aplicar_ifft(espectro_r)

    img_rec = reconstruir_imagen_color(img_proc)

    # Mostrar imágenes (original y reconstruida)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(np.array(img_proc))
    axs[0].set_title("Imagen Original")
    axs[0].axis('off')

    axs[1].imshow(img_rec)
    axs[1].set_title("Imagen Reconstruida FFT + IFFT")
    axs[1].axis('off')

    plt.show()

if __name__ == '__main__':
    prueba_fft_color()
