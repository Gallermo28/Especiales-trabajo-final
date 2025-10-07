from image_utils.procesamiento import elegir_archivo, cargar_imagen, procesar_imagen_segun_modo_real, separar_canales_rgb, a_senal_1d
from fourier_transform.transform import aplicar_fft, aplicar_ifft, aplicar_fft_2d, aplicar_ifft_2d
import numpy as np
import matplotlib.pyplot as plt

def prueba_fft_completo():
    ruta = elegir_archivo()
    if not ruta:
        print("No se seleccionó ninguna imagen.")
        return

    img = cargar_imagen(ruta)
    img_proc, es_gris = procesar_imagen_segun_modo_real(img)

    # Para imagen color: extraer fila media para la señal 1D
    if es_gris:
        arr = np.array(img_proc)
        senal = a_senal_1d(arr[arr.shape[0]//2, :])  # fila media
    else:
        r, g, b = separar_canales_rgb(img_proc)
        senal = a_senal_1d(r[r.shape[0]//2, :])  # fila media canal rojo

    # Señal 1D y espectro 1D
    espectro_1d = aplicar_fft(senal)
    senal_rec_1d = aplicar_ifft(espectro_1d)

    # FFT 2D e IFFT 2D para imagen completa
    if es_gris:
        fft_2d = aplicar_fft_2d(arr)
        img_rec_2d = aplicar_ifft_2d(fft_2d)
    else:
        r_fft = aplicar_fft_2d(r)
        g_fft = aplicar_fft_2d(g)
        b_fft = aplicar_fft_2d(b)

        r_rec = aplicar_ifft_2d(r_fft)
        g_rec = aplicar_ifft_2d(g_fft)
        b_rec = aplicar_ifft_2d(b_fft)

        def normalizar(x):
            x = np.clip(x, 0, 255)
            return x.astype(np.uint8)

        img_rec_2d = np.stack((normalizar(r_rec),
                               normalizar(g_rec),
                               normalizar(b_rec)), axis=-1)

    # Graficar resultados
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))

    # Señal original 1D
    axs[0,0].plot(senal, color='blue')
    axs[0,0].set_title('Señal 1D (fila media)')
    axs[0,0].set_xlabel('Índice')
    axs[0,0].set_ylabel('Amplitud')
    axs[0,0].grid(True)

    # Espectro 1D
    n = len(senal)
    freq = np.arange(n//2)
    magnitud = np.abs(espectro_1d[:n//2])
    axs[0,1].plot(freq, magnitud, color='red')
    axs[0,1].set_title('Espectro 1D (mitad positiva)')
    axs[0,1].set_xlabel('Frecuencia')
    axs[0,1].set_ylabel('Magnitud')
    axs[0,1].grid(True)

    # Señal reconstruida 1D
    axs[0,2].plot(senal_rec_1d, color='green')
    axs[0,2].set_title('Señal reconstruida (IFFT)')
    axs[0,2].set_xlabel('Índice')
    axs[0,2].set_ylabel('Amplitud')
    axs[0,2].grid(True)

    # Imagen Original
    axs[1,0].imshow(np.array(img_proc), cmap='gray' if es_gris else None)
    axs[1,0].set_title('Imagen Original')
    axs[1,0].axis('off')

    # Imagen Reconstruida FFT 2D + IFFT 2D
    axs[1,1].imshow(img_rec_2d, cmap='gray' if es_gris else None)
    axs[1,1].set_title('Imagen Reconstruida FFT 2D + IFFT 2D')
    axs[1,1].axis('off')

    axs[1,2].axis('off')  # Celda vacía para balance

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    prueba_fft_completo()
