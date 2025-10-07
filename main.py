from image_utils.procesamiento import elegir_archivo, cargar_imagen, procesar_imagen_segun_modo_real, separar_canales_rgb, a_senal_1d
from fourier_transform.transform import aplicar_fft, aplicar_ifft, aplicar_fft_2d, aplicar_ifft_2d
import numpy as np
import matplotlib.pyplot as plt

def mostrar_resultados_fft_imagen(img_proc, es_gris):
    if es_gris:
        arr = np.array(img_proc)
        senal = a_senal_1d(arr[arr.shape[0]//2, :])
        imagen_rec = aplicar_ifft_2d(aplicar_fft_2d(arr))
        titulo_imagen = "Imagen en Escala de Grises"
    else:
        r, g, b = separar_canales_rgb(img_proc)
        senal = a_senal_1d(r[r.shape[0]//2, :])
        r_rec = aplicar_ifft_2d(aplicar_fft_2d(r))
        g_rec = aplicar_ifft_2d(aplicar_fft_2d(g))
        b_rec = aplicar_ifft_2d(aplicar_fft_2d(b))

        def normalizar(x):
            x = np.clip(x, 0, 255)
            return x.astype(np.uint8)
        imagen_rec = np.stack((normalizar(r_rec), normalizar(g_rec), normalizar(b_rec)), axis=-1)
        titulo_imagen = "Imagen a Color"

    espectro = aplicar_fft(senal)
    senal_rec = aplicar_ifft(espectro)

    n = len(senal)
    freq = np.arange(n//2)
    magnitud = np.abs(espectro[:n//2])

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Señal 1D original (fila media)
    axs[0,0].plot(senal, color='blue')
    axs[0,0].set_title("Señal 1D Original (Fila Media)")
    axs[0,0].set_xlabel("Índice de Pixel")
    axs[0,0].set_ylabel("Amplitud Normalizada")
    axs[0,0].grid(True)

    # Espectro 1D de la señal
    axs[0,1].plot(freq, magnitud, color='red')
    axs[0,1].set_title("Espectro FFT 1D (Mitad Positiva)")
    axs[0,1].set_xlabel("Frecuencia Spatial")
    axs[0,1].set_ylabel("Magnitud")
    axs[0,1].grid(True)

    # Señal 1D reconstruida (IFFT)
    axs[0,2].plot(senal_rec, color='green')
    axs[0,2].set_title("Señal 1D Reconstruida (IFFT)")
    axs[0,2].set_xlabel("Índice de Pixel")
    axs[0,2].set_ylabel("Amplitud Normalizada")
    axs[0,2].grid(True)

    # Imagen Original - salida visual 2D o 3D
    axs[1,0].imshow(np.array(img_proc), cmap='gray' if es_gris else None)
    axs[1,0].set_title("Imagen Original")
    axs[1,0].axis('off')

    # Espacio vacío para balance
    axs[1,1].axis('off')

    # Imagen Reconstruida usando FFT 2D e IFFT 2D
    axs[1,2].imshow(imagen_rec, cmap='gray' if es_gris else None)
    axs[1,2].set_title(titulo_imagen + " Reconstruida")
    axs[1,2].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    opcion = input("¿Quieres usar la imagen de prueba gris? (s/n): ").strip().lower()
    if opcion == 's':
        from image_utils.procesamiento import crear_imagen_gris_prueba
        img = crear_imagen_gris_prueba()
        es_gris = True
    else:
        ruta = elegir_archivo()
        if not ruta:
            print("No se seleccionó ninguna imagen.")
            return
        img = cargar_imagen(ruta)
        img, es_gris = procesar_imagen_segun_modo_real(img)

    mostrar_resultados_fft_imagen(img, es_gris)

if __name__ == "__main__":
    main()
