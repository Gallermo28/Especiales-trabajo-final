from image_utils.procesamiento import elegir_archivo, cargar_imagen, procesar_imagen_segun_modo_real, separar_canales_rgb, a_senal_1d
from fourier_transform.transform import aplicar_fft, aplicar_ifft, aplicar_fft_2d, aplicar_ifft_2d, aplicar_filtro_gaussiano
from ventana_graficos import VentanaGraficos
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog


def espectro_a_sonido(espectro, duracion=2.0, fs=44100, fmin=200, fmax=2000):
    magnitud = np.abs(espectro)
    n = len(magnitud)
    t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)
    freqs = np.linspace(fmin, fmax, n)

    señal = np.zeros_like(t)
    for i in range(n):
        amplitud = magnitud[i]
        freq = freqs[i]
        señal += amplitud * np.sin(2 * np.pi * freq * t)

    señal /= np.max(np.abs(señal))
    return señal.astype(np.float32)


def grafico_senal(ax, senal):
    ax.plot(senal, color='blue')
    ax.set_title("Señal 1D Original (Fila Media)")
    ax.set_xlabel("Índice de Pixel")
    ax.set_ylabel("Amplitud Normalizada")
    ax.grid(True)


def grafico_espectro(ax, espectro):
    n = len(espectro)
    freq = np.arange(n // 2)
    magnitud = np.abs(espectro[:n // 2])
    ax.plot(freq, magnitud, color='red')
    ax.set_title("Espectro FFT 1D (Mitad Positiva)")
    ax.set_xlabel("Frecuencia Spatial")
    ax.set_ylabel("Magnitud")
    ax.grid(True)

def grafico_espectro_filtrado(ax, espectro, sigma=3, tam_filtro=11):
    n = len(espectro)
    freq = np.arange(n // 2)
    magnitud_filtrada = aplicar_filtro_gaussiano(espectro, sigma, tam_filtro)[:n // 2]
    ax.plot(freq, magnitud_filtrada, color='orange')
    ax.set_title("Espectro Filtrado Gaussiano")
    ax.set_xlabel("Frecuencia Spatial")
    ax.set_ylabel("Magnitud")
    ax.grid(True)


def grafico_senal_rec(ax, senal_rec):
    ax.plot(senal_rec, color='green')
    ax.set_title("Señal 1D Reconstruida (IFFT)")
    ax.set_xlabel("Índice de Pixel")
    ax.set_ylabel("Amplitud Normalizada")
    ax.grid(True)


def grafico_imagen(ax, img_proc, es_gris):
    ax.imshow(np.array(img_proc), cmap='gray' if es_gris else None)
    ax.set_title("Imagen Original")
    ax.axis('off')


def grafico_imagen_rec(ax, imagen_rec, es_gris, titulo_imagen):
    ax.imshow(imagen_rec, cmap='gray' if es_gris else None)
    ax.set_title(titulo_imagen + " Reconstruida")
    ax.axis('off')


def preparar_datos(img_proc, es_gris):
    if es_gris:
        arr = np.array(img_proc)
        senal = a_senal_1d(arr[arr.shape[0] // 2, :])
        imagen_rec = aplicar_ifft_2d(aplicar_fft_2d(arr))
        titulo_imagen = "Imagen en Escala de Grises"
    else:
        r, g, b = separar_canales_rgb(img_proc)
        senal = a_senal_1d(r[r.shape[0] // 2, :])
        r_rec = aplicar_ifft_2d(aplicar_fft_2d(r))
        g_rec = aplicar_ifft_2d(aplicar_fft_2d(g))
        b_rec = aplicar_ifft_2d(aplicar_fft_2d(b))

        def normalizar(x):
            x = np.clip(x, 0, 255)
            return x.astype(np.uint8)

        imagen_rec = np.stack((normalizar(r_rec),
                               normalizar(g_rec),
                               normalizar(b_rec)), axis=-1)
        titulo_imagen = "Imagen a Color"

    espectro = aplicar_fft(senal)
    esp_filtrado = aplicar_filtro_gaussiano(espectro, sigma=3, tam_filtro=11)
    senal_rec = aplicar_ifft(espectro)

    lista_funciones_parejas = [
        (lambda ax: grafico_senal(ax, senal), lambda ax: grafico_senal_rec(ax, senal_rec)),
        (lambda ax: grafico_espectro(ax, espectro), lambda ax: grafico_espectro_filtrado(ax, espectro, sigma=3, tam_filtro=11)),
        (lambda ax: grafico_imagen(ax, img_proc, es_gris), lambda ax: grafico_imagen_rec(ax, imagen_rec, es_gris, titulo_imagen))
    ]

    audio_espectro = espectro_a_sonido(espectro)
    audio_filtrado = espectro_a_sonido(esp_filtrado)

    lista_audios = [
        (None, None),
        (audio_espectro, audio_filtrado),
        (None, None)
    ]

    return lista_funciones_parejas, lista_audios


if __name__ == "__main__":
    opcion = input("¿Quieres usar la imagen de prueba gris? (s/n): ").strip().lower()
    if opcion == 's':
        from image_utils.procesamiento import crear_imagen_gris_prueba
        img = crear_imagen_gris_prueba()
        es_gris = True
    else:
        ruta = elegir_archivo()
        if not ruta:
            print("No se seleccionó ninguna imagen.")
            exit()
        img = cargar_imagen(ruta)
        img, es_gris = procesar_imagen_segun_modo_real(img)


    lista_funciones_parejas, lista_audios = preparar_datos(img, es_gris)

    ventana = VentanaGraficos(lista_funciones_parejas, lista_audios)

