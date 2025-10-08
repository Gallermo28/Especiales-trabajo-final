import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def aplicar_fft(senal_1d):
    return np.fft.fft(senal_1d)


def aplicar_ifft(espectro):
    return np.real(np.fft.ifft(espectro))


def aplicar_fft_2d(matriz):
    return np.fft.fft2(matriz)


def aplicar_ifft_2d(espectro):
    return np.real(np.fft.ifft2(espectro))


def filtro_gaussiano(tam, sigma):
    x = np.linspace(-3*sigma, 3*sigma, tam)
    filtro = norm.pdf(x, 0, sigma)
    filtro /= np.sum(filtro)
    return filtro

def aplicar_filtro_gaussiano(espectro, sigma=2, tam_filtro=11):
    filtro = filtro_gaussiano(tam_filtro, sigma)
    espectro_abs = np.abs(espectro)
    espectro_filtrado = np.convolve(espectro_abs, filtro, mode='same')
    return espectro_filtrado

def mostrar_comparacion_filtro(espectro, sigma=2, tam_filtro=11):
    n = len(espectro)
    freq = np.arange(n // 2)
    magnitud_original = np.abs(espectro)[:n // 2]
    magnitud_filtrada = aplicar_filtro_gaussiano(espectro, sigma, tam_filtro)[:n // 2]

    plt.figure(figsize=(10, 5))
    plt.plot(freq, magnitud_original, label='Espectro Original', alpha=0.7)
    plt.plot(freq, magnitud_filtrada, label='Espectro Filtrado - Gaussiano', linewidth=2)
    plt.title('Comparación espectro original y filtrado')
    plt.xlabel('Frecuencia')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid(True)
    plt.show()