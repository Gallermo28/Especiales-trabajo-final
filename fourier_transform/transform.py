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


def filtro_gaussiano(n, sigma):
    """
    Genera un filtro gaussiano 1D de tamaño n y desviación estándar sigma.
    """
    x = np.linspace(-3*sigma, 3*sigma, n)
    filtro = norm.pdf(x, 0, sigma)
    filtro /= np.sum(filtro)  # Normalizar para que la suma sea 1
    return filtro


def aplicar_filtro_gaussiano(espectro, sigma):
    """
    Aplica un filtro gaussiano sobre el espectro 1D convolucionando la magnitud.
    """
    n = len(espectro)
    filtro = filtro_gaussiano(n, sigma)
    espectro_abs = np.abs(espectro)
    espectro_filtrado = np.convolve(espectro_abs, filtro, mode='same')
    return espectro_filtrado


def mostrar_espectro(espectro, titulo="Espectro de frecuencias", sigma=None):
    n = espectro.shape[0]
    freq = np.arange(n // 2)

    if sigma is not None:
        magnitud = aplicar_filtro_gaussiano(espectro, sigma)
    else:
        magnitud = np.abs(espectro)

    magnitud = magnitud[:n // 2]

    plt.figure(figsize=(10, 4))
    plt.plot(freq, magnitud)
    plt.title(titulo)
    plt.xlabel('Frecuencia')
    plt.ylabel('Magnitud')
    plt.grid(True)
    plt.show()
