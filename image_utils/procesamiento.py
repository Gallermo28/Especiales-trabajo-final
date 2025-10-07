# image_utils/procesamiento.py

from PIL import Image
import numpy as np

def elegir_archivo():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    ruta = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg")]
    )
    return ruta

def cargar_imagen(ruta):
    return Image.open(ruta)

def crear_imagen_gris_prueba():
    arr = np.zeros((100,100), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            arr[i,j] = int(255 * (i+j) / 200)
    img_gris = Image.fromarray(arr, mode='L')
    return img_gris

def procesar_imagen_segun_modo_real(img):
    modo_original = img.mode
    if modo_original == 'L':
        return img, True
    if modo_original == 'P':
        paleta = img.getpalette()
        if paleta:
            tripletas = np.array(paleta).reshape(-1,3)
            es_paleta_gris = all((r==g==b) for r,g,b in tripletas)
            if es_paleta_gris:
                return img.convert('L'), True
            else:
                return img.convert('RGB'), False
        else:
            return img.convert('RGB'), False
    if modo_original == 'RGB':
        return img, False
    return img.convert('RGB'), False

def separar_canales_rgb(img):
    r, g, b = img.split()
    return np.array(r), np.array(g), np.array(b)

def a_senal_1d(img_o_np):
    if not isinstance(img_o_np, np.ndarray):
        arr = np.array(img_o_np)
    else:
        arr = img_o_np
    flattened = arr.flatten()
    normalized = 2 * (flattened / 255.0) - 1
    return normalized
