# test_image_utils.py

from image_utils.procesamiento import elegir_archivo, cargar_imagen, a_escala_grises, separar_canales_rgb, a_senal_1d
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def crear_imagen_gris_prueba():
    arr = np.zeros((100,100), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            arr[i,j] = int(255 * (i+j) / 200)  # degradado gris

    img_gris = Image.fromarray(arr, mode='L')
    img_gris.save('imagen_gris_prueba.png')
    print("Imagen de prueba en escala de grises guardada como 'imagen_gris_prueba.png'")
    return img_gris

def procesar_imagen_según_modo_real(img):
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

def mostrar_imagen(imagen_np, es_gris=False, titulo="Imagen"):
    if es_gris:
        plt.imshow(imagen_np, cmap='gray')
    else:
        plt.imshow(imagen_np)
    plt.title(titulo)
    plt.axis('off')
    plt.show()

def prueba():
    opcion = input("¿Usar imagen de prueba gris creada automáticamente? (s/n): ").strip().lower()
    if opcion == 's':
        img = crear_imagen_gris_prueba()
    else:
        ruta = elegir_archivo()
        if not ruta:
            print("No se seleccionó ninguna imagen.")
            return
        img = cargar_imagen(ruta)

    print(f"Imagen cargada: tamaño={img.size}, modo={img.mode}")

    img_proc, es_gris = procesar_imagen_según_modo_real(img)
    print(f"Modo procesado: {img_proc.mode}, es gris: {es_gris}")

    arr = np.array(img_proc)

    if es_gris:
        mostrar_imagen(arr, es_gris=True, titulo="Imagen Gris Procesada")
        senal = a_senal_1d(arr)
        print("Ejemplo de señal gris 1D:", senal[:10])
    else:
        mostrar_imagen(arr, es_gris=False, titulo="Imagen Color Procesada")
        r, g, b = separar_canales_rgb(img_proc)
        senal_r = a_senal_1d(r)
        print("Ejemplo de señal canal rojo 1D:", senal_r[:10])

if __name__ == '__main__':
    prueba()
