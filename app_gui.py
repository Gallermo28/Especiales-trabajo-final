import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from image_utils.procesamiento import crear_imagen_gris_prueba, procesar_imagen_según_modo_real, separar_canales_rgb, a_senal_1d

class App:
    def __init__(self, master):
        self.master = master
        master.title("Selector de imagen")

        self.label = tk.Label(master, text="¿Quieres usar la imagen de prueba en escala de grises?")
        self.label.pack(padx=20, pady=10)

        self.button_si = tk.Button(master, text="Sí", command=self.usar_prueba)
        self.button_si.pack(side=tk.LEFT, padx=20, pady=10)

        self.button_no = tk.Button(master, text="No, cargar archivo", command=self.cargar_archivo)
        self.button_no.pack(side=tk.RIGHT, padx=20, pady=10)

    def usar_prueba(self):
        self.img = crear_imagen_gris_prueba()
        self.procesar_y_mostrar()

    def cargar_archivo(self):
        ruta = filedialog.askopenfilename(title="Selecciona una imagen",
                                          filetypes=[("Imágenes", "*.png *.jpg *.jpeg")])
        if not ruta:
            messagebox.showinfo("Información", "No se seleccionó ninguna imagen.")
            return
        self.img = Image.open(ruta)
        self.procesar_y_mostrar()

    def procesar_y_mostrar(self):
        img_proc, es_gris = procesar_imagen_según_modo_real(self.img)
        arr = np.array(img_proc)

        if es_gris:
            plt.imshow(arr, cmap='gray')
            tipo_txt = "Imagen Gris"
        else:
            plt.imshow(arr)
            tipo_txt = "Imagen Color"
        plt.title(tipo_txt)
        plt.axis('off')
        plt.show(block=False)  # No bloqueante para que continúe ejecución

        print(f"{tipo_txt} procesada")
        print("Ejemplo de señal 1D:")

        if es_gris:
            senal = a_senal_1d(arr)
            print(senal[:10])
        else:
            r, g, b = separar_canales_rgb(img_proc)
            senal_r = a_senal_1d(r)
            print(senal_r[:10])

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == '__main__':
    main()
