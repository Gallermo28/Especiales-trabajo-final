import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import sounddevice as sd
import threading


class VentanaGraficos:
    def __init__(self, lista_funciones_parejas, lista_sonidos=None, func_cambiar_imagen=None):
        self.root = tk.Tk()
        self.root.title("Visualización con Navegación")
        self.lista_funciones = lista_funciones_parejas
        self.lista_sonidos = lista_sonidos or []
        self.func_cambiar_imagen = func_cambiar_imagen
        self.indice = 0

        self.figura, self.axs = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.figura, master=self.root)
        self.canvas.get_tk_widget().pack()

        frame_botones = tk.Frame(self.root)
        frame_botones.pack()

        self.boton_anterior = tk.Button(frame_botones, text="Anterior", command=self.grafico_anterior)
        self.boton_anterior.grid(row=0, column=0, padx=5, pady=5)

        self.boton_reproducir_izq = tk.Button(frame_botones, text="Reproducir Izquierda", command=self.reproducir_sonido_izq)
        self.boton_reproducir_izq.grid(row=0, column=1, padx=5, pady=5)

        self.boton_reproducir_der = tk.Button(frame_botones, text="Reproducir Derecha", command=self.reproducir_sonido_der)
        self.boton_reproducir_der.grid(row=0, column=2, padx=5, pady=5)

        self.boton_siguiente = tk.Button(frame_botones, text="Siguiente", command=self.siguiente_grafico)
        self.boton_siguiente.grid(row=0, column=3, padx=5, pady=5)

        self.mostrar_grafico()
        self.root.mainloop()

    def mostrar_grafico(self):
        self.axs[0].clear()
        self.axs[1].clear()

        func_ori, func_rec = self.lista_funciones[self.indice]
        func_ori(self.axs[0])
        func_rec(self.axs[1])
        self.canvas.draw()

        sonido_izq, sonido_der = (None, None)
        if self.lista_sonidos and self.indice < len(self.lista_sonidos):
            sonido_izq, sonido_der = self.lista_sonidos[self.indice]

        if sonido_izq is not None:
            self.boton_reproducir_izq.grid()
        else:
            self.boton_reproducir_izq.grid_remove()

        if sonido_der is not None:
            self.boton_reproducir_der.grid()
        else:
            self.boton_reproducir_der.grid_remove()

    def siguiente_grafico(self):
        self.indice = (self.indice + 1) % len(self.lista_funciones)
        self.mostrar_grafico()

    def grafico_anterior(self):
        self.indice = (self.indice - 1) % len(self.lista_funciones)
        self.mostrar_grafico()

    def reproducir_sonido_izq(self):
        if self.lista_sonidos:
            sonido = self.lista_sonidos[self.indice][0]
            if sonido is not None:
                threading.Thread(target=lambda: self._reproducir(sonido), daemon=True).start()

    def reproducir_sonido_der(self):
        if self.lista_sonidos:
            sonido = self.lista_sonidos[self.indice][1]
            if sonido is not None:
                threading.Thread(target=lambda: self._reproducir(sonido), daemon=True).start()

    def _reproducir(self, audio):
        sd.play(audio, 44100)
        sd.wait()
    