import matplotlib.pyplot as plt # para hacer un grafico de confianza de la prediccion de YOLO y asi saber que imagenes seleccionar
import numpy as np


def make_plot(x, outputPath):
    plt.figure(figsize=(10, 6))
    plt.plot(x, marker='o', linestyle='-', color='b')
    plt.title('Confianza de Predicción de YOLO por Imagen')
    plt.xlabel('Número de Imagen')
    plt.ylabel('Confianza de Predicción')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(outputPath)
    plt.close()

if __name__ == '__main__':
    # Ejemplo de uso
    x = np.random.rand(20)  # Datos de ejemplo
    outputPath = 'confianza_yolo_plot.png'
    make_plot(x, outputPath)
    print(f"Gráfico guardado en {outputPath}")
