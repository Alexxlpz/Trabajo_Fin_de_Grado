import matplotlib.pyplot as plt # para hacer un grafico de confianza de la prediccion de YOLO y asi saber que imagenes seleccionar
import numpy as np


def make_plot(x, name):
    print(len(x))
    plt.figure(figsize=(10, 6))
    plt.plot(x, marker='o', linestyle='-', color='b')
    plt.title('Confidence Scores of Detected Objects')
    plt.xlabel('Number of objets detected')
    plt.ylabel('Confidence Score')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig("C:/Users/alexl/PycharmProjects/Trabajo_Fin_de_Grado/leavesDetection/data/plots/"+name)
    plt.close()

if __name__ == '__main__':
    # Ejemplo de uso
    x = np.random.rand(20)  # Datos de ejemplo
    name = 'confianza_yolo_plot'
    make_plot(x, name)
    print(f"Gráfico guardado en ../../data/plots/{name}")
