import os

import matplotlib.pyplot as plt # para hacer un grafico de confianza de la prediccion de YOLO y asi saber que imagenes seleccionar
import numpy as np
from matplotlib import ticker


def make_plot(x, name, bins=20, fontsize=12, density=False):
    print(len(x))
    plt.figure(figsize=(10, 6))
    plt.hist(x, bins=bins, color='skyblue', edgecolor='black', alpha=0.8, density=density)
    plt.title('Distribución de confidencias', fontsize=fontsize+2)
    plt.xlabel('Confidence Score', fontsize=fontsize)
    plt.ylabel('Frecuencia' if not density else 'Densidad', fontsize=fontsize)
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_path = os.path.join(r'../../data/plots', f"{name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

if __name__ == '__main__':
    # Ejemplo de uso
    x = np.random.rand(20)  # Datos de ejemplo
    name = 'confianza_yolo_plot'
    make_plot(x, name)
    print(f"Gráfico guardado en ../../data/plots/{name}")
