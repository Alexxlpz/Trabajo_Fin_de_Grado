import torch
# script para verificar si CUDA está disponible en el entorno de PyTorchh, si lo esta predecirá mas rapido y
# entrenará mas rapido, ya que usará la GPU en lugar de la CPU
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo: {torch.cuda.get_device_name(0)}")