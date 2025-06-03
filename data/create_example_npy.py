# create_example_npy.py

import numpy as np
import os

# Asegúrate de que la carpeta "data/" exista
os.makedirs("data", exist_ok=True)

# Crea datos de ejemplo: un array aleatorio de forma (10, 1024)
# que simula un embedding de 10 frames con 1024 dimensiones cada uno.
dummy = np.random.rand(10, 1024).astype(np.float32)

# Guarda el array en "data/example.npy"
np.save("data/example.npy", dummy)

print("Created data/example.npy with shape", dummy.shape)
