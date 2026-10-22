# Task 4.3 : le SGD mini-batch de la 4.2, accéléré avec numba.
# Lancer depuis le dossier handin2 :  .venv/bin/python task4_numba.py
import time

import numpy as np
from numba import njit


# Le même SGD que la 4.2, écrit avec des boucles simples (c'est ce que numba accélère).
# Modèle : y = m·x + h   (x et y : tableaux 1D de nombres)
def sgd(x, y, learning_rate, batch_size, n_epochs):
    m = 0.0
    h = 0.0
    n = x.shape[0]
    for epoch in range(n_epochs):
        order = np.random.permutation(n)               # on mélange les points
        for start in range(0, n, batch_size):          # un paquet de batch_size points
            grad_m = 0.0
            grad_h = 0.0
            count = 0
            for k in order[start:start + batch_size]:  # on additionne les gradients du paquet
                error = (m * x[k] + h) - y[k]          # ŷ − y
                grad_m += error * x[k]                 # (ŷ − y) × x
                grad_h += error                        # (ŷ − y) × 1
                count += 1
            m -= learning_rate * grad_m / count        # un pas, avec la moyenne du paquet
            h -= learning_rate * grad_h / count
    return m, h


# La MÊME fonction, compilée en code machine par numba.
sgd_fast = njit(sgd)


if __name__ == "__main__":
    # beaucoup de données pour que la différence de vitesse se voie
    rng = np.random.default_rng(0)
    n = 100_000
    x = rng.uniform(0, 2, n)
    y = 1.0 * x + 0.5 + rng.normal(0, 0.1, n)          # la droite y = 1·x + 0.5, avec du bruit
    params = dict(learning_rate=0.1, batch_size=8, n_epochs=5)

    # 1) Python normal
    t0 = time.perf_counter()
    m, h = sgd(x, y, **params)
    t_python = time.perf_counter() - t0
    print(f"Python : m = {m:.3f}, h = {h:.3f}   en {t_python:.2f} s")

    # 2) numba : le 1er appel compile la fonction, on ne le chronomètre donc pas
    sgd_fast(x[:16], y[:16], 0.1, 8, 1)
    t0 = time.perf_counter()
    m, h = sgd_fast(x, y, **params)
    t_numba = time.perf_counter() - t0
    print(f"Numba  : m = {m:.3f}, h = {h:.3f}   en {t_numba:.4f} s")

    print(f"Numba est environ {t_python / t_numba:.0f} fois plus rapide")
