import numpy as np

from ObjectiveFunction import ObjectiveFunction

class Supervisor:

    def __init__(self):
        self.of = ObjectiveFunction
        self.x_global : np.ndarray = np.array([])
        self.u_global : np.ndarray = np.array([])

    def compute_lyapunov(self) -> float:
        P = self.of.compute_P(self.x_global[-1], self.u_global[-1])
        return float(self.x_global[-1].T @ P @ self.x_global[-1])

