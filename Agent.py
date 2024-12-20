import numpy as np
from ObjectiveFunction import ObjectiveFunction
from Supervisor import Supervisor

class Agent:

    def __init__(self, i: int, m: float):
        self.of = ObjectiveFunction
        self.sf = Supervisor
        self.i : int = i
        self.m : float  = m
        self.t : float = 0
        self.k : int = 0
        self.u_current : np.ndarray = np.array([])
        self.u_past : np.ndarray = np.array([])
        self.N_p : int = 50
        self.dt : float = 0.01
        self.x_current: np.ndarray = np.array([])
        self.x_past : np.ndarray = np.array([])
        self.x0 : np.ndarray = np.array([0, 0])
        self.X_feasible : np.ndarray = np.array([[-5, 5], [-2, 2]])

    def predict_state(self, x: np.ndarray, u: float) -> np.ndarray:
        x_dot = self.of.f_i(self.i, self.m, u, float(x[0]), float(x[1]))
        return x + x_dot * self.dt

    def initialize(self):
        if self.t != 0:
            self.u_current = np.concatenate((self.u_past[:-1], np.array([self.u_past[-2]])))
            self.x0 = np.array([self.x_past[1]])
        else:
            self.u_current = np.zeros(self.N_p)
            self.x0 = np.array([np.random.uniform(self.X_feasible[0][0], self.X_feasible[0][1]),
                                np.random.uniform(self.X_feasible[1][0], self.X_feasible[1][1])])
        x_predict = []
        for k, _ in enumerate(np.arange(self.t, self.N_p, 1)):
            if k == 0:
                x_predict.append(self.predict_state(self.x0, float(self.u_current[0])))
            else:
                x_predict.append(self.predict_state(x_predict[-1], float(self.u_current[k-1])))














