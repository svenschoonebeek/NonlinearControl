import numpy as np
from ObjectiveFunction import ObjectiveFunction
from Supervisor import Supervisor

class Agent:

    def __init__(self, i: int, m: float, dt: float, t: float, X_feasible: np.ndarray, U_feasible: np.ndarray, x_global: np.ndarray, n_agents : int):
        self.of = ObjectiveFunction
        self.sf = Supervisor
        self.i : int = i
        self.m : float  = m
        self.t : float = t
        self.k : int = 0
        self.n_agents = n_agents
        self.x_global: [] = list(x_global)
        self.N_p: int = 50
        self.u_current : np.ndarray = np.zeros(self.N_p)
        self.u_past : np.ndarray = np.array([])
        self.dt : float = dt
        self.X_feasible : np.ndarray = X_feasible
        self.U_feasible : np.ndarray = U_feasible

    def predict_input_sequence(self, u_global_set: np.ndarray, x_global_set: np.ndarray):
        pass


    def predict_state_sequence(self):
        x_predict_global_k = self.x_global[0].copy()
        for k, _ in enumerate(np.arange(self.t, self.N_p, 1)):
            x_predict_local_k = self.of.predict_state(
                np.array(x_predict_global_k[:self.n_agents]),
                float(x_predict_global_k[self.n_agents + self.i]),
                float(self.u_current[k]), self.i, self.m, self.dt)
            x_predict_global_k = self.x_global[0].copy(); x_predict_global_k[self.i::self.i + 2] = x_predict_local_k
            self.x_global.append(x_predict_global_k)





























