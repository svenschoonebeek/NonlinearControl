import numpy as np
from ObjectiveFunction import ObjectiveFunction
from scipy.optimize import minimize

class Agent:

    def __init__(
            self, i: int, m: float, dt: float, t: float, X_feasible: np.ndarray,
            U_feasible: np.ndarray, n_agents : int, Q : np.ndarray, R: np.ndarray, tolerance : float):
        self.of = ObjectiveFunction
        self.i : int = i
        self.m : float  = m
        self.t : float = t
        self.k : int = 0
        self.n_agents = n_agents
        self.x_global: [] = []
        self.x_global_set : [] = []
        self.N_p: int = 50
        self.Np_agents : np.ndarray = np.array([50, 50, 50])
        self.u_current : [] = np.zeros(self.N_p).tolist()
        self.J_current : float = 1e9
        self.J_past : float = 1e9
        self.dt : float = dt
        self.X_feasible : np.ndarray = X_feasible
        self.U_feasible : np.ndarray = U_feasible
        self.Q : np.ndarray = Q
        self.R : np.ndarray = R
        self.tolerance : float = tolerance

    def predict_input_sequence(self, u_global_set: np.ndarray, x_global_set: np.ndarray, V_f: float):
        self.J_past = self.J_current
        bounds = [(self.U_feasible[0], self.U_feasible[1])]
        result = minimize(self.of.objective, self.u_current, args=(self.Np_agents, u_global_set, x_global_set, self.i, self.t, self.dt, self.Q, self.R, V_f), bounds=bounds, method='SLSQP')
        self.J_current, self.u_current = result.fun, result.x
        #if np.abs(self.J_current - self.J_past) < self.tolerance: break

    def predict_state_sequence(self, x_current : np.ndarray):
        for k, _ in enumerate(np.arange(self.t, self.N_p, self.dt)):
            x_predict_global_k = x_current
            x_predict_local_k = self.of.predict_state(
                np.array(x_predict_global_k[:self.n_agents]),
                float(x_predict_global_k[self.n_agents + self.i]),
                float(self.u_current[k]), self.i, self.m, self.dt)
            x_predict_global_k[self.i:self.i + 2] = x_predict_local_k
            self.x_global.append(x_predict_global_k)





























