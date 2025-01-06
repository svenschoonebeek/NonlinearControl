import numpy as np
from ObjectiveFunction import ObjectiveFunction
from scipy.optimize import minimize

class Agent:

    def __init__(
            self, i: int, m: float, dt: float, t: float, X_feasible: np.ndarray,
            U_feasible: np.ndarray, Np_agents : np.ndarray, Q : np.ndarray, R: np.ndarray, tolerance : float):
        self.of = ObjectiveFunction
        self.i : int = i
        self.m : float  = m
        self.t : float = t
        self.k : int = 0
        self.Np_agents : np.ndarray = Np_agents
        self.n_agents : int = Np_agents.shape[0]
        self.x_global: [] = []
        self.x_global_set : [] = []
        self.u_current : [] = np.zeros(self.Np_agents[self.i]).tolist()
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

    def weigh_input_sequence(self, gamma: float, u_past: np.ndarray):
        self.u_current = gamma * self.u_current + (1-gamma) * u_past

    def predict_state_sequence(self, x_current : np.ndarray):
        for k, _ in enumerate(np.arange(0, self.Np_agents[self.i], 1)):
            x_predict_global_k = x_current
            x_predict_local_k = self.of.predict_state(
                np.array(x_predict_global_k[:self.n_agents]),
                float(x_predict_global_k[self.n_agents + self.i]),
                float(self.u_current[k]), self.i, self.m, self.dt)
            x_predict_global_k[self.i:self.i + 2] = x_predict_local_k
            self.x_global.append(x_predict_global_k)





























