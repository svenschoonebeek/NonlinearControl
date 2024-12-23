import numpy as np
import cvxpy as cp

class ObjectiveFunction:

    k_0 : float = 1.1
    k_c : float = 0.25
    h_d : float = 0.30
    dt : float = 0.01
    m_1 : float = 1.5
    m_2 : float = 2
    m_3 : float = 1
    m : np.ndarray = np.array([m_1, m_2, m_3])
    B : np.ndarray = np.vstack([np.zeros((3,3)), np.diag([1/m_1, 1/m_2, 1/m_3])])

    @staticmethod
    def f_i(i: int, m: float, u: float, r: np.ndarray, v: float) -> np.ndarray:
        of = ObjectiveFunction
        interaction : np.ndarray = np.array([])
        if i == 0:
            interaction = r[0] - r[1]
        elif i == 1:
            interaction = 2 * r[1] - r[0] - r[2]
        elif i == 2:
            interaction = r[2] - r[1]
        return np.array([v, 1/m * (u - of.k_0 * r * np.exp(-r) - of.h_d * v - of.k_c * interaction)])

    @staticmethod
    def objective():
        pass



    @staticmethod
    def predict_state(r: np.ndarray, v: float, u: float, i: int, m: float, dt: float) -> np.ndarray:
        of = ObjectiveFunction
        x = [r[i], v]
        x_dot = of.f_i(i, m, u, r, v)
        return np.array(x + x_dot * dt)

    @staticmethod
    def compute_P(x_f: np.ndarray, u_f: np.ndarray, m: np.ndarray) -> np.ndarray:
        of = ObjectiveFunction
        n = len(x_f)
        f_terminal = []
        P = cp.Variable((n, n), symmetric=True)
        for i in np.arange(0, len(u_f), 1):
            f_terminal.append(of.f_i(i, m[i], u_f[i], x_f[:len(m)], float(x_f[len(m) + i])))
        constraints = [P >> 0, 2 * x_f.T @ P @ np.array(f_terminal).flatten() <= 0]
        objective = cp.Minimize(cp.trace(P))
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return P.value


