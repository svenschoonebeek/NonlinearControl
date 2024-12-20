import numpy as np
import cvxpy as cp
from scipy.optimize import fsolve
from scipy.linalg import solve_continuous_lyapunov

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
    def f_i(i: int, m: float, u: float, r: float, v: float) -> np.ndarray:
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
    def compute_P(x_f: np.ndarray, u_f: np.ndarray) -> np.ndarray:
        of = ObjectiveFunction
        n = x_f.shape[0]
        f_terminal = []
        P = cp.Variable((n, n), symmetric=True)
        for i in np.arange(0, len(u_f), 1):
            f_terminal.append(of.f_i(i, of.m[i], u_f[i], x_f[i], x_f[i + len(of.m)]))
        constraints = [P >> 0, 2 * x_f.T @ P @ np.array([f_terminal]) <= 0]
        objective = cp.Minimize(cp.trace(P))
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return P.value

    @staticmethod
    def predict_state(x: np.ndarray, x_t: np.ndarray, u: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        of = ObjectiveFunction
        dvi_dvj: np.ndarray = of.compute_dvi_drj(float(x_t[0]), float(x_t[1]), float(x_t[2]))
        A_t: np.ndarray = of.compute_A(dvi_dvj)
        return np.matmul(A_t, x - x_t) + np.matmul(of.B, u - u_t)










    @staticmethod
    def predict_state(x: np.ndarray, x_t: np.ndarray, u: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        of = ObjectiveFunction
        dvi_dvj : np.ndarray = of.compute_dvi_drj(float(x_t[0]), float(x_t[1]), float(x_t[2]))
        A_t : np.ndarray = of.compute_A(dvi_dvj)
        return np.matmul(A_t, x - x_t) + np.matmul(of.B, u - u_t)

    @staticmethod
    def initialize(t: float, N_p: int, u: np.ndarray) -> np.ndarray:
        of = ObjectiveFunction
        u_guess : np.ndarray = np.zeros(N_p)
        if t != 0:
            u_guess = np.concatenate((u[1:], u[-1]))
        of.predict_state()
        return u





    @staticmethod
    def terminal_input_control(u: np.ndarray) -> float:

        @staticmethod
        def compute_A(dvi_drj: np.ndarray) -> np.ndarray:
            of = ObjectiveFunction
            return np.vstack([np.hstack([np.zeros((3, 3)), np.identity(3)]),
                              np.hstack([dvi_drj, -of.h_d * np.diag([1 / of.m_1, 1 / of.m_2, 1 / of.m_3])])])

        @staticmethod
        def compute_dvi_drj(r_1: float, r_2: float, r_3: float) -> np.ndarray:
            of = ObjectiveFunction
            return np.vstack([
                np.hstack([1 / of.m_1 * (of.k_0 * np.exp(-r_1) * (r_1 - 1) - of.k_c), of.k_c / of.m_1, 0]),
                np.hstack(
                    [of.k_c / of.m_2, 1 / of.m_2 * (of.k_0 * np.exp(-r_2) * (r_2 - 1) - 2 * of.k_c), of.k_c / of.m_2]),
                np.hstack([0, of.k_c / of.m_3, 1 / of.m_3 * (of.k_0 * np.exp(-r_3) * (r_3 - 1) - of.k_c)])
            ])














