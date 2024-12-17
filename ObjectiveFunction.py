import numpy as np
from scipy.optimize import fsolve

class ObjectiveFunction:

    k_0 : float = 1.1
    k_c : float = 0.25
    h_d : float = 0.30
    dt : float = 0.01
    m_1 : float = 1.5
    m_2 : float = 2
    m_3 : float = 1
    B : np.ndarray = np.vstack([np.zeros((3,3)), np.diag([1/m_1, 1/m_2, 1/m_3])])

    @staticmethod
    def v_dot(i: int, m: np.ndarray, u: np.ndarray, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        of = ObjectiveFunction
        interaction : np.ndarray = np.array([])
        if i == 1:
            interaction = r[0] - r[1]
        elif i == 2:
            interaction = 2 * r[1] - r[0] - r[2]
        elif i == 3:
            interaction = r[2] - r[1]
        return 1/m[i] * (u - of.k_0 * r * np.exp(-r) - of.h_d * v - of.k_c * interaction)

    @staticmethod
    def r_dot(i: int, v : np.ndarray) -> np.ndarray:
        return v[i]

    @staticmethod
    def x_predict_iteration(v: np.ndarray, v_dot : np.ndarray) -> np.ndarray:
        return np.array([v, v_dot])

    @staticmethod
    def compute_A(dvidrj : np.ndarray) -> np.ndarray:
        of = ObjectiveFunction
        return np.vstack([np.hstack([np.zeros((3,3)), np.identity(3)]), np.hstack([dvidrj, -of.h_d * np.diag([1/of.m_1, 1/of.m_2, 1/of.m_3])])])

    @staticmethod
    def compute_dvidrj(r_1: float, r_2: float, r_3: float) -> np.ndarray:
        of = ObjectiveFunction
        return np.vstack([
            np.hstack([1/of.m_1 * (of.k_0 * np.exp(-r_1) * (r_1 - 1) - of.k_c), of.k_c/of.m_1, 0]),
            np.hstack([of.k_c/of.m_2, 1/of.m_2 * (of.k_0 * np.exp(-r_2) * (r_2 - 1) - 2*of.k_c), of.k_c/of.m_2]),
            np.hstack([0, of.k_c/of.m_3, 1/of.m_3 * (of.k_0 * np.exp(-r_3) * (r_3 - 1) - of.k_c)])
        ])


    @staticmethod
    def equilibrium():

    @staticmethod
    def compute_equilibrium():




    @staticmethod
    def u_predict_iteration():








