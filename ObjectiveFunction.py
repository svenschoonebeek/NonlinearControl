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
        return np.array([v, 1/m * (u - of.k_0 * r[i] * np.exp(-r[i]) - of.h_d * v - of.k_c * interaction)])

    @staticmethod
    def objective(
            u: np.ndarray, Np_agents: np.ndarray, u_global_set: np.ndarray, x_global_set: np.ndarray,
            i: int, t: float, dt: float, Q: np.ndarray, R: np.ndarray, V_f: float):
        x_objective, u_objective = 0, 0
        for agent, N_p in enumerate(Np_agents):
            x_term_static = []
            u_term_dynamic = []
            for delta, _ in enumerate(np.arange(t, N_p, dt)):
                print (len(np.arange(t, N_p, dt)), len(x_global_set[agent]))
                x_term_static.append(x_global_set[agent][delta].T @ Q @ x_global_set[agent][delta])
                if agent == i:
                    #print (u)
                    u_term_dynamic.append(u[delta].T * u[delta])
                else:
                    u_term_dynamic.append(u_global_set[agent][delta].T * u_global_set[agent][delta])
            x_objective += np.sum(x_term_static, axis=0); u_objective += np.sum(u_term_dynamic, axis=0)
        return np.sum(x_objective) + np.sum(u_objective) + V_f

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
        constraints = [P >> cp.Constant(1e-6 * np.eye(n)), 2 * x_f.T @ P @ np.array(f_terminal).flatten() <= 0]
        objective = cp.Minimize(cp.trace(P))
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return P.value


