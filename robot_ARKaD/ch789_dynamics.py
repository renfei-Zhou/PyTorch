"""
ARKaD Kapitel 7 / 8 / 9 — Dynamik / 动力学
==========================================

运动学只管"几何"，动力学问"要多大力矩才能这么动"：

    τ = M(q)·q̈ + C(q, q̇)·q̇ + g(q)          （讲义写作 M·q̈ + G·q̇ + g）

  M(q)      质量/惯量矩阵，对称正定 —— 由 **Jacobian** 拼出来（Ch5 在这里回收）
  C(q,q̇)·q̇  科氏力 + 离心力 —— 由 M 的偏导（Christoffel 符号）算出
  g(q)      重力项 —— 由质心位置对 q 的偏导算出

两条等价路线：
  Ch7 Lagrange     能量法。先写 T − V，再求导。**结构清晰**，直接给出 M、C、g，
                   适合做控制器设计、参数辨识。代价：符号推导量巨大。
  Ch8 Newton–Euler 递归法。前向传播速度/加速度，后向传播力/力矩。
                   **O(n) 计算量**，适合实时控制。代价：中间量没有物理"结构"。

Ch9 Direct / Inverse Dynamics —— 同一个方程的两个方向：
  逆动力学 (q, q̇, q̈) → τ    用于前馈控制、力矩规划   → Newton–Euler 最快
  正动力学 (q, q̇, τ) → q̈    用于仿真                → q̈ = M⁻¹(τ − Cq̇ − g)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ch2_orientation import skew
from ch56_jacobian import com_jacobians


@dataclass
class LinkInertia:
    """连杆的惯性参数。

    mass      : 质量 [kg]
    com_local : ⁱr_{CG,link i, i0}  质心在自身坐标系中的位置 [m]
    inertia   : ⁱI_{link i}  相对**质心**、在自身坐标系表示的惯性张量
    """

    mass: float
    com_local: np.ndarray
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    def __post_init__(self):
        self.com_local = np.asarray(self.com_local, float)
        self.inertia = np.asarray(self.inertia, float)


GRAVITY = np.array([0.0, 0.0, -9.81])


# ============================================================================
# Ch7 —— Lagrange
# ============================================================================


def mass_matrix(robot, q, links) -> np.ndarray:
    """M(q) = Σᵢ [ mᵢ·J_Pᵢᵀ J_Pᵢ + J_Oᵢᵀ ⁰Rᵢ ⁱIᵢ ⁰Rᵢᵀ J_Oᵢ ]  —— Ex 7 ⑧

    第一项是平动动能，第二项是转动动能。注意惯性张量必须先旋到基坐标系
    （⁰RᵢIᵢ⁰Rᵢᵀ），因为 J_O 是在基坐标系里表示的。
    """
    n = robot.n
    Ts = robot.T_abs(q)
    joint_idx = [i for i, lk in enumerate(robot.links) if lk.joint in ("R", "P")]
    Js = com_jacobians(robot, q, [lk.com_local for lk in links])

    M = np.zeros((n, n))
    for li, (JP, JO, _pc) in enumerate(Js):
        R = Ts[joint_idx[li] + 1][:3, :3]
        M += links[li].mass * (JP.T @ JP) + JO.T @ (R @ links[li].inertia @ R.T) @ JO
    return 0.5 * (M + M.T)          # 数值对称化


def gravity_vector(robot, q, links, g0=GRAVITY) -> np.ndarray:
    """g(q)_i = −Σⱼ mⱼ·g₀ᵀ·∂⁰r_{CG,j}/∂qᵢ  —— Ex 7 ⑩

    等价写法 g = −Σⱼ mⱼ J_Pⱼᵀ g₀（用质心 Jacobian 更快，不用数值求导）。
    """
    Js = com_jacobians(robot, q, [lk.com_local for lk in links])
    g = np.zeros(robot.n)
    for li, (JP, _JO, _pc) in enumerate(Js):
        g -= links[li].mass * (JP.T @ np.asarray(g0, float))
    return g


def christoffel(robot, q, links, eps: float = 1e-6) -> np.ndarray:
    """Γ[i,j,k] = ½(∂m_ij/∂q_k + ∂m_ik/∂q_j − ∂m_jk/∂q_i)  —— Ex 7 ⑨

    讲义里写作 g_ijk。用中心差分算 ∂M/∂q（手推要几页，代码三行）。
    """
    n = robot.n
    dM = np.zeros((n, n, n))
    q = np.asarray(q, float)
    for k in range(n):
        dq = np.zeros(n)
        dq[k] = eps
        dM[:, :, k] = (mass_matrix(robot, q + dq, links)
                       - mass_matrix(robot, q - dq, links)) / (2 * eps)
    G = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                G[i, j, k] = 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i])
    return G


def coriolis_matrix(robot, q, qd, links) -> np.ndarray:
    """C(q,q̇)，其中 C_ij = Σ_k Γ_ijk·q̇_k  —— 讲义的 G 矩阵。

    性质检查：Ṁ − 2C 反对称（能量守恒的体现），可用来验算。
    """
    G = christoffel(robot, q, links)
    qd = np.asarray(qd, float)
    return np.einsum("ijk,k->ij", G, qd)


def inverse_dynamics_lagrange(robot, q, qd, qdd, links, g0=GRAVITY) -> np.ndarray:
    """τ = M(q)q̈ + C(q,q̇)q̇ + g(q)  —— Ex 7 ⑪ / Ex 9.1 逆动力学。"""
    M = mass_matrix(robot, q, links)
    C = coriolis_matrix(robot, q, qd, links)
    g = gravity_vector(robot, q, links, g0)
    return M @ np.asarray(qdd, float) + C @ np.asarray(qd, float) + g


def forward_dynamics(robot, q, qd, tau, links, g0=GRAVITY) -> np.ndarray:
    """q̈ = M⁻¹(τ − C q̇ − g)  —— Ex 9.1 正动力学（仿真用）。"""
    M = mass_matrix(robot, q, links)
    C = coriolis_matrix(robot, q, qd, links)
    g = gravity_vector(robot, q, links, g0)
    return np.linalg.solve(M, np.asarray(tau, float) - C @ np.asarray(qd, float) - g)


# ============================================================================
# Ch8 —— 递归 Newton–Euler
# ============================================================================


def inverse_dynamics_newton_euler(robot, q, qd, qdd, links,
                                  g0=GRAVITY, f_ext=None, mu_ext=None):
    """递归 Newton–Euler 逆动力学 —— Ex 8。

    **前向递推**（基座 → 末端），全部量在**各自局部坐标系**里表示：
        ⁱω_i   = ⁱ⁻¹R_iᵀ(ⁱ⁻¹ω_{i−1} + δ̇_i·ⁱ⁻¹e_{z,i−1})            转动
        ⁱω̇_i   = ⁱ⁻¹R_iᵀ(ⁱ⁻¹ω̇_{i−1} + δ̈_i·e_z + δ̇_i·ω_{i−1}×e_z)
        ⁱr̈_i   = ⁱ⁻¹R_iᵀ·ⁱ⁻¹r̈_{i−1} + ω̇_i×r_{i,i−1} + ω_i×(ω_i×r_{i,i−1})
        ⁱr̈_CGi = ⁱr̈_i + ω̇_i×r_CGi + ω_i×(ω_i×r_CGi)

    **重力技巧**：把基座加速度取成 ⁰r̈_00 = −g₀ = [0,0,9.81]ᵀ，
    重力就自动混进惯性力里，不用单独算 g(q)。

    **后向递推**（末端 → 基座）：
        ⁱf_{i,i−1} = ⁱR_{i+1}·ⁱ⁺¹f_{i+1,i} + mᵢ·ⁱr̈_CGi
        ⁱμ_{i,i−1} = −ⁱf_{i,i−1}×(ⁱr_{i0,(i−1)0} + ⁱr_CGi)
                     + ⁱR_{i+1}·ⁱ⁺¹μ_{i+1,i}
                     + (ⁱR_{i+1}ⁱ⁺¹f_{i+1,i})×ⁱr_CGi
                     + ⁱIᵢ·ω̇_i + ω_i×(ⁱIᵢ·ω_i)
        τ_i = ⁱμ_{i,i−1}ᵀ·ⁱ⁻¹R_iᵀ·ⁱ⁻¹e_{z,(i−1)0}   （移动关节换成 f 投影）

    讲义的实用提示：因为最后只取 z 分量，手算时**只需算第 3 行**。
    """
    q = np.atleast_1d(np.asarray(q, float))
    qd = np.atleast_1d(np.asarray(qd, float))
    qdd = np.atleast_1d(np.asarray(qdd, float))

    T_rel = robot.T_rel(q)                       # ⁱ⁻¹T_i
    nl = len(robot.links)
    R = [T[:3, :3] for T in T_rel]               # ⁱ⁻¹R_i
    r = [T[:3, 3] for T in T_rel]                # ⁱ⁻¹r_{i0,(i−1)0}
    types = [lk.joint for lk in robot.links]

    ez = np.array([0.0, 0.0, 1.0])

    # ---- 前向递推 --------------------------------------------------------
    w = [np.zeros(3)]
    wd = [np.zeros(3)]
    ad = [-np.asarray(g0, float)]                 # ⁰r̈_00 = −g₀
    ac = []                                       # 质心加速度
    k = 0
    for i in range(nl):
        Ri = R[i]
        r_i = Ri.T @ r[i]                         # 表示到坐标系 i
        if types[i] == "R":
            qdi, qddi = qd[k], qdd[k]; k += 1
            w_i = Ri.T @ (w[i] + qdi * ez)
            wd_i = Ri.T @ (wd[i] + qddi * ez + qdi * np.cross(w[i], ez))
            a_i = (Ri.T @ ad[i]
                   + np.cross(wd_i, r_i) + np.cross(w_i, np.cross(w_i, r_i)))
        elif types[i] == "P":
            qdi, qddi = qd[k], qdd[k]; k += 1
            w_i = Ri.T @ w[i]
            wd_i = Ri.T @ wd[i]
            a_i = (Ri.T @ ad[i]
                   + np.cross(wd_i, r_i) + np.cross(w_i, np.cross(w_i, r_i))
                   + qddi * ez + 2 * np.cross(w_i, qdi * ez))
        else:                                     # 固定连杆
            w_i = Ri.T @ w[i]
            wd_i = Ri.T @ wd[i]
            a_i = (Ri.T @ ad[i]
                   + np.cross(wd_i, r_i) + np.cross(w_i, np.cross(w_i, r_i)))
        w.append(w_i); wd.append(wd_i); ad.append(a_i)

    # 质心加速度（只对有惯性的连杆）
    inert_idx = [i for i, t in enumerate(types) if t in ("R", "P")]
    for li, i in enumerate(inert_idx):
        rc = links[li].com_local
        ac.append(ad[i + 1] + np.cross(wd[i + 1], rc)
                  + np.cross(w[i + 1], np.cross(w[i + 1], rc)))

    # ---- 后向递推 --------------------------------------------------------
    f_next = np.zeros(3) if f_ext is None else np.asarray(f_ext, float)
    mu_next = np.zeros(3) if mu_ext is None else np.asarray(mu_ext, float)
    R_next = np.eye(3)                            # ⁿR_{n+1}
    tau = np.zeros(robot.n)

    for i in reversed(range(nl)):
        if types[i] in ("R", "P"):
            li = inert_idx.index(i)
            m = links[li].mass
            rc = links[li].com_local
            I = links[li].inertia
            f = R_next @ f_next + m * ac[li]
            mu = (-np.cross(f, R[i].T @ r[i] + rc)
                  + R_next @ mu_next
                  + np.cross(R_next @ f_next, rc)
                  + I @ wd[i + 1]
                  + np.cross(w[i + 1], I @ w[i + 1]))
            if types[i] == "R":
                tau[li] = mu @ (R[i].T @ np.array([0.0, 0.0, 1.0]))
            else:
                tau[li] = f @ (R[i].T @ np.array([0.0, 0.0, 1.0]))
        else:                                     # 固定连杆：只传递
            f = R_next @ f_next
            mu = R_next @ mu_next - np.cross(f, R[i].T @ r[i])
        f_next, mu_next, R_next = f, mu, R[i]

    return tau


# ============================================================================
# Ch9 —— 正/逆动力学 + 简单仿真
# ============================================================================


def simulate(robot, links, q0, qd0, tau_fn, t_end: float = 1.0,
             dt: float = 1e-3, g0=GRAVITY):
    """半隐式欧拉积分正动力学 —— Ex 9.1 的"正动力学"用途。

    tau_fn(t, q, qd) → τ，可以塞进任何控制律（PD、重力补偿、计算力矩法）。
    """
    q = np.asarray(q0, float).copy()
    qd = np.asarray(qd0, float).copy()
    ts, qs = [0.0], [q.copy()]
    t = 0.0
    while t < t_end - 1e-12:
        tau = np.asarray(tau_fn(t, q, qd), float)
        qdd = forward_dynamics(robot, q, qd, tau, links, g0)
        qd = qd + dt * qdd
        q = q + dt * qd
        t += dt
        ts.append(t); qs.append(q.copy())
    return np.array(ts), np.array(qs)


# ----------------------------------------------------------------------------
# 讲义 Ex 7 / Ex 8 那台 RRR 机器人的惯性参数
# ----------------------------------------------------------------------------

EX7_LINKS = [
    LinkInertia(mass=2.0, com_local=[0.0, -0.5, 0.0],
                inertia=np.diag([10.0, 10.0, 10.0])),
    LinkInertia(mass=2.0, com_local=[-0.5, 0.0, 0.0],
                inertia=np.diag([10.0, 10.0, 10.0])),
    LinkInertia(mass=1.0, com_local=[0.0, 0.0, 0.25],
                inertia=np.diag([5.0, 5.0, 5.0])),
]
