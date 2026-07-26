"""
ARKaD Kapitel 5 & 6 — Jacobi-Matrix / 微分运动学
================================================

Ch3/Ch4 处理的是"位置层面"的映射；本章处理**速度层面**：

    v_E = [ ṗ_E ; ω_E ] = J_G(q) · q̇          几何 Jacobian
    ẋ_E = [ ṗ_E ; Φ̇_E ] = J_A(q) · q̇          解析 Jacobian
    J_G = T_A(Φ_E) · J_A,   T_A = diag(I, J_R(Φ_E))

Jacobian 是整门课的枢纽，它同时是：
  · 速度映射         q̇ → v_E                 (5.1, 5.3, 6.1)
  · 力/力矩映射      τ = Jᵀ·F   （虚功原理）   (动力学、力控)
  · 可操作性判据     det J = 0 → 奇异         (5.2)
  · 数值逆解的下降方向                        (4.3, 6.3)
  · 动力学质量矩阵的积木  M = Σ mJ_PᵀJ_P + J_OᵀRIRᵀJ_O   (Ch7)

几何 Jacobian 的列（讲义黄框）：
    转动关节 i:  j_GPi = ⁰e_{z,i-1} × (p_E − ⁰r_{i-1,0}),  j_GOi = ⁰e_{z,i-1}
    移动关节 i:  j_GPi = ⁰e_{z,i-1},                        j_GOi = 0
"""

from __future__ import annotations

import numpy as np

from ch2_orientation import rot_x, rot_y, rot_z, unskew


# ============================================================================
# 5.1 几何 Jacobian
# ============================================================================


def geometric_jacobian(robot, q, upto: int | None = None) -> np.ndarray:
    """J_G(q) ∈ ℝ^{6×n} —— Ex 5.1 b)。

    upto: 只算到第 upto 个坐标系（算连杆质心 Jacobian 时用）。
    """
    Ts = robot.T_abs(q)
    end = len(robot.links) if upto is None else upto
    p_E = Ts[end][:3, 3]

    cols_P, cols_O = [], []
    k = 0
    for i, lk in enumerate(robot.links):
        if lk.joint not in ("R", "P"):
            continue
        z = Ts[i][:3, 2]            # ⁰e_{z,(i-1)0}
        p = Ts[i][:3, 3]            # ⁰r_{(i-1)0,00}
        active = (i < end)          # 第 i 个关节影响不到它后面的坐标系
        if not active:
            cols_P.append(np.zeros(3))
            cols_O.append(np.zeros(3))
        elif lk.joint == "R":
            cols_P.append(np.cross(z, p_E - p))
            cols_O.append(z)
        else:                       # 'P'
            cols_P.append(z)
            cols_O.append(np.zeros(3))
        k += 1
    return np.vstack([np.column_stack(cols_P), np.column_stack(cols_O)])


def position_jacobian(robot, q, upto=None) -> np.ndarray:
    """J_GP —— 几何 Jacobian 的上半 3 行。"""
    return geometric_jacobian(robot, q, upto)[:3, :]


def orientation_jacobian(robot, q, upto=None) -> np.ndarray:
    """J_GO —— 下半 3 行。"""
    return geometric_jacobian(robot, q, upto)[3:, :]


def com_jacobians(robot, q, com_local):
    """连杆质心的 (J_P, J_O) 列表 —— Ch7 拉格朗日建模要用。

    com_local[i] = ⁱr_{CG,link i, i0}，即质心在**自身坐标系**里的位置。
    """
    Ts = robot.T_abs(q)
    n = robot.n
    out = []
    joint_idx = [i for i, lk in enumerate(robot.links) if lk.joint in ("R", "P")]
    for li, i_link in enumerate(joint_idx):
        T_i = Ts[i_link + 1]
        p_c = T_i[:3, :3] @ np.asarray(com_local[li], float) + T_i[:3, 3]
        JP = np.zeros((3, n))
        JO = np.zeros((3, n))
        for j, i_j in enumerate(joint_idx[: li + 1]):
            z = Ts[i_j][:3, 2]
            p = Ts[i_j][:3, 3]
            if robot.links[i_j].joint == "R":
                JP[:, j] = np.cross(z, p_c - p)
                JO[:, j] = z
            else:
                JP[:, j] = z
        out.append((JP, JO, p_c))
    return out


# ============================================================================
# 5.2 奇异 (Singularities)
# ============================================================================


def manipulability(J: np.ndarray) -> float:
    """Yoshikawa 可操作度 w = √det(J Jᵀ)；方阵时 = |det J|。w=0 即奇异。"""
    if J.shape[0] == J.shape[1]:
        return abs(np.linalg.det(J))
    return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))


def is_singular(J: np.ndarray, tol: float = 1e-6) -> bool:
    return manipulability(J) < tol


def rrp_det_position_jacobian(q, l2: float = 0.4) -> float:
    """Ex 5.2 手推的结果：det J_GP = s2·q3² + c2·q3·l2。

    → 令其为 0（且 q3 ≠ 0）得  tan q2 = −l2/q3
    这就是**肩奇异 (shoulder singularity)**：TCP 落在第 1 关节轴线上，
    转 q1 不改变 TCP 位置 → 少了一个可控方向。
    """
    q2, q3 = q[1], q[2]
    return np.sin(q2) * q3 ** 2 + np.cos(q2) * q3 * l2


def rrp_shoulder_singularity_q2(q3: float, l2: float = 0.4) -> float:
    """给定伸出量 q3，返回触发肩奇异的 q2。"""
    return np.arctan2(-l2, q3)


# ============================================================================
# 5.3 递归微分运动学（速度沿运动链前向传播）
# ============================================================================


def forward_velocity_recursion(robot, q, qd):
    """Ex 5.3：逐杆递推 ⁰ω_i 和 ⁰ṙ_{i0,00}，返回 (omegas, vels)。

    转动关节:  ⁰ω_i = ⁰ω_{i−1} + ⁰R_{i−1}·[0,0,q̇_i]ᵀ
               ⁰ṙ_{i0,00} = ⁰ṙ_{i−1,0} + ⁰ω_i × ⁰r_{i0,(i−1)0}
    移动关节:  ⁰ω_i = ⁰ω_{i−1}
               ⁰ṙ_{i0,00} = ⁰ṙ_{i−1,0} + ⁰R_{i−1}·[0,0,q̇_i]ᵀ + ⁰ω_i × ⁰r_{i0,(i−1)0}

    统一写法（讲义用 ν_i = 1 移动 / 0 转动）：
        ⁰ω_i = ⁰ω_{i−1} + (1−ν_i)·⁰R_{i−1}·[0,0,q̇_i]ᵀ
        ⁰ṙ_i = ⁰ṙ_{i−1} + ν_i·⁰R_{i−1}·[0,0,q̇_i]ᵀ + ⁰ω_i × ⁰r_{i0,(i−1)0}

    结果必须和 J_G(q)·q̇ 完全一致 —— 这是本章最好的自检。
    """
    Ts = robot.T_abs(q)
    qd = np.atleast_1d(np.asarray(qd, float))
    omegas = [np.zeros(3)]
    vels = [np.zeros(3)]
    k = 0
    for i, lk in enumerate(robot.links):
        R_prev = Ts[i][:3, :3]
        r_rel = Ts[i + 1][:3, 3] - Ts[i][:3, 3]      # ⁰r_{i0,(i−1)0}
        if lk.joint == "R":
            nu, qdi = 0.0, qd[k]; k += 1
        elif lk.joint == "P":
            nu, qdi = 1.0, qd[k]; k += 1
        else:
            nu, qdi = 0.0, 0.0
        axis = R_prev @ np.array([0.0, 0.0, qdi])
        w = omegas[-1] + (1 - nu) * axis
        v = vels[-1] + nu * axis + np.cross(w, r_rel)
        omegas.append(w)
        vels.append(v)
    return omegas, vels


# ============================================================================
# 6.1 解析 Jacobian（平面 RRR 闭式）
# ============================================================================


def planar_rrr_fk(q, l=(1.0, 1.0, 1.0)) -> np.ndarray:
    """x_E = [px, py, ψ]ᵀ  —— Ex 6.1 ①"""
    q1, q2, q3 = q
    l1, l2, l3 = l
    return np.array([
        l1 * np.cos(q1) + l2 * np.cos(q1 + q2) + l3 * np.cos(q1 + q2 + q3),
        l1 * np.sin(q1) + l2 * np.sin(q1 + q2) + l3 * np.sin(q1 + q2 + q3),
        q1 + q2 + q3,
    ])


def planar_rrr_analytical_jacobian(q, l=(1.0, 1.0, 1.0)) -> np.ndarray:
    """J_A = ∂x_E/∂q  —— Ex 6.1 ②（直接对 x_E 求偏导，这就是"解析"的含义）。"""
    q1, q2, q3 = q
    l1, l2, l3 = l
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    s123, c123 = np.sin(q1 + q2 + q3), np.cos(q1 + q2 + q3)
    return np.array([
        [-l1 * s1 - l2 * s12 - l3 * s123, -l2 * s12 - l3 * s123, -l3 * s123],
        [l1 * c1 + l2 * c12 + l3 * c123, l2 * c12 + l3 * c123, l3 * c123],
        [1.0, 1.0, 1.0],
    ])


def numeric_jacobian(f, q, eps: float = 1e-7) -> np.ndarray:
    """通用数值 Jacobian（中心差分）—— 验算手推结果的利器。"""
    q = np.asarray(q, float)
    f0 = np.atleast_1d(f(q))
    J = np.zeros((f0.size, q.size))
    for i in range(q.size):
        dq = np.zeros_like(q)
        dq[i] = eps
        J[:, i] = (np.atleast_1d(f(q + dq)) - np.atleast_1d(f(q - dq))) / (2 * eps)
    return J


# ============================================================================
# 6.2 旋转 Jacobian J_R(Φ)  —— 欧拉角速度 Φ̇ → 角速度 ω
# ============================================================================


def rotation_jacobian(euler_fn, Phi, eps: float = 1e-7) -> np.ndarray:
    """通用 J_R(Φ)：ω = J_R(Φ)·Φ̇。

    原理（讲义蓝框）：
        ∂R(Φ)/∂χ · R(Φ)ᵀ = [ ∂r_E/∂χ ]_×      χ ∈ {φ, θ, ψ}
    即：对每个欧拉角求导，右乘 Rᵀ 得到一个反对称阵，vee 出来就是 J_R 的一列。
    """
    Phi = np.asarray(Phi, float)
    R = euler_fn(*Phi)
    cols = []
    for i in range(3):
        dP = np.zeros(3)
        dP[i] = eps
        dR = (euler_fn(*(Phi + dP)) - euler_fn(*(Phi - dP))) / (2 * eps)
        cols.append(unskew(dR @ R.T))
    return np.column_stack(cols)


def rotation_jacobian_zyx_closed_form(Phi) -> np.ndarray:
    """Ex 6.2 手推结果（ZY'X'' current frame）：

        J_R(Φ) = ⎡ 0   −sφ    cφ·cθ ⎤
                 ⎢ 0    cφ    sφ·cθ ⎥
                 ⎣ 1    0    −sθ    ⎦
    """
    phi, theta, _psi = np.asarray(Phi, float)
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    return np.array([
        [0.0, -sp, cp * ct],
        [0.0, cp, sp * ct],
        [1.0, 0.0, -st],
    ])


def euler_zyx(phi, theta, psi):
    return rot_z(phi) @ rot_y(theta) @ rot_x(psi)


def euler_zyz(phi, theta, psi):
    return rot_z(phi) @ rot_y(theta) @ rot_z(psi)


def T_A(Phi, euler_fn=euler_zyx) -> np.ndarray:
    """T_A(Φ) = diag(I₃, J_R(Φ))，使 J_G = T_A · J_A。"""
    T = np.eye(6)
    T[3:, 3:] = rotation_jacobian(euler_fn, Phi)
    return T


# ============================================================================
# 6.3 逆微分运动学（Jacobian 转置算法）
# ============================================================================


def rrp_position_analytical(q, l1: float = 0.2, l2: float = 0.4) -> np.ndarray:
    """RRP 的 ⁰r_{30,00} 闭式 —— Ex 6.3 ①。"""
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    return np.array([
        c1 * s2 * q3 + l2 * c1 * c2,
        s1 * s2 * q3 + l2 * s1 * c2,
        -c2 * q3 + l2 * s2 + l1,
    ])


def rrp_position_jacobian_analytical(q, l2: float = 0.4) -> np.ndarray:
    """J_{A,P}(Φ,q) —— Ex 6.3 题面给出的那个矩阵。"""
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    return np.array([
        [-s1 * (l2 * c2 + s2 * q3), -c1 * (l2 * s2 - c2 * q3), c1 * s2],
        [c1 * (l2 * c2 + s2 * q3), -s1 * (l2 * s2 - c2 * q3), s1 * s2],
        [0.0, l2 * c2 + s2 * q3, -c2],
    ])


def jacobian_transpose_ik(f, J, p_desired, q0, dt: float = 0.5,
                          iterations: int = 3, K=None, verbose: bool = False):
    """Ex 6.3 的 5 步算法：

      1. 正运动学          p_{k−1,E} = k(q_{k−1})
      2. 加权误差向量      e_{k−1,wp} = K·(p_d − p_{k−1,E})
      3. 关节速度          q̇_k = J_Aᵀ(q_{k−1})·e_{k−1,wp}
      4. 关节位形          q_k = q_{k−1} + Δt·q̇_k
      5. 未收敛 → 回到 1

    和 4.3 的梯度法是同一个东西，只是把 α 解释成积分步长 Δt。
    """
    q = np.asarray(q0, float).copy()
    K = np.eye(3) if K is None else np.asarray(K, float)
    hist = []
    for k in range(iterations):
        p = f(q)
        e = K @ (np.asarray(p_desired, float) - p)
        qd = J(q).T @ e
        q = q + dt * qd
        hist.append({"k": k + 1, "p": p.copy(), "e": e.copy(),
                     "qd": qd.copy(), "q": q.copy()})
        if verbose:
            print(f"  Iteration {k+1}: p={np.round(p,4)}  q̇={np.round(qd,4)}  "
                  f"q={np.round(q,4)}")
    return q, hist


def inverse_differential(J: np.ndarray, v, damping: float = 0.0):
    """真正的"逆微分运动学"：q̇ = J⁻¹v（方阵）/ J⁺v（伪逆）/ 阻尼伪逆。

    damping > 0 时用 q̇ = Jᵀ(JJᵀ + λ²I)⁻¹v —— 过奇异点时必须加阻尼，
    否则 J⁻¹ 爆炸，关节速度冲上天（这就是 5.2 奇异分析的工程意义）。
    """
    v = np.asarray(v, float)
    if damping > 0:
        m = J.shape[0]
        return J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(m), v)
    return np.linalg.pinv(J) @ v
