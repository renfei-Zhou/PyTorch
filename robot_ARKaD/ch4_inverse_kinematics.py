"""
ARKaD Kapitel 4 — Inverse Kinematik / 逆运动学
==============================================

本章回答：**想让末端到某个位姿，各关节该转多少？** 即求 q = k⁻¹(x_E)。

和正运动学不同，逆解是"难的那一半"：
  · 可能无解（超出工作空间）
  · 可能多解（UR5 通常 8 组：肩左右 × 肘上下 × 腕翻转）
  · 一般没有闭式解

三种做法：
  4.1 解析法 - 位置   把 ⁰T_3 的平移部分和目标对应，代数消元 → q1,q2,q3
  4.2 解析法 - 姿态   把 ⁰R_3 和目标 Euler 矩阵**逐元素对应**，用
                      tan = r_ij/r_kj 的比值形式解角度（避免 acos 丢符号）
  4.3 数值法          梯度/Jacobian 转置迭代，通用但只给一个解

**球腕解耦 (spherical wrist)** 是解析法能用的关键：后三轴交于一点 →
位置只由前三轴决定，姿态只由后三轴决定，6 元非线性方程组拆成两个 3 元。
"""

from __future__ import annotations

import numpy as np

from ch2_orientation import euler_zyz_current


# ============================================================================
# 4.1 解析逆解 —— 位置（UR5 手臂段）
# ============================================================================


def ur5_arm_ik_position(p_target, l21: float = 0.425, l32: float = 0.392,
                        d10: float = 0.089):
    """Ex 4.1：给 ⁰r_{30,00}，求 (q1, q2, q3)。

    正运动学给出
        ⁰r_{30,00} = [ c1·(l21c2 + l32c23),
                       s1·(l21c2 + l32c23),
                       l21s2 + l32s23 + d10 ]ᵀ

    步骤：
      ① y/x 相除 → tan q1 = py/px  →  q1 与 q1+π 两支（肩左 / 肩右）
      ② 消去 q1 后两式平方相加 → c3 = (px²+py²+(pz−d10)² − l21² − l32²)/(2 l21 l32)
         → q3 = ±acos(c3)（肘上 / 肘下）
      ③ 回代用 atan2 解 q2
    返回所有满足的解 list[(q1,q2,q3)]。
    """
    p = np.asarray(p_target, dtype=float)
    px, py, pz = p
    pz = pz - d10                       # 去掉基座高度

    r2 = px ** 2 + py ** 2
    c3 = (r2 + pz ** 2 - l21 ** 2 - l32 ** 2) / (2 * l21 * l32)
    if abs(c3) > 1 + 1e-9:
        return []                        # 够不着
    c3 = np.clip(c3, -1.0, 1.0)

    sols = []
    for q1 in (np.arctan2(py, px), np.arctan2(py, px) + np.pi):
        # 平面内的"径向"坐标：把点投影到 q1 决定的竖直平面上
        rho = px * np.cos(q1) + py * np.sin(q1)
        for q3 in (np.arccos(c3), -np.arccos(c3)):
            k1 = l21 + l32 * np.cos(q3)
            k2 = l32 * np.sin(q3)
            q2 = np.arctan2(pz, rho) - np.arctan2(k2, k1)
            sols.append(_wrap(np.array([q1, q2, q3])))
    return _dedup(sols)


# ============================================================================
# 4.2 解析逆解 —— 姿态（球腕，ZY'Z'' 欧拉角）
# ============================================================================


def spherical_wrist_ik_orientation(Phi):
    """Ex 4.2：给 Euler ZY'Z'' 角 Φ = [α, β, γ]，求腕部 (q1, q2, q3)。

    腕部 DH 给出（讲义 ③）：
        ⁰R_3 = ⎡ c1c2c3 − s1s3   −c1c2s3 − s1c3    c1s2 ⎤
               ⎢ s1c2c3 + c1s3   −s1c2s3 + c1c3    s1s2 ⎥
               ⎣ −s2c3            s2s3            −c2   ⎦
    这正好是 ZY'Z'' 的形式（差一个符号约定），所以：
        q2 = atan2(±√(r31²+r32²), −r33)
        q1 = atan2(±r23, ±r13)
        q3 = atan2(±r32, ∓r31)
    ± 对应腕翻转两支。**永远用 atan2 的比值形式，不要用单个 acos。**
    """
    R = euler_zyz_current(*np.asarray(Phi, dtype=float))
    return so3_to_zyz_like(R)


def so3_to_zyz_like(R: np.ndarray):
    """把目标旋转矩阵拆成讲义那套腕部三角（含两支解）。"""
    sols = []
    for sign in (+1, -1):
        s2 = sign * np.hypot(R[2, 0], R[2, 1])
        q2 = np.arctan2(s2, -R[2, 2])
        if abs(s2) < 1e-9:                      # 奇异：q1、q3 只有和/差确定
            q1 = 0.0
            q3 = np.arctan2(-R[0, 1], R[0, 0])
            sols.append(_wrap(np.array([q1, q2, q3])))
            continue
        q1 = np.arctan2(R[1, 2] / s2, R[0, 2] / s2)
        q3 = np.arctan2(R[2, 1] / s2, -R[2, 0] / s2)
        sols.append(_wrap(np.array([q1, q2, q3])))
    return _dedup(sols)


def wrist_fk_rotation(q):
    """腕部正解 ⁰R_3，用来校验 4.2 的逆解。"""
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    c3, s3 = np.cos(q3), np.sin(q3)
    return np.array([
        [c1 * c2 * c3 - s1 * s3, -c1 * c2 * s3 - s1 * c3, c1 * s2],
        [s1 * c2 * c3 + c1 * s3, -s1 * c2 * s3 + c1 * c3, s1 * s2],
        [-s2 * c3, s2 * s3, -c2],
    ])


# ============================================================================
# 4.3 数值逆解 —— 梯度法 / Jacobian 转置
# ============================================================================


def ur5_arm_position(q, l21: float = 0.425, l32: float = 0.392, d10: float = 0.089):
    """f(q) = ⁰r_{30,00}，Ex 4.3 的正运动学。"""
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    c23, s23 = np.cos(q2 + q3), np.sin(q2 + q3)
    return np.array([
        c1 * (l21 * c2 + l32 * c23),
        s1 * (l21 * c2 + l32 * c23),
        l21 * s2 + l32 * s23 + d10,
    ])


def ur5_arm_position_jacobian(q, l21: float = 0.425, l32: float = 0.392):
    """J_AP(q) = ∂⁰r_{30,00}/∂q  —— Ex 4.3 a)，解析求导得到。"""
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    c23, s23 = np.cos(q2 + q3), np.sin(q2 + q3)
    A = l21 * c2 + l32 * c23
    B = l21 * s2 + l32 * s23
    return np.array([
        [-s1 * A, -c1 * B, -c1 * l32 * s23],
        [c1 * A, -s1 * B, -s1 * l32 * s23],
        [0.0, A, l32 * c23],
    ])


def ik_gradient(f, J, x_desired, q0, alpha: float = 0.1, steps: int = 3,
                K=None, verbose: bool = False):
    """Ex 4.3 b) / Ex 6.3 的通用迭代格式。

        e_k     = K·(x_d − f(q_k))          加权误差
        q_{k+1} = q_k + α·J(q_k)ᵀ·e_k       Jacobian 转置 / 梯度法

    α 在 4.3 里叫"步长"，在 6.3 里叫 Δt（把它看成一步欧拉积分
    q_{k+1} = q_k + Δt·q̇_k，q̇_k = Jᵀe，两者是同一个算法的两种解读）。

    返回 (q_final, 轨迹列表)。
    """
    q = np.asarray(q0, dtype=float).copy()
    K = np.eye(len(np.atleast_1d(x_desired))) if K is None else np.asarray(K, float)
    traj = [q.copy()]
    for k in range(steps):
        e = K @ (np.asarray(x_desired, float) - f(q))
        dq = J(q).T @ e
        q = q + alpha * dq
        traj.append(q.copy())
        if verbose:
            print(f"  iter {k+1}: |e| = {np.linalg.norm(e):.6f}  q = {np.round(q, 4)}")
    return q, traj


def ik_newton(f, J, x_desired, q0, tol: float = 1e-10, max_iter: int = 200,
              damping: float = 1e-6):
    """对比用：阻尼最小二乘（Levenberg–Marquardt）版本，收敛快得多。

    q_{k+1} = q_k + (JᵀJ + λI)⁻¹ Jᵀ e
    考试不考，但能一眼看出 Jacobian 转置法为什么"慢但稳"。
    """
    q = np.asarray(q0, dtype=float).copy()
    for _ in range(max_iter):
        e = np.asarray(x_desired, float) - f(q)
        if np.linalg.norm(e) < tol:
            break
        Jq = J(q)
        H = Jq.T @ Jq + damping * np.eye(len(q))
        q = q + np.linalg.solve(H, Jq.T @ e)
    return q


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _wrap(q):
    """把角度收进 (−π, π]。"""
    return (np.asarray(q, float) + np.pi) % (2 * np.pi) - np.pi


def _dedup(sols, tol: float = 1e-6):
    """按 2π 周期去重（−π 和 +π 是同一个解）。"""
    out = []
    for s in sols:
        if not any(np.max(np.abs(_wrap(s - o))) < tol for o in out):
            out.append(s)
    return out
