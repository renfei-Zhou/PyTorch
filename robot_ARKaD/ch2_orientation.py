"""
ARKaD Kapitel 2 — Orientierung / 姿态表示
==========================================

本章回答一个问题：**如何用数字描述"一个坐标系相对另一个坐标系转了多少"**。

四种表示法（考试 Exam 1 的全部内容）：
  1. 旋转矩阵 R ∈ SO(3)        —— 9 个数，冗余但复合最方便
  2. 最小表示 (Euler / RPY)     —— 3 个数，有奇异 (gimbal lock)
  3. 轴角 (θ, r)               —— 4 个数，几何直观，Rodrigues 公式
  4. 单位四元数 Q = {η, ε}      —— 4 个数，无奇异，插值/复合数值稳定

核心易错点：**current frame（内旋，绕动轴）右乘，fixed frame（外旋，绕定轴）左乘。**
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# 1. 基本旋转矩阵 (Ex 2.1 方框里的三个公式)
# ----------------------------------------------------------------------------


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


_ROT = {"x": rot_x, "y": rot_y, "z": rot_z}


# ----------------------------------------------------------------------------
# 2. 最小表示：欧拉角 (Ex 2.1)
# ----------------------------------------------------------------------------


def euler_current(axes: str, angles) -> np.ndarray:
    """current frame / 内旋 / 绕**动轴**依次旋转  →  按顺序 **右乘**。

    R = R_{a1}(θ1) · R_{a2}(θ2) · R_{a3}(θ3)

    >>> euler_current("xyx", [np.pi/2, np.pi/2, np.pi])   # Ex 2.1 a)
    """
    R = np.eye(3)
    for ax, an in zip(axes, angles):
        R = R @ _ROT[ax](an)
    return R


def euler_fixed(axes: str, angles) -> np.ndarray:
    """fixed frame / 外旋 / 绕**定轴 RF0** 依次旋转  →  按顺序 **左乘**。

    R = R_{a3}(θ3) · R_{a2}(θ2) · R_{a1}(θ1)

    >>> euler_fixed("xzy", [np.pi/2, np.pi/2, np.pi])     # Ex 2.1 b)
    """
    R = np.eye(3)
    for ax, an in zip(axes, angles):
        R = _ROT[ax](an) @ R
    return R


def euler_zyz_current(phi: float, theta: float, psi: float) -> np.ndarray:
    """讲义常用的 ZY'Z'' 序列（Ex 4.2 用它做姿态逆解）。"""
    return rot_z(phi) @ rot_y(theta) @ rot_z(psi)


def euler_zyx_current(phi: float, theta: float, psi: float) -> np.ndarray:
    """ZY'X'' 序列（Ex 6.2 求它的 rotation Jacobian）。"""
    return rot_z(phi) @ rot_y(theta) @ rot_x(psi)


# ----------------------------------------------------------------------------
# 3. 轴角表示 (Ex 2.2)
# ----------------------------------------------------------------------------


def auxiliary_angles(r) -> tuple[float, float]:
    """由转轴 r = [rx, ry, rz] 求辅助角 (γ, β)。

    γ = atan2(ry, rx)                     绕 z 把轴转到 xz 平面
    β = atan2(sqrt(rx²+ry²), rz)          绕 y 把轴转到 z 上

    注意：手算时用 arctan 必须自己判象限（讲义里 x<0, y>0 要 +π），
    代码里用 atan2 自动搞定 —— 这正是 atan2 存在的意义。
    """
    r = np.asarray(r, dtype=float)
    gamma = np.arctan2(r[1], r[0])
    beta = np.arctan2(np.hypot(r[0], r[1]), r[2])
    return gamma, beta


def rot_axis_angle(theta: float, r, normalize: bool = True) -> np.ndarray:
    """Rodrigues 公式 —— Ex 2.2 方框里那个 3x3 大矩阵。

    R = r rᵀ (1-cθ) + I cθ + [r]_× sθ

    normalize=False 可以复现讲义"直接代入非严格单位向量"的做法
    （Ex 2.2 给的 r 模长是 1.0024，结果差在小数点后第 3 位）。
    """
    r = np.asarray(r, dtype=float)
    n = np.linalg.norm(r)
    if n < 1e-12:
        return np.eye(3)
    if normalize:
        r = r / n
    c, s = np.cos(theta), np.sin(theta)
    return np.outer(r, r) * (1 - c) + np.eye(3) * c + skew(r) * s


def axis_angle_from_rot(R: np.ndarray) -> tuple[float, np.ndarray]:
    """逆问题：R → (θ, r)。θ = acos((tr R - 1)/2)，r 由反对称部分取出。"""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if abs(np.sin(theta)) < 1e-9:                       # θ≈0 或 π，退化
        if theta < 1e-9:
            return 0.0, np.array([0.0, 0.0, 1.0])
        # θ = π: 从 (R+I)/2 = r rᵀ 取列
        rr = (R + np.eye(3)) / 2
        r = np.sqrt(np.clip(np.diag(rr), 0, None))
        k = int(np.argmax(r))
        r = rr[:, k] / r[k]
        return theta, r / np.linalg.norm(r)
    r = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    return theta, r


def skew(v) -> np.ndarray:
    """[v]_×  —— 叉乘的矩阵形式，后面 Jacobian / 角速度到处要用。"""
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]])


def unskew(S: np.ndarray) -> np.ndarray:
    """[v]_× → v （vee 算子），Ex 6.2 从 dR·Rᵀ 里抠出角速度分量时用。"""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


# ----------------------------------------------------------------------------
# 4. 单位四元数 (Ex 2.3)
# ----------------------------------------------------------------------------


class Quaternion:
    """单位四元数 Q = {η, ε}, η = cos(θ/2), ε = sin(θ/2)·r。"""

    __slots__ = ("eta", "eps")

    def __init__(self, eta: float, eps):
        self.eta = float(eta)
        self.eps = np.asarray(eps, dtype=float)

    # --- 构造 ---------------------------------------------------------------
    @classmethod
    def from_axis_angle(cls, theta: float, r) -> "Quaternion":
        r = np.asarray(r, dtype=float)
        r = r / np.linalg.norm(r)
        return cls(np.cos(theta / 2), np.sin(theta / 2) * r)

    @classmethod
    def from_matrix(cls, R: np.ndarray) -> "Quaternion":
        eta = 0.5 * np.sqrt(max(0.0, 1 + np.trace(R)))
        if eta > 1e-8:
            eps = np.array([R[2, 1] - R[1, 2],
                            R[0, 2] - R[2, 0],
                            R[1, 0] - R[0, 1]]) / (4 * eta)
        else:                                            # θ ≈ π
            theta, r = axis_angle_from_rot(R)
            return cls.from_axis_angle(theta, r)
        return cls(eta, eps)

    # --- 运算 ---------------------------------------------------------------
    def to_matrix(self) -> np.ndarray:
        """Ex 2.3 方框里的 R(η, ε)。"""
        eta = self.eta
        ex, ey, ez = self.eps
        return np.array([
            [2 * (eta ** 2 + ex ** 2) - 1, 2 * (ex * ey - eta * ez), 2 * (ex * ez + eta * ey)],
            [2 * (ex * ey + eta * ez), 2 * (eta ** 2 + ey ** 2) - 1, 2 * (ey * ez - eta * ex)],
            [2 * (ex * ez - eta * ey), 2 * (ey * ez + eta * ex), 2 * (eta ** 2 + ez ** 2) - 1],
        ])

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """四元数乘法 = 旋转的复合（等价于 R1·R2）。Ex 2.3 d)

        η12 = η1η2 − ε1ᵀε2
        ε12 = η1ε2 + η2ε1 + ε1 × ε2
        """
        eta = self.eta * other.eta - self.eps @ other.eps
        eps = (self.eta * other.eps + other.eta * self.eps
               + np.cross(self.eps, other.eps))
        return Quaternion(eta, eps)

    def conj(self) -> "Quaternion":
        return Quaternion(self.eta, -self.eps)

    def rotate(self, v) -> np.ndarray:
        return self.to_matrix() @ np.asarray(v, dtype=float)

    def norm(self) -> float:
        return np.sqrt(self.eta ** 2 + self.eps @ self.eps)

    def __repr__(self) -> str:
        e = np.round(self.eps, 4)
        return f"Q(eta={self.eta:.4f}, eps=[{e[0]}, {e[1]}, {e[2]}])"


# ----------------------------------------------------------------------------
# 5. 齐次变换的小工具（Ch3 会大量用，放这里因为它本质是"旋转+平移"）
# ----------------------------------------------------------------------------


def homogeneous(R: np.ndarray, p) -> np.ndarray:
    """iT_j = [[R, r],[0,1]]  —— Ex 3.1 b) 方框公式。"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p, dtype=float).ravel()
    return T


def T_inverse(T: np.ndarray) -> np.ndarray:
    """(iT_j)⁻¹ = jT_i = [[Rᵀ, −Rᵀr],[0,1]]（别用 np.linalg.inv，浪费且不稳）。"""
    R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti
