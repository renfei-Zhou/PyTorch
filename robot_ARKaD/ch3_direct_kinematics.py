"""
ARKaD Kapitel 3 — Direkte Kinematik / 正运动学
==============================================

本章回答：**给定关节变量 q，末端在哪、朝向如何？** 即映射 x_E = k(q)。

两条路：
  a) Ex 3.1 那种"看图硬推"—— 简单平面机构可以，自由度一多就废。
  b) Ex 3.2/3.3 的 **DH 约定** —— 把每个连杆压缩成 4 个参数
     (l, λ, d, δ) = (a, α, d, θ)，机械地拼出 ⁰T_n。工业界只走这条路。

讲义用的变换矩阵（标准/远端 DH）：

    ⁱ⁻¹T_i = Rot_z(δ) · Trans_z(d) · Trans_x(l) · Rot_x(λ)

           ⎡ cδ   −sδ·cλ    sδ·sλ   l·cδ ⎤
         = ⎢ sδ    cδ·cλ   −cδ·sλ   l·sδ ⎥
           ⎢ 0     sλ       cλ      d    ⎥
           ⎣ 0     0        0       1    ⎦

画 DH 坐标系的口诀（讲义黄字）：
  ① 画关节轴 → ② 每个坐标系的 z 沿关节轴 → ③ x = z_{i-1} × z_i
     （两 z 平行时 x 沿杆的延伸方向）→ ④ 填表
  d_{j,i}: 沿 z 方向两条 x 轴的间距   l_{j,i}: 沿 x 方向两条 z 轴的间距
  λ_{j,i}: 绕 x 从 z_{i-1} 转到 z_i   δ_{j,i}: 绕 z 从 x_{i-1} 转到 x_i
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ch2_orientation import T_inverse  # noqa: F401  (re-export, 常用)


@dataclass
class DHLink:
    """一行 DH 表。

    l     : a_{j,i}  连杆长度  [m]  —— 沿 x_{i-1}
    lam   : α_{j,i}  连杆扭角  [rad] —— 绕 x_{i-1}
    d     : d_{j,i}  连杆偏距  [m]  —— 沿 z_{i-1}
    delta : θ_{j,i}  关节转角  [rad] —— 绕 z_{i-1}
    joint : 'R' 转动关节 (q → delta) / 'P' 移动关节 (q → d) / 'F' 固定
    """

    l: float = 0.0
    lam: float = 0.0
    d: float = 0.0
    delta: float = 0.0
    joint: str = "R"

    def with_q(self, q: float) -> "DHLink":
        if self.joint == "R":
            return replace(self, delta=self.delta + q)
        if self.joint == "P":
            return replace(self, d=self.d + q)
        return self

    def T(self) -> np.ndarray:
        cd, sd = np.cos(self.delta), np.sin(self.delta)
        cl, sl = np.cos(self.lam), np.sin(self.lam)
        return np.array([
            [cd, -sd * cl, sd * sl, self.l * cd],
            [sd, cd * cl, -cd * sl, self.l * sd],
            [0.0, sl, cl, self.d],
            [0.0, 0.0, 0.0, 1.0],
        ])


class DHRobot:
    """由 DH 表定义的串联机器人。"""

    def __init__(self, links, name: str = "robot"):
        self.links = list(links)
        self.name = name

    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        """驱动关节数（'F' 固定连杆不算自由度）。"""
        return sum(1 for lk in self.links if lk.joint in ("R", "P"))

    def _links_at(self, q):
        """把 q 填进 DH 表。q 只喂给 R/P 关节，固定连杆跳过。"""
        q = np.atleast_1d(np.asarray(q, dtype=float))
        out, k = [], 0
        for lk in self.links:
            if lk.joint in ("R", "P"):
                out.append(lk.with_q(q[k]))
                k += 1
            else:
                out.append(lk)
        return out

    # ------------------------------------------------------------------ #
    def T_rel(self, q):
        """所有相邻变换 [⁰T_1, ¹T_2, ...]  —— Ex 3.3 a)"""
        return [lk.T() for lk in self._links_at(q)]

    def T_abs(self, q):
        """所有绝对变换 [I, ⁰T_1, ⁰T_2, ..., ⁰T_n]  —— Ex 3.3 b)"""
        Ts, T = [np.eye(4)], np.eye(4)
        for Ti in self.T_rel(q):
            T = T @ Ti
            Ts.append(T.copy())
        return Ts

    def fk(self, q, upto: int | None = None) -> np.ndarray:
        """⁰T_upto（默认末端）。这就是正运动学 x_E = k(q)。"""
        Ts = self.T_abs(q)
        return Ts[len(self.links) if upto is None else upto]

    # --- 常用切片 ------------------------------------------------------- #
    def position(self, q, upto=None) -> np.ndarray:
        """⁰r_{i0,00}"""
        return self.fk(q, upto)[:3, 3]

    def rotation(self, q, upto=None) -> np.ndarray:
        """⁰R_i"""
        return self.fk(q, upto)[:3, :3]

    def origins(self, q):
        """所有坐标系原点 ⁰r_{i0,00}, i = 0..n （Jacobian 要用）。"""
        return [T[:3, 3] for T in self.T_abs(q)]

    def z_axes(self, q):
        """所有 z 轴 ⁰e_{z,i0}, i = 0..n （Jacobian 要用）。"""
        return [T[:3, 2] for T in self.T_abs(q)]

    def joint_types(self):
        return [lk.joint for lk in self.links if lk.joint in ("R", "P")]

    def __repr__(self):
        return f"<DHRobot {self.name}: n={self.n}, links={len(self.links)}>"


# ============================================================================
# 讲义里出现的三台机器人
# ============================================================================

#: UR5 —— Ex 3.2 求出的 DH 表 (Ex 3.3 / 4.1 / 4.3 都用它)
UR5 = DHRobot([
    DHLink(l=0.000, lam=np.pi / 2, d=0.089, joint="R"),   # 1,0
    DHLink(l=0.425, lam=0.0,       d=0.000, joint="R"),   # 2,1
    DHLink(l=0.392, lam=0.0,       d=0.000, joint="R"),   # 3,2
    DHLink(l=0.000, lam=np.pi / 2, d=0.109, joint="R"),   # 4,3
    DHLink(l=0.000, lam=np.pi / 2, d=0.095, joint="R"),   # 5,4
    DHLink(l=0.000, lam=0.0,       d=0.082, joint="R"),   # 6,5
], name="UR5")

#: UR5 的"手臂段"（前 3 轴），Ex 4.1 / 4.3 只需要它来定位
UR5_ARM = DHRobot(UR5.links[:3], name="UR5-arm")


def rrp_robot(l1: float = 0.2, l2: float = 0.4) -> DHRobot:
    """Ex 5.1 / 5.2 / 5.3 / 6.3 的 RRP 机构。

    DH:  1,0: (0,  π/2, l1, q1)  R
         2,1: (l2, π/2, 0,  q2)  R
         3,2: (0,  0,   q3, 0 )  P
    """
    return DHRobot([
        DHLink(l=0.0, lam=np.pi / 2, d=l1, joint="R"),
        DHLink(l=l2,  lam=np.pi / 2, d=0.0, joint="R"),
        DHLink(l=0.0, lam=0.0,       d=0.0, joint="P"),
    ], name=f"RRP(l1={l1}, l2={l2})")


def rrr_spatial_robot(l1: float = 1.0, l2: float = 1.0, l3: float = 0.5) -> DHRobot:
    """Ex 7 / Ex 8 动力学那台 RRR 机构（第 4 个是固定连杆到 TCP）。

    DH:  0,1: (0,  π/2, l1, q1)  R
         1,2: (l2, 0,   0,  q2)  R
         2,3: (0,  π/2, 0,  q3)  R
         3,4: (0,  0,   l3, 0 )  fixed
    """
    return DHRobot([
        DHLink(l=0.0, lam=np.pi / 2, d=l1,  joint="R"),
        DHLink(l=l2,  lam=0.0,       d=0.0, joint="R"),
        DHLink(l=0.0, lam=np.pi / 2, d=0.0, joint="R"),
        DHLink(l=0.0, lam=0.0,       d=l3,  joint="F"),
    ], name="RRR-spatial")


# ============================================================================
# Ex 3.1 —— 不用 DH，直接几何推导的平面 RP 机构（对照用）
# ============================================================================


def planar_rp_fk(theta10: float, d21: float, l10: float = 1.0,
                 theta21: float = np.pi / 2):
    """平面 R-P 机构的正运动学，返回 (x_E, ⁰T_2)。

    ⁰r_{20,00} = [ c1·l10 − s1·d21 ,  s1·l10 + c1·d21 ]ᵀ
    ψ          = θ10 + θ21
    """
    from ch2_orientation import homogeneous, rot_z

    c1, s1 = np.cos(theta10), np.sin(theta10)
    p = np.array([c1 * l10 - s1 * d21,
                  s1 * l10 + c1 * d21,
                  0.0])
    psi = theta10 + theta21
    x_E = np.array([p[0], p[1], psi])
    return x_E, homogeneous(rot_z(psi), p)
