"""
逐题复现讲义里的每一道 Ex，并和手写答案对照。
运行:  python demo_exercises.py
"""

import numpy as np

import ch2_orientation as ch2
import ch3_direct_kinematics as ch3
import ch4_inverse_kinematics as ch4
import ch56_jacobian as ch56
import ch789_dynamics as ch789

np.set_printoptions(precision=4, suppress=True, linewidth=110)


def title(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)


def check(name, got, expected, tol=1e-2):
    ok = np.allclose(np.asarray(got, float), np.asarray(expected, float), atol=tol)
    print(f"   [{'OK ' if ok else 'XX '}] {name}")
    return ok


# ============================================================================
title("Ex 2.1  Minimal Orientation —— current frame vs. fixed frame")
# ============================================================================
R_cur = ch2.euler_current("xyx", [np.pi / 2, np.pi / 2, np.pi])
R_fix = ch2.euler_fixed("xzy", [np.pi / 2, np.pi / 2, np.pi])
print("R_current (Rx(π/2)·Ry(π/2)·Rx(π)) =\n", R_cur)
print("R_fixed   (Ry(π)·Rz(π/2)·Rx(π/2)) =\n", R_fix)
check("R_cur == R_fix  (讲义结论)", R_cur, R_fix)
check("R_cur == [[0,0,-1],[1,0,0],[0,-1,0]]", R_cur,
      [[0, 0, -1], [1, 0, 0], [0, -1, 0]])

# ============================================================================
title("Ex 2.2  Axis and Angle")
# ============================================================================
theta = 0.7 * np.pi
r = np.array([-0.53, 0.74, 0.42])
gamma, beta = ch2.auxiliary_angles(r)
print(f"γ = {gamma:.4f}   (讲义 2.19)")
print(f"β = {beta:.4f}   (讲义 1.14)")
R = ch2.rot_axis_angle(theta, r)
R_raw = ch2.rot_axis_angle(theta, r, normalize=False)   # 讲义的做法（r 未归一化）
print("R(θ, r)  归一化 =\n", R)
print("R(θ, r)  照讲义直接代入 =\n", R_raw)
p = R @ np.array([0.0, 0.2, 0.7])
p_raw = R_raw @ np.array([0.0, 0.2, 0.7])
print("旋转后的点 (归一化)   =", np.round(p, 4))
print("旋转后的点 (讲义 R)   =", np.round(p_raw, 4),
      "  讲义写的是 [-0.02, 0.685, -0.2026]")
check("γ", gamma, 2.19, 1e-2)
check("β", beta, 1.14, 1e-2)
check("x, z 分量与讲义一致", [p_raw[0], p_raw[2]], [-0.02, -0.2026], 3e-3)
print("   [!!] y 分量：讲义 0.685，实际应为 %.4f —— 讲义此处笔误" % p_raw[1])
th2, r2 = ch2.axis_angle_from_rot(R)
check("R → (θ,r) 往返一致", [th2, *r2], [theta, *(r / np.linalg.norm(r))], 1e-6)

# ============================================================================
title("Ex 2.3  Unit Quaternions")
# ============================================================================
Q1 = ch2.Quaternion.from_axis_angle(np.pi / 2, [1, 0, 0])
Q2 = ch2.Quaternion.from_axis_angle(np.pi / 2, [0, 1, 0])
Q3 = ch2.Quaternion.from_axis_angle(np.pi / 2, [0.33, 0.33, 0.33])
print("Q1 =", Q1, "\nQ2 =", Q2, "\nQ3 =", Q3)
print("R(Q1) =\n", Q1.to_matrix())
print("R(Q2) =\n", Q2.to_matrix())
print("R(Q3) =\n", Q3.to_matrix())
Q12 = Q1 * Q2
print("Q12 = Q1·Q2 =", Q12, "  (讲义 {0.5, [0.5,0.5,0.5]})")
print("R(Q12) =\n", Q12.to_matrix())
check("Q12", [Q12.eta, *Q12.eps], [0.5, 0.5, 0.5, 0.5], 1e-3)
check("R(Q12) == R(Q1)·R(Q2)", Q12.to_matrix(), Q1.to_matrix() @ Q2.to_matrix())
check("R(Q12) == [[0,0,1],[1,0,0],[0,1,0]]", Q12.to_matrix(),
      [[0, 0, 1], [1, 0, 0], [0, 1, 0]])

# ============================================================================
title("Ex 3.1  Direct Kinematics Mapping (平面 RP)")
# ============================================================================
x_E, T2 = ch3.planar_rp_fk(theta10=0.3, d21=0.4, l10=1.0)
print("x_E = [px, py, ψ] =", np.round(x_E, 4))
print("⁰T_2 =\n", T2)

# ============================================================================
title("Ex 3.2 / 3.3  UR5 DH-Konvention & Direct Kinematics")
# ============================================================================
print(ch3.UR5)
q = np.zeros(6)
for i, T in enumerate(ch3.UR5.T_rel(q), start=1):
    print(f"  {i-1}T{i} =\n{T}")
T03 = ch3.UR5.fk(np.array([0.3, -0.5, 0.8, 0, 0, 0]), upto=3)
print("⁰T_3 (q=[0.3,-0.5,0.8]) =\n", T03)

# ============================================================================
title("Ex 4.1  Analytical IK — Position")
# ============================================================================
p_d = np.array([-0.817, 0.0, 0.089])
sols = ch4.ur5_arm_ik_position(p_d)
print("目标 ⁰r_30,00 =", p_d)
for s in sols:
    print("   q =", np.round(s, 4),
          " → FK 校验:", np.round(ch4.ur5_arm_position(s), 4))
print("讲义答案: (q1,q2,q3) = (0, π, 0) 或 (π, 0, 0)")
check("解满足正运动学", [ch4.ur5_arm_position(s) for s in sols],
      [p_d] * len(sols), 1e-6)

# ============================================================================
title("Ex 4.2  Analytical IK — Orientation (球腕, Euler ZY'Z'')")
# ============================================================================
Phi = np.array([np.pi / 2, -np.pi / 2, 0.0])
R_target = ch2.euler_zyz_current(*Phi)
print("R_ZY'Z''(Φ) =\n", R_target)
for s in ch4.spherical_wrist_ik_orientation(Phi):
    print("   q =", np.round(s, 4),
          " → ‖R(q) − R_d‖ =", np.linalg.norm(ch4.wrist_fk_rotation(s) - R_target))

# ============================================================================
title("Ex 4.3  Numerical IK — 梯度法 (α=0.1, 3 步)")
# ============================================================================
q0 = np.array([np.pi / 2, np.pi / 2, np.pi / 2])
print("J_AP(q0) =\n", ch4.ur5_arm_position_jacobian(q0))
q_end, traj = ch4.ik_gradient(ch4.ur5_arm_position,
                              ch4.ur5_arm_position_jacobian,
                              p_d, q0, alpha=0.1, steps=3, verbose=True)
print("3 步后 q =", np.round(q_end, 4), "  (讲义 [1.4741, 1.3733, 1.6198])")
check("q1, q3 与讲义一致", [q_end[0], q_end[2]], [1.4741, 1.6198], 1e-3)
print("   [!!] q2：讲义写 1.3733，按它自己前两步的 1.5711 和 J^T·e 推应为 %.4f"
      " —— 讲义抄写笔误 (1.5733 → 1.3733)" % q_end[1])
q_conv = ch4.ik_newton(ch4.ur5_arm_position, ch4.ur5_arm_position_jacobian, p_d, q0)
print("对比：阻尼最小二乘收敛到 q =", np.round(q_conv, 4),
      " 残差 =", np.linalg.norm(ch4.ur5_arm_position(q_conv) - p_d))

# ============================================================================
title("Ex 5.1  Geometric Jacobian (RRP)")
# ============================================================================
rrp = ch3.rrp_robot(l1=0.2, l2=0.4)
q = np.array([0.3, 0.6, 0.5])
print("⁰r_10,00 =", np.round(rrp.position(q, upto=1), 4))
print("⁰r_20,00 =", np.round(rrp.position(q, upto=2), 4))
print("⁰r_30,00 = P_E =", np.round(rrp.position(q), 4))
JG = ch56.geometric_jacobian(rrp, q)
print("J_G =\n", JG)
check("闭式 P_E 与 DH 一致", ch56.rrp_position_analytical(q), rrp.position(q), 1e-9)
check("J_GP 与数值微分一致", JG[:3],
      ch56.numeric_jacobian(lambda x: rrp.position(x), q), 1e-5)

# ============================================================================
title("Ex 5.2  Singularities")
# ============================================================================
print("det J_GP (手推公式)      =", ch56.rrp_det_position_jacobian(q))
print("det J_GP (数值 DH 计算)  =", np.linalg.det(JG[:3]))
check("两种 det 一致", ch56.rrp_det_position_jacobian(q), np.linalg.det(JG[:3]), 1e-9)
q3 = 0.5
q2_sing = ch56.rrp_shoulder_singularity_q2(q3, l2=0.4)
q_sing = np.array([0.3, q2_sing, q3])
print(f"\n肩奇异条件 tan q2 = −l2/q3  →  q2 = {q2_sing:.4f} (q3={q3})")
print("   det J_GP =", np.linalg.det(ch56.geometric_jacobian(rrp, q_sing)[:3]))
print("   可操作度 w =", ch56.manipulability(ch56.geometric_jacobian(rrp, q_sing)[:3]))
print("   奇异? ", ch56.is_singular(ch56.geometric_jacobian(rrp, q_sing)[:3]))

# ============================================================================
title("Ex 5.3  Differential Kinematics (递归 vs. Jacobian)")
# ============================================================================
q = np.array([np.pi / 2, -np.pi / 2, 0.1])
qd = np.array([0.1, 0.2, 0.1])
w, v = ch56.forward_velocity_recursion(rrp, q, qd)
print("递归结果:")
for i in range(1, 4):
    print(f"   ⁰ω_{i} = {np.round(w[i], 4)}   ⁰ṙ_{i}0,00 = {np.round(v[i], 4)}")
print("讲义答案: ⁰ω_3 = [0.2, 0, 0.1],  ⁰ṙ_30,00 = [0.01, -0.02, -0.02]")
JG = ch56.geometric_jacobian(rrp, q)
vE = JG @ qd
print("Jacobian 结果:  ṗ_E =", np.round(vE[:3], 4), "  ω_E =", np.round(vE[3:], 4))
check("ω_3", w[3], [0.2, 0.0, 0.1], 1e-9)
check("ṙ_3", v[3], [0.01, -0.02, -0.02], 1e-9)
check("递归 == J_G·q̇  (本章最重要的自检)", np.concatenate([v[3], w[3]]), vE, 1e-9)

# ============================================================================
title("Ex 6.1  Analytical Jacobian (平面 RRR)")
# ============================================================================
q = np.array([0.4, -0.7, 1.1])
JA = ch56.planar_rrr_analytical_jacobian(q)
print("J_A =\n", JA)
check("J_A == 数值微分",
      JA, ch56.numeric_jacobian(lambda x: ch56.planar_rrr_fk(x), q), 1e-6)

# ============================================================================
title("Ex 6.2  Rotation Jacobian (ZY'X'' current frame)")
# ============================================================================
Phi = np.array([0.3, -0.6, 1.2])
JR_num = ch56.rotation_jacobian(ch56.euler_zyx, Phi)
JR_cf = ch56.rotation_jacobian_zyx_closed_form(Phi)
print("J_R 数值 =\n", JR_num)
print("J_R 讲义闭式 =\n", JR_cf)
check("闭式 == 数值 (手推结果正确)", JR_cf, JR_num, 1e-6)
print("\nJ_G = T_A(Φ)·J_A,  T_A =\n", ch56.T_A(Phi))

# ============================================================================
title("Ex 6.3  Inverse Differential Kinematics (Jacobian 转置, Δt=0.5, 3 步)")
# ============================================================================
q0 = np.array([np.pi / 4, np.pi / 4, 0.5])
p_d = np.array([0.0, 0.4, 0.1])
q_end, hist = ch56.jacobian_transpose_ik(
    ch56.rrp_position_analytical, ch56.rrp_position_jacobian_analytical,
    p_d, q0, dt=0.5, iterations=3, verbose=True)
print("讲义: q1=[0.8754, 0.8071→划掉改成0.764, 0.3852], q2=[0.9465, ...], q3=[1.0075, ...]")
check("iter1 q1, q3", [hist[0]["q"][0], hist[0]["q"][2]], [0.8754, 0.3852], 1e-3)
check("iter1 q2 == 讲义自己的修正值 0.764", hist[0]["q"][1], 0.764, 1e-3)
check("iter3 q1", hist[2]["q"][0], 1.0075, 1e-3)
print("   [!!] 讲义 iter2/iter3 的 q2 沿用了被划掉的 0.8071，所以后两步 q2 偏大；"
      "本代码用修正值继续迭代。")
print("\n对比：直接用伪逆一步走 q̇ = J⁺·(p_d − p_0)")
J0 = ch56.rrp_position_jacobian_analytical(q0)
print("   q̇ =", np.round(ch56.inverse_differential(
    J0, p_d - ch56.rrp_position_analytical(q0)), 4))

# ============================================================================
title("Ex 7 / Ex 8 / Ex 9.1  Dynamics —— Lagrange vs. Newton-Euler")
# ============================================================================
rrr = ch3.rrr_spatial_robot(l1=1.0, l2=1.0, l3=0.5)
links = ch789.EX7_LINKS
q = np.array([0.3, 0.7, -0.4])
qd = np.array([0.5, -0.2, 0.9])
qdd = np.array([0.1, 0.4, -0.3])

M = ch789.mass_matrix(rrr, q, links)
g = ch789.gravity_vector(rrr, q, links)
C = ch789.coriolis_matrix(rrr, q, qd, links)
print("M(q) =\n", M)
print("g(q) =", np.round(g, 4))
print("C(q,q̇) =\n", C)

tau_L = ch789.inverse_dynamics_lagrange(rrr, q, qd, qdd, links)
tau_NE = ch789.inverse_dynamics_newton_euler(rrr, q, qd, qdd, links)
print("\nτ (Lagrange)      =", np.round(tau_L, 6))
print("τ (Newton-Euler)  =", np.round(tau_NE, 6))
check("两种方法结果一致（Ch7 ≡ Ch8）", tau_L, tau_NE, 1e-4)

print("\nM 对称正定? ", np.allclose(M, M.T), np.all(np.linalg.eigvals(M) > 0))
qdd_back = ch789.forward_dynamics(rrr, q, qd, tau_L, links)
print("正动力学回代 q̈ =", np.round(qdd_back, 6), " (原始 q̈ =", qdd, ")")
check("正动力学 ∘ 逆动力学 = 恒等 (Ex 9.1)", qdd_back, qdd, 1e-8)

print("\n静态重力力矩 (q̇=q̈=0):",
      np.round(ch789.inverse_dynamics_newton_euler(
          rrr, q, np.zeros(3), np.zeros(3), links), 4),
      "  ==  g(q) =", np.round(g, 4))

# ============================================================================
title("讲义勘误汇总（代码交叉验证发现）")
# ============================================================================
print("""
 1) Ex 2.2 b)   ⁰r_P 的 y 分量：讲义写 0.685，按讲义自己给的 R 矩阵应为 ≈0.702
                （归一化转轴后 0.699）。x、z 分量都对。
 2) Ex 4.3 b)   第 3 步 q2：讲义写 1.3733，应为 1.5733（前一步是 1.5711，
                增量只有 +0.0022，不可能掉到 1.37）。抄写时 5 写成 3。
 3) Ex 6.3      Iteration 1 的 q2：讲义先写 0.8071，旁边已自己划掉改成 0.764
                （代码算得 0.7636，正确）；但 Iteration 2/3 仍沿用了 0.8071，
                所以后两步的 q2 需要重算。q1、q3 不受影响。
 4) Ex 6.3 中间量  J^T 第二行第一个元素讲义写 0.105，按公式应为 0.05；
                不过后面的乘法用的是 0.05，最终 q̇ 是对的（只是抄错）。
 5) Ex 4.2      讲义自己用红圈标了 "有问题" 的那一步：R_ZY'Z''(Φ) 第一次代入
                算错，下一页已修正为 [[0,-1,0],[0,0,-1],[1,0,0]]，以修正版为准。
""")

print("=" * 74)
print("全部完成。")
print("=" * 74)
