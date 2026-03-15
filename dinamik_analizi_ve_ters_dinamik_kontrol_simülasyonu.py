

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import odeint

#parametreler
L1, L2 = 0.5, 0.5
m1, m2 = 1.0, 0.5
g_acc = 9.81
r1, r2 = L1/2, L2/2
I1 = (1/12)*m1*L1**2
I2 = (1/12)*m2*L2**2

#dinamik model için fonksiyon tanımlama
def get_dynamics(q, dq):
    q1, q2 = q
    dq1, dq2 = dq
    m11 = m1*r1**2 + m2*(L1**2 + r2**2 + 2*L1*r2*np.cos(q2)) + I1 + I2
    m12 = m2*(r2**2 + L1*r2*np.cos(q2)) + I2
    m22 = m2*r2**2 + I2
    M = np.array([[m11, m12], [m12, m22]])
    h = -m2*L1*r2*np.sin(q2)
    C = np.array([[h*dq2, h*(dq1+dq2)], [-h*dq1, 0]])
    g1 = (m1*r1 + m2*L1)*g_acc*np.cos(q1) + m2*r2*g_acc*np.cos(q1+q2)
    g2 = m2*r2*g_acc*np.cos(q1+q2)
    G = np.array([g1, g2])
    return M, C, G

#ters kinematik için fonksiyon tanımlama
def inverse_kinematics(x, y):
    cos_q2 = (x**2 + y**2 - L1**2 - L2**2) / (2*L1*L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    q2 = np.arccos(cos_q2)
    q1 = np.arctan2(y, x) - np.arctan2(L2*np.sin(q2), L1+L2*np.cos(q2))
    return np.array([q1, q2])

#dairesel yörünge için fonksiyon tanımlama
def get_desired_trajectory(t_arr):
    center = [0.4, 0.2]
    radius = 0.1
    omega = 1.0

    xd = center[0] + radius * np.cos(omega * t_arr)
    yd = center[1] + radius * np.sin(omega * t_arr)

    qd = np.array([inverse_kinematics(xd[i], yd[i]) for i in range(len(t_arr))])

    dt = t_arr[1] - t_arr[0]
    dqd = np.gradient(qd, dt, axis=0)
    ddqd = np.gradient(dqd, dt, axis=0)

    return qd, dqd, ddqd, xd, yd

#simülasyon parametreleri
t = np.linspace(0, 10, 1000)
qd_arr, dqd_arr, ddqd_arr, xd_arr, yd_arr = get_desired_trajectory(t)

#robot durum tanımlama
def robot_system(state, t_val, Kp, Kv):
    q = state[:2]
    dq = state[2:]
    idx = np.argmin(np.abs(t - t_val))
    qd   = qd_arr[idx]
    dqd  = dqd_arr[idx]
    ddqd = ddqd_arr[idx]
    M, C, G = get_dynamics(q, dq)
    e    = qd - q
    edot = dqd - dq
    tau  = M @ (ddqd + Kv @ edot + Kp @ e) + C @ dq + G
    ddq  = np.linalg.inv(M) @ (tau - C @ dq - G)
    return [dq[0], dq[1], ddq[0], ddq[1]]

#simülasyon çalıştırma
def run_simulation(Kp_val, Kv_val):
    Kp = np.diag([Kp_val, Kp_val])
    Kv = np.diag([Kv_val, Kv_val])
    q0 = qd_arr[0]
    sol = odeint(robot_system, [q0[0], q0[1], 0, 0], t, args=(Kp, Kv))
    return sol

# Kp=150 , Kv = 50
first = run_simulation(150, 50)

#tork hesaplama
tau_history = []
for i in range(len(t)):
    q  = first[i, :2]
    dq = first[i, 2:]
    M, C, G = get_dynamics(q, dq)
    e    = qd_arr[i] - q
    edot = dqd_arr[i] - dq
    tau  = M @ (ddqd_arr[i] + np.diag([50,50]) @ edot + np.diag([150,150]) @ e) + C @ dq + G
    tau_history.append(tau)
tau_history = np.array(tau_history)

#uç efektör
def forward_kinematics(q_arr):
    x = L1*np.cos(q_arr[:,0]) + L2*np.cos(q_arr[:,0]+q_arr[:,1])
    y = L1*np.sin(q_arr[:,0]) + L2*np.sin(q_arr[:,0]+q_arr[:,1])
    return x, y

x_real, y_real = forward_kinematics(first[:, :2])

#kazanç analizi karşılaştırma
gains = [(50, 20), (150, 50), (300, 80)]
gain_results = {}
for Kp_v, Kv_v in gains:
    s = run_simulation(Kp_v, Kv_v)
    gain_results[(Kp_v, Kv_v)] = s

#GRAFİK
#Uç Efektör Yörüngesi
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(xd_arr, yd_arr, 'b--', linewidth=2, label='İstenen')
ax.plot(x_real, y_real, 'r-', linewidth=1.5, label='Gerçek')
ax.set_title("Uç Efektör Yörüngesi (XY Uzayı)")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.legend()
ax.set_aspect('equal')
ax.grid(True)
plt.show()

#Takip Hatası q1
fig, ax = plt.subplots(figsize=(8, 4))
e1 = np.rad2deg(qd_arr[:, 0] - first[:, 0])
ax.plot(t, e1, 'g-', linewidth=1.5)
ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax.set_title("Takip Hatası - q1")
ax.set_xlabel("Zaman [s]")
ax.set_ylabel("Hata [°]")
ax.grid(True)
plt.show()

#Takip Hatası q2
fig, ax = plt.subplots(figsize=(8, 4))
e2 = np.rad2deg(qd_arr[:, 1] - first[:, 1])
ax.plot(t, e2, 'm-', linewidth=1.5)
ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax.set_title("Takip Hatası - q2")
ax.set_xlabel("Zaman [s]")
ax.set_ylabel("Hata [°]")
ax.grid(True)
plt.show()

# Tork Analizi
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, tau_history[:, 0], 'b-', label='τ1', linewidth=1.5)
ax.plot(t, tau_history[:, 1], 'r-', label='τ2', linewidth=1.5)
ax.set_title("Eklem Torku (Zaman-Tork)")
ax.set_xlabel("Zaman [s]")
ax.set_ylabel("Tork [Nm]")
ax.legend()
ax.grid(True)

# Kazanç Analizi q1 Hatası
fig, ax = plt.subplots(figsize=(8, 4))
colors = ['r', 'g', 'b']
for (Kp_v, Kv_v), color in zip(gains, colors):
    s = gain_results[(Kp_v, Kv_v)]
    err = np.rad2deg(qd_arr[:, 0] - s[:, 0])
    ax.plot(t, err, color=color, label=f'Kp={Kp_v}, Kv={Kv_v}')
ax.set_title("Kazanç Analizi - q1 Hatası")
ax.set_xlabel("Zaman [s]")
ax.set_ylabel("Hata [°]")
ax.legend()
ax.grid(True)
plt.show()

#Kazanç Analizi q2 Hatası
fig, ax = plt.subplots(figsize=(8, 4))
for (Kp_v, Kv_v), color in zip(gains, colors):
    s = gain_results[(Kp_v, Kv_v)]
    err = np.rad2deg(qd_arr[:, 1] - s[:, 1])
    ax.plot(t, err, color=color, label=f'Kp={Kp_v}, Kv={Kv_v}')
ax.set_title("Kazanç Analizi - q2 Hatası")
ax.set_xlabel("Zaman [s]")
ax.set_ylabel("Hata [°]")
ax.legend()
ax.grid(True)
plt.show()

#Kazanç Analizi Yörünge Karşılaştırma
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(xd_arr, yd_arr, 'k--', linewidth=2, label='İstenen', zorder=5)
for (Kp_v, Kv_v), color in zip(gains, colors):
    s = gain_results[(Kp_v, Kv_v)]
    xr, yr = forward_kinematics(s[:, :2])
    ax.plot(xr, yr, color=color, label=f'Kp={Kp_v}', linewidth=1.2)
ax.set_title("Kazanç Analizi - Yörünge Karşılaştırma")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.legend()
ax.set_aspect('equal')
ax.grid(True)
plt.show()
