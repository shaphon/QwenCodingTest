"""
Kwant 基础演示代码：量子点接触 (Quantum Point Contact)

这个示例展示了如何使用 Kwant 库：
1. 定义晶格和系统形状
2. 创建散射区域和引线
3. 添加 onsite 能量和跃迁项
4. 计算透射系数和电导
"""

import kwant
from kwant import builder
import numpy as np
import matplotlib.pyplot as plt

# 定义正方晶格
lat = kwant.lattice.square(a=1, name='a')

# 定义系统形状函数（宽度为 W 的条形区域）
def rectangle(pos):
    x, y = pos
    return -W/2 < y < W/2 and 0 <= x < L

# 定义引线形状函数
def lead_shape(pos):
    x, y = pos
    return -W/2 < y < W/2

# 系统参数
W = 10  # 宽度（以晶格常数为单位）
L = 20  # 长度（以晶格常数为单位）
t = 1.0  # 跃迁强度
E_F = 0.5  # 费米能级

# 创建系统
syst = builder.Builder()

# 添加散射区域的格点和跃迁
syst[lat.shape(rectangle, (0, 0))] = 4 * t  # onsite 能量
syst[builder.HoppingKind((1, 0), lat)] = -t  # x 方向跃迁
syst[builder.HoppingKind((0, 1), lat)] = -t  # y 方向跃迁

# 添加左引线（沿 -x 方向）
lead0 = builder.Builder(kwant.TranslationalSymmetry((-1, 0)))
lead0[lat.shape(lead_shape, (0, 0))] = 4 * t
lead0[builder.HoppingKind((1, 0), lat)] = -t
lead0[builder.HoppingKind((0, 1), lat)] = -t
syst.attach_lead(lead0)

# 添加右引线（沿 +x 方向）
lead1 = builder.Builder(kwant.TranslationalSymmetry((1, 0)))
lead1[lat.shape(lead_shape, (0, 0))] = 4 * t
lead1[builder.HoppingKind((1, 0), lat)] = -t
lead1[builder.HoppingKind((0, 1), lat)] = -t
syst.attach_lead(lead1)

# 最终化系统
syst = syst.finalized()

# 可视化系统结构
print("系统已创建并可视化...")
print(f"系统尺寸：{L} x {W}")
print(f"总格点数：{len(syst.sites)}")
print(f"总跃迁数：{len(syst.hoppings)}")

# 绘制系统结构图
plt.figure(figsize=(10, 4))
kwant.plot(syst, site_size=50, site_color='lightblue', 
           hop_color='gray', hop_lw=1)
plt.title('Kwant Demo: Quantum Point Contact System')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('kwant_system_structure.png', dpi=150)
print("System structure saved as 'kwant_system_structure.png'")

# 计算透射系数随能量的变化
energies = np.linspace(-3*t, 3*t, 100)
transmissions = []

print("\n计算透射系数...")
for E in energies:
    smatrix = kwant.smatrix(syst, energy=E)
    # 从引线 0 到引线 1 的透射
    T = smatrix.transmission(1, 0)
    transmissions.append(T)

transmissions = np.array(transmissions)

# 绘制透射系数图
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(energies/t, transmissions, 'b-', linewidth=2)
ax.set_xlabel('Energy / t', fontsize=12)
ax.set_ylabel('Transmission T(E)', fontsize=12)
ax.set_title('Transmission vs Energy', fontsize=14)
ax.axvline(x=E_F/t, color='r', linestyle='--', label=f'Fermi Energy (E_F/t={E_F/t})')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-3, 3)
ax.set_ylim(0, max(transmissions)*1.1)
plt.tight_layout()
plt.savefig('kwant_transmission.png', dpi=150)
print("Transmission plot saved as 'kwant_transmission.png'")

# 计算在费米能级处的电导（Landauer 公式：G = (2e²/h) * T）
smatrix = kwant.smatrix(syst, energy=E_F)
T_F = smatrix.transmission(1, 0)
G0 = 2 * (1.602e-19)**2 / 6.626e-34  # 2e²/h ≈ 7.748e-5 S
conductance = G0 * T_F

print(f"\n=== 计算结果 ===")
print(f"费米能级 E_F = {E_F:.2f} t")
print(f"透射系数 T(E_F) = {T_F:.4f}")
print(f"电导 G = {conductance:.4e} S  (或 {T_F:.4f} × 2e²/h)")

# 展示模式信息
print(f"\n引领导模数（在 E_F 处）:")
for i in range(len(syst.leads)):
    n_modes = smatrix.num_propagating(i)
    print(f"  引线 {i}: {n_modes} 个传播模式")

print("\n演示完成！请查看生成的图片文件。")
plt.show()
