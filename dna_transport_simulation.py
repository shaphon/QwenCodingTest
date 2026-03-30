"""
DNA 双螺旋输运过程模拟 - 基于紧束缚模型和量子输运理论
===========================================================
本程序模拟电子在 DNA 双螺旋结构中的量子输运过程，包括:
1. DNA 双螺旋几何结构构建
2. 紧束缚哈密顿量构造
3. 非平衡格林函数 (NEGF) 方法计算透射系数
4. 温度相关的电流 - 电压特性
5. 波函数传播可视化

作者：AI Assistant
日期：2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# ==================== 物理常数 ====================
hbar = 6.582e-16  # 约化普朗克常数 (eV·s)
e_charge = 1.602e-19  # 元电荷 (C)
kB = 8.617e-5  # 玻尔兹曼常数 (eV/K)


class DNABasePair:
    """DNA 碱基对类，存储碱基类型和能量参数"""
    
    BASE_ENERGIES = {
        'A': -0.5,  # 腺嘌呤
        'T': -0.4,  # 胸腺嘧啶
        'G': -0.6,  # 鸟嘌呤
        'C': -0.55, # 胞嘧啶
    }
    
    def __init__(self, base1='A', base2='T'):
        self.base1 = base1
        self.base2 = base2
        self.energy1 = self.BASE_ENERGIES.get(base1, -0.5)
        self.energy2 = self.BASE_ENERGIES.get(base2, -0.5)
        
    def __repr__(self):
        return f"{self.base1}-{self.base2}"


class DNAHelixBuilder:
    """DNA 双螺旋结构构建器"""
    
    def __init__(self, n_bases=20, radius=1.8, rise=0.34, twist=36.0):
        """
        参数:
            n_bases: 碱基对数量
            radius: 螺旋半径 (nm)
            rise: 每个碱基对的上升距离 (nm)
            twist: 每个碱基对的旋转角度 (度)
        """
        self.n_bases = n_bases
        self.radius = radius
        self.rise = rise
        self.twist = np.radians(twist)
        
        # 生成碱基序列
        self.sequence = self._generate_sequence()
        
        # 计算所有位点的空间坐标
        self.positions = self._calculate_positions()
        
    def _generate_sequence(self):
        """生成随机 DNA 序列"""
        bases = ['A', 'T', 'G', 'C']
        sequence = []
        for i in range(self.n_bases):
            if i % 2 == 0:
                b1 = np.random.choice(bases)
                if b1 == 'A':
                    b2 = 'T'
                elif b1 == 'T':
                    b2 = 'A'
                elif b1 == 'G':
                    b2 = 'C'
                else:
                    b2 = 'G'
            else:
                b2 = np.random.choice(bases)
                if b2 == 'A':
                    b1 = 'T'
                elif b2 == 'T':
                    b1 = 'A'
                elif b2 == 'G':
                    b1 = 'C'
                else:
                    b1 = 'G'
            sequence.append(DNABasePair(b1, b2))
        return sequence
    
    def _calculate_positions(self):
        """计算所有原子的三维坐标"""
        positions_chain1 = []
        positions_chain2 = []
        
        for i in range(self.n_bases):
            z = i * self.rise
            theta = i * self.twist
            
            # 链 1 的坐标
            x1 = self.radius * np.cos(theta)
            y1 = self.radius * np.sin(theta)
            positions_chain1.append(np.array([x1, y1, z]))
            
            # 链 2 的坐标 (相位差π)
            x2 = self.radius * np.cos(theta + np.pi)
            y2 = self.radius * np.sin(theta + np.pi)
            positions_chain2.append(np.array([x2, y2, z]))
            
        return {'chain1': positions_chain1, 'chain2': positions_chain2}
    
    def get_coordinates(self):
        """返回所有坐标用于可视化"""
        chain1 = np.array(self.positions['chain1'])
        chain2 = np.array(self.positions['chain2'])
        return chain1, chain2


class TightBindingHamiltonian:
    """紧束缚哈密顿量构造器"""
    
    def __init__(self, dna_builder, t_intra=0.1, t_inter=0.05, t_stack=0.08):
        """
        参数:
            dna_builder: DNAHelixBuilder 实例
            t_intra: 链内跃迁积分 (eV)
            t_inter: 链间跃迁积分 (碱基配对，eV)
            t_stack: π-π堆叠跃迁积分 (eV)
        """
        self.dna = dna_builder
        self.n_sites = 2 * dna_builder.n_bases  # 两条链
        self.t_intra = t_intra
        self.t_inter = t_inter
        self.t_stack = t_stack
        
        # 构造哈密顿量矩阵
        self.H = self._build_hamiltonian()
        
    def _build_hamiltonian(self):
        """构建紧束缚哈密顿量矩阵"""
        H = np.zeros((self.n_sites, self.n_sites), dtype=complex)
        n = self.dna.n_bases
        
        for i in range(n):
            # 对角项 - 位点能量
            base_pair = self.dna.sequence[i]
            H[i, i] = base_pair.energy1  # 链 1
            H[i + n, i + n] = base_pair.energy2  # 链 2
            
            # 链间耦合 (碱基配对)
            H[i, i + n] = -self.t_inter
            H[i + n, i] = -self.t_inter
            
            # 链内耦合 (相邻碱基)
            if i < n - 1:
                H[i, i + 1] = -self.t_intra
                H[i + 1, i] = -self.t_intra
                H[i + n, i + n + 1] = -self.t_intra
                H[i + n + 1, i + n] = -self.t_intra
                
        return H
    
    def get_eigenstates(self):
        """计算本征态和本征值"""
        eigenvalues, eigenvectors = np.linalg.eigh(self.H)
        return eigenvalues, eigenvectors


class NEGFTransport:
    """非平衡格林函数输运计算器"""
    
    def __init__(self, hamiltonian, n_lead_sites=5):
        """
        参数:
            hamiltonian: TightBindingHamiltonian 实例
            n_lead_sites: 引线区域格点数
        """
        self.H = hamiltonian.H
        self.n_sites = hamiltonian.n_sites
        self.n_lead = n_lead_sites
        
        # 定义引线耦合区域
        self.left_region = list(range(n_lead_sites))
        self.right_region = list(range(self.n_sites - n_lead_sites, self.n_sites))
        
        # 构造引线自能
        self.gamma_L = self._construct_coupling(self.left_region)
        self.gamma_R = self._construct_coupling(self.right_region)
        
    def _construct_coupling(self, region):
        """构造引线 - 样品耦合矩阵"""
        Gamma = np.zeros_like(self.H)
        coupling_strength = 0.1
        
        for idx in region:
            Gamma[idx, idx] = coupling_strength
            
        return Gamma
    
    def retarded_green(self, energy):
        """计算推迟格林函数 G^r(E)"""
        E = energy * np.eye(self.n_sites)
        Sigma_L = -0.5j * self.gamma_L
        Sigma_R = -0.5j * self.gamma_R
        
        G_r = np.linalg.inv(E - self.H - Sigma_L - Sigma_R)
        return G_r
    
    def transmission(self, energy):
        """计算透射系数 T(E)"""
        G_r = self.retarded_green(energy)
        G_a = G_r.conj().T  # 超前格林函数
        
        # Fisher-Lee 关系
        T_matrix = self.gamma_L @ G_r @ self.gamma_R @ G_a
        T = np.real(np.trace(T_matrix))
        
        return T
    
    def calculate_transmission_spectrum(self, energy_range=(-1.0, 1.0), n_points=200):
        """计算透射谱"""
        energies = np.linspace(energy_range[0], energy_range[1], n_points)
        transmission = []
        
        for E in energies:
            try:
                T = self.transmission(E)
                transmission.append(T)
            except:
                transmission.append(0.0)
                
        return energies, np.array(transmission)
    
    def calculate_current(self, voltage, temperature=300):
        """
        计算电流 - 电压特性
        使用 Landauer-Büttiker 公式的简化版本
        """
        # 使用数值积分的简化方法，避免 scipy.integrate.quad 的高计算成本
        n_energy_points = 50
        E_min, E_max = -1.5, 1.5
        energies = np.linspace(E_min, E_max, n_energy_points)
        
        integral = 0.0
        dE = (E_max - E_min) / (n_energy_points - 1)
        
        for E in energies:
            T_E = self.transmission(E)
            f_L = 1.0 / (1.0 + np.exp((E - voltage/2) / (kB * temperature + 1e-10)))
            f_R = 1.0 / (1.0 + np.exp((E + voltage/2) / (kB * temperature + 1e-10)))
            integral += T_E * (f_L - f_R) * dE
        
        # I = (2e/h) * ∫ T(E)[f_L(E) - f_R(E)] dE
        G0 = 2 * e_charge**2 / hbar  # 电导量子
        current = integral * G0 / (2 * np.pi)
        
        return current
    
    def iv_characteristic(self, voltage_range=(-1.0, 1.0), n_points=50, temperature=300):
        """计算 I-V 特性曲线"""
        voltages = np.linspace(voltage_range[0], voltage_range[1], n_points)
        currents = []
        
        print("计算 I-V 特性...")
        for V in voltages:
            I = self.calculate_current(V, temperature)
            currents.append(I)
            print(f"  V = {V:.2f} V, I = {I*1e9:.4f} nA", end='\r')
            
        print()
        return voltages, np.array(currents)


class WavePacketPropagator:
    """波包传播模拟器"""
    
    def __init__(self, hamiltonian):
        self.H = hamiltonian.H
        self.n_sites = hamiltonian.n_sites
        self.dt = 0.1  # 时间步长 (fs)
        
    def initial_wavepacket(self, site=0, width=3):
        """构造初始高斯波包"""
        psi0 = np.zeros(self.n_sites, dtype=complex)
        
        for i in range(self.n_sites):
            psi0[i] = np.exp(-(i - site)**2 / (2 * width**2))
            psi0[i] *= np.exp(1j * 0.5 * i)  # 添加初始动量
            
        # 归一化
        psi0 /= np.sqrt(np.sum(np.abs(psi0)**2))
        return psi0
    
    def propagate(self, psi0, n_steps=100):
        """
        使用时间演化算符传播波包
        ψ(t+dt) = exp(-iH·dt/ℏ) ψ(t)
        使用 Crank-Nicolson 方法
        """
        psi = psi0.copy()
        history = [psi.copy()]
        
        # Crank-Nicolson 算符
        U = np.linalg.inv(np.eye(self.n_sites) + 0.5j * self.H * self.dt) @ \
            (np.eye(self.n_sites) - 0.5j * self.H * self.dt)
        
        for step in range(n_steps):
            psi = U @ psi
            history.append(psi.copy())
            
        return np.array(history)
    
    def plot_propagation(self, history, dna_coords, save_path='wavepacket_propagation.png'):
        """绘制波包传播过程"""
        chain1, chain2 = dna_coords
        n_frames = min(len(history), 20)
        frame_step = max(1, len(history) // n_frames)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # 选择几个时间点展示
        times_to_show = [0, len(history)//3, 2*len(history)//3, len(history)-1]
        
        for idx, t_idx in enumerate(times_to_show):
            if idx >= 4:
                break
                
            ax = axes[idx]
            psi_t = history[t_idx]
            
            # 计算概率密度
            prob = np.abs(psi_t)**2
            prob_chain1 = prob[:self.n_sites//2]
            prob_chain2 = prob[self.n_sites//2:]
            
            # 绘制 DNA 骨架
            ax.plot(chain1[:, 2], chain1[:, 0], 'r-', alpha=0.3, linewidth=1, label='Chain 1')
            ax.plot(chain2[:, 2], chain2[:, 0], 'b-', alpha=0.3, linewidth=1, label='Chain 2')
            
            # 绘制概率分布
            ax.fill_between(chain1[:, 2], prob_chain1 * 2, alpha=0.6, color='red')
            ax.fill_between(chain2[:, 2], prob_chain2 * 2, alpha=0.6, color='blue')
            
            ax.set_xlabel('Z (nm)', fontsize=10)
            ax.set_ylabel('X (nm) / Probability', fontsize=10)
            ax.set_title(f't = {t_idx * self.dt:.1f} fs', fontsize=12)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"波包传播图已保存为 '{save_path}'")
        plt.close()


def visualize_dna_structure(dna_builder, save_path='dna_helix_structure.png'):
    """可视化 DNA 双螺旋结构"""
    chain1, chain2 = dna_builder.get_coordinates()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制两条链
    ax.scatter(chain1[:, 0], chain1[:, 1], chain1[:, 2], 
               c=chain1[:, 2], cmap='Reds', s=80, label='Chain 1 (5\'→3\')', 
               depthshade=True, edgecolors='darkred')
    ax.scatter(chain2[:, 0], chain2[:, 1], chain2[:, 2], 
               c=chain2[:, 2], cmap='Blues', s=80, label='Chain 2 (3\'→5\')', 
               depthshade=True, edgecolors='darkblue')
    
    # 绘制螺旋线
    ax.plot(chain1[:, 0], chain1[:, 1], chain1[:, 2], 'r-', alpha=0.4, linewidth=1.5)
    ax.plot(chain2[:, 0], chain2[:, 1], chain2[:, 2], 'b-', alpha=0.4, linewidth=1.5)
    
    # 绘制碱基对连接
    for i in range(len(chain1)):
        ax.plot([chain1[i, 0], chain2[i, 0]], 
                [chain1[i, 1], chain2[i, 1]], 
                [chain1[i, 2], chain2[i, 2]], 
                'gray', alpha=0.3, linewidth=0.8)
    
    # 标注序列信息
    for i in range(min(5, len(dna_builder.sequence))):
        base_pair = dna_builder.sequence[i]
        ax.text(chain1[i, 0]*1.2, chain1[i, 1]*1.2, chain1[i, 2], 
                f'{base_pair}', fontsize=8, color='darkred')
    
    ax.set_xlabel('X (nm)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (nm)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (nm)', fontsize=12, labelpad=10)
    ax.set_title('DNA Double Helix Structure\n(Tight-Binding Model)', fontsize=14, pad=20)
    ax.legend(loc='upper left', fontsize=10)
    
    # 设置视角
    ax.view_init(elev=25, azim=-60)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"DNA 结构图已保存为 '{save_path}'")
    plt.close()


def plot_transmission_spectrum(energies, transmission, save_path='transmission_spectrum.png'):
    """绘制透射谱"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 透射谱
    ax1.plot(energies, transmission, 'b-', linewidth=2, label='Transmission T(E)')
    ax1.fill_between(energies, transmission, alpha=0.3, color='blue')
    ax1.set_xlabel('Energy (eV)', fontsize=12)
    ax1.set_ylabel('Transmission Coefficient', fontsize=12)
    ax1.set_title('Electron Transmission Spectrum', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax1.legend(fontsize=10)
    
    # 透射系数的统计分布
    ax2.hist(transmission[transmission > 0.01], bins=30, color='green', 
             alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Transmission Coefficient', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Transmission Values', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"透射谱图已保存为 '{save_path}'")
    plt.close()


def plot_iv_curve(voltages, currents, temperature=300, save_path='iv_characteristic.png'):
    """绘制 I-V 特性曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # I-V 曲线
    ax1.plot(voltages, currents * 1e9, 'ro-', linewidth=2, markersize=4, label=f'T = {temperature} K')
    ax1.set_xlabel('Voltage (V)', fontsize=12)
    ax1.set_ylabel('Current (nA)', fontsize=12)
    ax1.set_title('Current-Voltage Characteristic', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 微分电导
    dI_dV = np.gradient(currents, voltages)
    ax2.plot(voltages, dI_dV * 1e9, 'g-', linewidth=2, label='Differential Conductance')
    ax2.fill_between(voltages, dI_dV * 1e9, alpha=0.3, color='green')
    ax2.set_xlabel('Voltage (V)', fontsize=12)
    ax2.set_ylabel('dI/dV (nA/V)', fontsize=12)
    ax2.set_title('Differential Conductance', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"I-V 特性曲线已保存为 '{save_path}'")
    plt.close()


def main():
    """主程序"""
    print("=" * 70)
    print("DNA 双螺旋量子输运模拟系统")
    print("=" * 70)
    print()
    
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    # ========== 1. 构建 DNA 结构 ==========
    print("[1/5] 构建 DNA 双螺旋结构...")
    n_bases = 30
    dna_builder = DNAHelixBuilder(
        n_bases=n_bases,
        radius=1.8,      # nm
        rise=0.34,       # nm
        twist=36.0       # 度
    )
    print(f"   ✓ 碱基对数量：{n_bases}")
    print(f"   ✓ 螺旋半径：{dna_builder.radius} nm")
    print(f"   ✓ 螺距：{360/dna_builder.twist*180/np.pi:.1f} 碱基对/圈")
    print()
    
    # 可视化结构
    visualize_dna_structure(dna_builder)
    print()
    
    # ========== 2. 构建紧束缚哈密顿量 ==========
    print("[2/5] 构建紧束缚哈密顿量...")
    hamiltonian = TightBindingHamiltonian(
        dna_builder,
        t_intra=0.10,    # eV
        t_inter=0.05,    # eV
        t_stack=0.08     # eV
    )
    print(f"   ✓ 系统维度：{hamiltonian.n_sites} × {hamiltonian.n_sites}")
    print(f"   ✓ 链内跃迁：{hamiltonian.t_intra} eV")
    print(f"   ✓ 链间跃迁：{hamiltonian.t_inter} eV")
    print()
    
    # 计算本征态
    eigenvalues, eigenvectors = hamiltonian.get_eigenstates()
    print(f"   ✓ 能量范围：[{eigenvalues.min():.3f}, {eigenvalues.max():.3f}] eV")
    print()
    
    # ========== 3. NEGF 输运计算 ==========
    print("[3/5] 计算量子输运性质 (NEGF 方法)...")
    transport = NEGFTransport(hamiltonian, n_lead_sites=5)
    
    # 透射谱
    energies, transmission = transport.calculate_transmission_spectrum(
        energy_range=(-0.8, 0.8), 
        n_points=200
    )
    plot_transmission_spectrum(energies, transmission)
    
    valid_T = transmission[transmission > 0.001]
    if len(valid_T) > 0:
        print(f"   ✓ 最大透射系数：{valid_T.max():.4f}")
        print(f"   ✓ 平均透射系数：{valid_T.mean():.4f}")
    print()
    
    # ========== 4. I-V 特性 ==========
    print("[4/5] 计算电流 - 电压特性...")
    temperature = 300  # K
    voltages, currents = transport.iv_characteristic(
        voltage_range=(-0.5, 0.5),
        n_points=30,
        temperature=temperature
    )
    plot_iv_curve(voltages, currents, temperature)
    print(f"   ✓ 温度：{temperature} K")
    print(f"   ✓ 最大电流：{currents.max()*1e9:.4f} nA")
    print()
    
    # ========== 5. 波包传播模拟 ==========
    print("[5/5] 模拟电子波包传播...")
    propagator = WavePacketPropagator(hamiltonian)
    psi0 = propagator.initial_wavepacket(site=3, width=4)
    history = propagator.propagate(psi0, n_steps=150)
    propagator.plot_propagation(history, dna_builder.get_coordinates())
    print(f"   ✓ 传播时间：{len(history) * propagator.dt:.1f} fs")
    print(f"   ✓ 时间步长：{propagator.dt} fs")
    print()
    
    # ========== 总结 ==========
    print("=" * 70)
    print("模拟完成！生成的文件:")
    print("  1. dna_helix_structure.png    - DNA 双螺旋三维结构")
    print("  2. transmission_spectrum.png  - 电子透射谱")
    print("  3. iv_characteristic.png      - I-V 特性曲线")
    print("  4. wavepacket_propagation.png - 波包传播过程")
    print("=" * 70)
    
    # 绘制能级图
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(eigenvalues))
    colors = np.abs(eigenvectors[0, :])**2  # 用第一个位点的权重着色
    
    scatter = ax.scatter(eigenvalues, y_pos, c=colors, cmap='viridis', 
                         s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('State Index', fontsize=12)
    ax.set_title('Energy Spectrum of DNA System', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('|ψ(site=0)|²', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('energy_spectrum.png', dpi=150)
    print("  5. energy_spectrum.png        - 能级结构图")
    plt.close()
    
    print()
    print("所有结果已保存至当前目录。")


if __name__ == "__main__":
    main()
