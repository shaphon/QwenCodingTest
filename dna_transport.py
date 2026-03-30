import kwant
import numpy as np
import matplotlib.pyplot as plt

def make_dna_system(length=40, radius=2.0, pitch=1.5, t_intra=1.0, t_inter=0.5):
    """
    构建DNA双螺旋紧束缚模型
    包含直线引线区和螺旋散射区
    """
    # 使用一般晶格定义3D位置 (注意是小写 general)
    lat = kwant.lattice.general(np.identity(3))
    
    # 在螺旋两端添加直线区以连接引线
    straight_len = 5
    total_len = length + 2 * straight_len
    
    syst = kwant.Builder()
    
    # 存储所有站点以便连接
    sites = []
    
    # 构建包含直线-螺旋-直线的完整系统
    for i in range(total_len):
        z = float(i)
        
        if i < straight_len or i >= total_len - straight_len:
            # 直线区域 (引线部分)
            theta = 0 if i < straight_len else np.pi  # 简单处理，实际应平滑过渡
            x_a = radius
            y_a = 0.0
            x_b = -radius
            y_b = 0.0
        else:
            # 螺旋区域
            theta = (i - straight_len) * pitch
            x_a = radius * np.cos(theta)
            y_a = radius * np.sin(theta)
            x_b = radius * np.cos(theta + np.pi)
            y_b = radius * np.sin(theta + np.pi)
            
        # 创建两个链的位点 (A链和B链)
        site_a = lat(x_a, y_a, z)
        site_b = lat(x_b, y_b, z)
        
        # 设置位点能量 (对角项)
        syst[site_a] = 0.0
        syst[site_b] = 0.0
        
        sites.append((site_a, site_b))
        
        # 链内跃迁 (连接到前一个位点)
        if i > 0:
            prev_a, prev_b = sites[i-1]
            syst[site_a, prev_a] = -t_intra  # A链内
            syst[site_b, prev_b] = -t_intra  # B链内
            
        # 链间跃迁 (碱基对配对)
        syst[site_a, site_b] = -t_inter

    # 构建左引线 (沿 -z 方向平移对称)
    lead_sym = kwant.TranslationalSymmetry((0, 0, -1.0))
    lead_l = kwant.Builder(lead_sym)
    
    # 引线单元胞：两个位点 (A和B链)
    pos_a_lead = (radius, 0.0, 0.0)
    pos_b_lead = (-radius, 0.0, 0.0)
    
    site_a0 = lat(*pos_a_lead)
    site_b0 = lat(*pos_b_lead)
    
    lead_l[site_a0] = 0.0
    lead_l[site_b0] = 0.0
    lead_l[site_a0, site_b0] = -t_inter  # 链间耦合
    
    # 连接到下一个单元胞 (沿 -z)
    site_a1 = lat(radius, 0.0, -1.0)
    site_b1 = lat(-radius, 0.0, -1.0)
    
    lead_l[site_a0, site_a1] = -t_intra
    lead_l[site_b0, site_b1] = -t_intra
    
    # 构建右引线 (沿 +z 方向平移对称)
    lead_sym_r = kwant.TranslationalSymmetry((0, 0, 1.0))
    lead_r = kwant.Builder(lead_sym_r)
    
    z_end = float(total_len - 1)
    site_a_end = lat(radius, 0.0, z_end)
    site_b_end = lat(-radius, 0.0, z_end)
    
    lead_r[site_a_end] = 0.0
    lead_r[site_b_end] = 0.0
    lead_r[site_a_end, site_b_end] = -t_inter
    
    site_a_next = lat(radius, 0.0, z_end + 1.0)
    site_b_next = lat(-radius, 0.0, z_end + 1.0)
    
    lead_r[site_a_end, site_a_next] = -t_intra
    lead_r[site_b_end, site_b_next] = -t_intra

    # 附加引线到散射区
    syst.attach_lead(lead_l)
    syst.attach_lead(lead_r)
    
    return syst.finalized()

def plot_dna_structure(syst):
    """绘制DNA结构"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取所有位点坐标
    xs_a, ys_a, zs_a = [], [], []
    xs_b, ys_b, zs_b = [], [], []
    
    for site in syst.sites:
        x, y, z = site.pos
        # 根据位置判断是A链还是B链 (简化判断)
        if x > 0:
            xs_a.append(x)
            ys_a.append(y)
            zs_a.append(z)
        else:
            xs_b.append(x)
            ys_b.append(y)
            zs_b.append(z)
    
    ax.scatter(xs_a, ys_a, zs_a, c='red', s=50, label='Chain A', depthshade=True)
    ax.scatter(xs_b, ys_b, zs_b, c='blue', s=50, label='Chain B', depthshade=True)
    
    # 绘制连线 (跃迁) - hoppings 返回的是元组 (site_i, site_j, value)
    for hop in syst.hoppings:
        # hop 是 (site_i, site_j, hopping_value) 或者只是 (site_i, site_j)
        if hasattr(hop, '__len__') and len(hop) >= 2:
            site_i, site_j = hop[0], hop[1]
            # 获取位点位置
            p1 = site_i.pos if hasattr(site_i, 'pos') else None
            p2 = site_j.pos if hasattr(site_j, 'pos') else None
            if p1 and p2:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'gray-', linewidth=0.5, alpha=0.5)
    
    ax.set_title("DNA Double Helix Tight-Binding Model")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (Propagation Direction)")
    ax.legend()
    ax.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    plt.savefig("dna_structure.png", dpi=150)
    print("DNA结构图已保存为 'dna_structure.png'")
    plt.close()

def calculate_transmission(syst):
    """计算透射系数随能量的变化"""
    energies = np.linspace(-2.5, 2.5, 150)
    trans = []
    
    print("正在计算透射谱...")
    for E in energies:
        try:
            smatrix = kwant.smatrix(syst, energy=E)
            t_val = smatrix.transmission(1, 0)  # 从左(0)到右(1)
            trans.append(t_val)
        except Exception as e:
            print(f"Warning at E={E:.2f}: {e}")
            trans.append(np.nan)
    
    plt.figure(figsize=(8, 5))
    plt.plot(energies, trans, 'b-', linewidth=2, label='Transmission')
    plt.title("Electron Transmission through DNA Wire")
    plt.xlabel("Energy (units of t)")
    plt.ylabel("Transmission T(E)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("dna_transmission.png", dpi=150)
    print("透射谱图已保存为 'dna_transmission.png'")
    plt.close()
    
    return energies, trans

def main():
    print("="*50)
    print("DNA Double Helix Transport Simulation")
    print("="*50)
    
    # 参数设置
    length = 30       # 螺旋区长度
    radius = 2.0      # 螺旋半径
    pitch = 0.8       # 螺距参数
    t_intra = 1.0     # 链内跃迁强度
    t_inter = 0.4     # 链间跃迁强度 (碱基配对)
    
    print(f"\n构建系统参数:")
    print(f"  螺旋长度: {length} 单位")
    print(f"  半径: {radius}")
    print(f"  链内跃迁: {t_intra}")
    print(f"  链间跃迁: {t_inter}")
    
    # 构建系统
    print("\n正在构建DNA双螺旋系统...")
    syst = make_dna_system(
        length=length, 
        radius=radius, 
        pitch=pitch, 
        t_intra=t_intra, 
        t_inter=t_inter
    )
    
    print(f"系统构建完成!")
    print(f"  总格点数: {len(syst.sites)}")
    print(f"  引数量: {len(syst.leads)}")
    
    # 可视化结构
    print("\n正在生成结构图...")
    plot_dna_structure(syst)
    
    # 计算输运性质
    print("\n正在计算电子输运...")
    energies, trans = calculate_transmission(syst)
    
    # 输出一些统计信息
    valid_trans = [t for t in trans if not np.isnan(t)]
    if valid_trans:
        max_t = max(valid_trans)
        avg_t = np.mean(valid_trans)
        print(f"\n结果统计:")
        print(f"  最大透射系数: {max_t:.4f}")
        print(f"  平均透射系数: {avg_t:.4f}")
    
    print("\n" + "="*50)
    print("模拟完成！请查看生成的图片文件:")
    print("  1. dna_structure.png - DNA双螺旋几何结构")
    print("  2. dna_transmission.png - 电子透射系数谱")
    print("="*50)

if __name__ == "__main__":
    main()
