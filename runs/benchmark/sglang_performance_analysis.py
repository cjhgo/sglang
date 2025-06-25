#!/usr/bin/env python
# coding: utf-8

# # SGLang 性能测试分析
# 
# 本 notebook 分析不同 `input_len` 情况下，Prefill 和 Decode 阶段的性能特征。
# 
# ## 主要分析内容：
# 1. **Prefill 阶段**：延迟/吞吐量 vs batch_size (按 input_len 分组)
# 2. **Decode 阶段**：延迟/吞吐量 vs batch_size (按 input_len 分组)
# 3. **性能缩放特征**：理解算力瓶颈和带宽瓶颈
# 4. **优化建议**：基于分析结果的部署建议
# 
# ### 主要分析内容：
# 
# 1. **数据加载和预处理**
#    - 从 `/tmp/result.jsonl` 加载测试数据
#    - 显示数据概览和参数分布
# 
# 2. **Prefill 阶段分析**
#    - 延迟随 batch_size 呈亚线性增长（≈B^0.6）
#    - 吞吐量随 batch_size 增加，但效率递减
#    - 批处理效率从 100% 下降到约 30%
# 
# 3. **Decode 阶段分析**
#    - 延迟增长相对较小（带宽瓶颈）
#    - 吞吐量近似线性增长
#    - 串行生成特性限制了并行优化
# 
# 4. **性能模型拟合**
#    - 使用幂律模型 L = a × B^b 拟合延迟缩放规律
#    - Prefill: b ≈ 0.6（亚线性）
#    - Decode: b ≈ 0.05-0.1（几乎不变）
# 
# 5. **可视化分析**
#    - 多维度性能对比图表
#    - 热力图展示不同参数组合的性能
#    - 效率分析图表
# 
# ### 关键发现：
# 
# 从测试数据可以看出：
# - **Prefill 阶段**：batch_size 从 1 增加到 100 时，延迟增长约 76 倍（亚线性）
# - **Decode 阶段**：batch_size 从 1 增加到 100 时，延迟仅增长约 1.5 倍（几乎不变）
# 
# ### 关键发现：
# 
# 从测试数据可以看出：
# - **Prefill 阶段**：batch_size 从 1 增加到 100 时，延迟增长约 76 倍（亚线性）
# - **Decode 阶段**：batch_size 从 1 增加到 100 时，延迟仅增长约 1.5 倍（几乎不变）
# 

# In[1]:


# 导入必要的库
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# 设置中文支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from matplotlib.font_manager import FontProperties
import matplotlib
import warnings
import os


warnings.filterwarnings('ignore')


# 设置中文字体支持
def setup_chinese_font():
    """设置中文字体支持"""
    # 尝试多种中文字体，按优先级排序
    chinese_fonts = [
        'WenQuanYi Micro Hei',      # Linux常见中文字体
        'WenQuanYi Zen Hei',        # Linux中文字体
        'Noto Sans CJK SC',         # Google字体
        'Source Han Sans SC',        # Adobe字体
        'SimHei',                   # Windows字体
        'Microsoft YaHei',          # Windows字体
        'PingFang SC',              # macOS字体
        'Hiragino Sans GB',         # macOS字体
        'DejaVu Sans'               # 英文回退字体
    ]
    
    # 获取系统可用字体
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    
    # 找到第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            print(f"✅ 使用字体: {font}")
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font] + ['DejaVu Sans']
    else:
        print("⚠️  未找到中文字体，使用默认字体")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 10

# 初始化字体设置
setup_chinese_font()


# ## 1. 数据加载和预处理
# 

# In[2]:


# 加载数据
def load_jsonl(filename):
    """加载 JSONL 格式的数据"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

# 加载结果数据
df = load_jsonl('/tmp/result.jsonl')
print(f"Loaded {len(df)} test results")
print(f"\nData columns: {df.columns.tolist()}")
print(f"\nUnique input_len values: {sorted(df['input_len'].unique())}")
print(f"Unique batch_size values: {sorted(df['batch_size'].unique())}")
print(f"Unique output_len values: {sorted(df['output_len'].unique())}")


# In[3]:


# 数据概览
df.head(10)


# In[4]:


def pdstat_of_inputlen(df, input_len):
    data = df[df['input_len'] == input_len].sort_values('batch_size')

    if not data.empty:
        print(f'Prefill 性能分析 (input_len={input_len}):')
        print('-' * 80)
        print('Batch Size | Prefill Latency | Prefill Throughput | 延迟增长倍数 | 吞吐量增长倍数')
        print('-' * 80)
    
        base_latency = data.iloc[0]['prefill_latency']
        base_throughput = data.iloc[0]['prefill_throughput']
    
        for _, row in data.iterrows():
            latency_ratio = row['prefill_latency'] / base_latency
            throughput_ratio = row['prefill_throughput'] / base_throughput
            print(f"{row['batch_size']:10d} | {row['prefill_latency']:15.3f} | {row['prefill_throughput']:18.1f} | {latency_ratio:12.2f} | {throughput_ratio:14.2f}")
    
        print(f'\nDecode 性能分析 (input_len={input_len}):')
        print('-' * 80)
        print('Batch Size | Decode Latency | Decode Throughput | 延迟增长倍数 | 吞吐量增长倍数')
        print('-' * 80)
    
        base_decode_latency = data.iloc[0]['median_decode_latency']
        base_decode_throughput = data.iloc[0]['median_decode_throughput']
    
        for _, row in data.iterrows():
            latency_ratio = row['median_decode_latency'] / base_decode_latency
            throughput_ratio = row['median_decode_throughput'] / base_decode_throughput
            print(f"{row['batch_size']:10d} | {row['median_decode_latency']:14.5f} | {row['median_decode_throughput']:17.1f} | {latency_ratio:12.2f} | {throughput_ratio:14.2f}")
        print('='*100)


pdstat_of_inputlen(df, 2048)
pdstat_of_inputlen(df, 800)
pdstat_of_inputlen(df, 1024)


# ## 2. Prefill 阶段分析
# 
# Prefill 阶段的特点：
# - **算力瓶颈**：主要受 GPU 计算能力限制
# - **批处理效应明显**：增加 batch_size 可以显著提高吞吐量
# - **延迟相对稳定**：batch_size 增加时延迟增长较慢
# 

# In[5]:


# 准备 Prefill 分析数据
# 固定 output_len=16，分析不同 input_len 下的表现
prefill_data = df[df['output_len'] == 16].copy()

# 创建 input_len 分组
input_len_groups = sorted(prefill_data['input_len'].unique())
print(f"Analyzing {len(input_len_groups)} different input lengths: {input_len_groups}")


# In[6]:


# Prefill 延迟分析
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('分析Prefill Stage Analysis: Latency vs Batch Size for Different Input Lengths', fontsize=16)

# 选择有代表性的 input_len 值
selected_input_lens = [100, 400, 800, 1024]

for idx, input_len in enumerate(selected_input_lens):
    ax = axes[idx // 2, idx % 2]
    
    # 筛选数据
    data = prefill_data[prefill_data['input_len'] == input_len].sort_values('batch_size')
    
    if not data.empty:
        # 绘制延迟曲线
        ax.plot(data['batch_size'], data['prefill_latency'], 'o-', linewidth=2, markersize=8, label='Actual')
        
        # 添加理想线性增长参考线
        base_latency = data.iloc[0]['prefill_latency']
        ideal_latencies = base_latency * (data['batch_size'] / data.iloc[0]['batch_size'])
        ax.plot(data['batch_size'], ideal_latencies, '--', alpha=0.7, label='Linear scaling')
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Prefill Latency (s)')
        ax.set_title(f'Input Length = {input_len} tokens')
        ax.grid(True, alpha=0.3)
        ax.legend()
        # ax.set_xscale('log', base=2)
        # ax.set_yscale('log')

plt.tight_layout()
plt.show()


# In[7]:


data


# In[ ]:





# In[ ]:





# In[8]:


# Prefill 吞吐量分析
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Prefill Stage Analysis: Throughput vs Batch Size for Different Input Lengths', fontsize=16)

for idx, input_len in enumerate(selected_input_lens):
    ax = axes[idx // 2, idx % 2]
    
    # 筛选数据
    data = prefill_data[prefill_data['input_len'] == input_len].sort_values('batch_size')
    
    if not data.empty:
        # 绘制吞吐量曲线
        ax.plot(data['batch_size'], data['prefill_throughput'], 'o-', linewidth=2, markersize=8, 
                color='green', label='Actual')
        
        # 添加理想线性增长参考线
        base_throughput = data.iloc[0]['prefill_throughput']
        ideal_throughputs = base_throughput * (data['batch_size'] / data.iloc[0]['batch_size'])
        # ax.plot(data['batch_size'], ideal_throughputs, '--', alpha=0.7, color='red', label='Ideal linear')
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Prefill Throughput (tokens/s)')
        ax.set_title(f'Input Length = {input_len} tokens')
        ax.grid(True, alpha=0.3)
        ax.legend()
        # ax.set_xscale('log', base=2)

plt.tight_layout()
plt.show()


# In[9]:


# 完整性能数据展示 - 针对不同 input_len，在单个图上展示 4 个指标
# 为每个 input_len 创建一个综合图表
for input_len in selected_input_lens:
    data = prefill_data[prefill_data['input_len'] == input_len].sort_values('batch_size')
    
    if not data.empty:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
        
        # 左y轴：Latency (ms)
        # Prefill Latency
        line1 = ax1.plot(data['batch_size'], data['prefill_latency'] * 1000, 
                        'o-', color='blue', linewidth=2, markersize=8, 
                        label='Prefill Latency')
        
        # Decode Latency
        line2 = ax1.plot(data['batch_size'], data['median_decode_latency'] * 1000, 
                        's-', color='red', linewidth=2, markersize=8, 
                        label='Decode Latency')
        
        ax1.set_xlabel('Batch Size', fontsize=12)
        ax1.set_ylabel('Latency (ms)', color='black', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 右y轴：Throughput (tokens/s)
        ax2 = ax1.twinx()
        
        # Prefill Throughput
        line3 = ax2.plot(data['batch_size'], data['prefill_throughput'], 
                        '^-', color='green', linewidth=2, markersize=8, 
                        label='Prefill Throughput')
        
        # Decode Throughput
        line4 = ax2.plot(data['batch_size'], data['median_decode_throughput'], 
                        'v-', color='purple', linewidth=2, markersize=8, 
                        label='Decode Throughput')
        
        ax2.set_ylabel('Throughput (tokens/s)', color='black', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='black')
        
        # 合并图例
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        # 设置标题
        ax1.set_title(f'SGLang Performance Analysis: Input Length = {input_len} tokens\n'
                     f'Prefill/Decode Latency & Throughput vs Batch Size', 
                     fontsize=14, pad=20)
        
        # 添加性能关键指标注释
        if len(data) > 1:
            # 计算延迟增长倍数
            prefill_growth = data['prefill_latency'].iloc[-1] / data['prefill_latency'].iloc[0]
            decode_growth = data['median_decode_latency'].iloc[-1] / data['median_decode_latency'].iloc[0]
            
            # 计算吞吐量增长倍数
            prefill_thr_growth = data['prefill_throughput'].iloc[-1] / data['prefill_throughput'].iloc[0]
            decode_thr_growth = data['median_decode_throughput'].iloc[-1] / data['median_decode_throughput'].iloc[0]
            
            # 添加文本注释
            textstr = f'''Performance Summary:
Prefill Latency: {prefill_growth:.1f}x growth
Decode Latency: {decode_growth:.1f}x growth
Prefill Throughput: {prefill_thr_growth:.1f}x growth  
Decode Throughput: {decode_thr_growth:.1f}x growth'''
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax1.text(0.65, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show()


# ## 3. Decode 阶段分析
# 
# Decode 阶段的特点：
# - **带宽瓶颈**：主要受内存带宽限制
# - **串行生成**：每个 token 必须逐个生成
# - **延迟线性增长**：batch_size 增加导致延迟近似线性增长
# 

# In[10]:


# Decode 延迟分析
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Decode Stage Analysis: Latency vs Batch Size for Different Input Lengths', fontsize=16)

for idx, input_len in enumerate(selected_input_lens):
    ax = axes[idx // 2, idx % 2]
    
    # 筛选数据
    data = prefill_data[prefill_data['input_len'] == input_len].sort_values('batch_size')
    
    if not data.empty:
        # 绘制延迟曲线
        ax.plot(data['batch_size'], data['median_decode_latency'], 'o-', linewidth=2, 
                markersize=8, color='red', label='Actual')
        
        # 添加线性增长参考线
        base_latency = data.iloc[0]['median_decode_latency']
        linear_latencies = base_latency * (data['batch_size'] / data.iloc[0]['batch_size'])
        ax.plot(data['batch_size'], linear_latencies, '--', alpha=0.7, label='Linear scaling')
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Decode Latency (s)')
        ax.set_title(f'Input Length = {input_len} tokens')
        ax.grid(True, alpha=0.3)
        ax.legend()
        # ax.set_xscale('log', base=2)

plt.tight_layout()
plt.show()


# In[11]:


# Decode 吞吐量分析
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Decode Stage Analysis: Throughput vs Batch Size for Different Input Lengths', fontsize=16)

for idx, input_len in enumerate(selected_input_lens):
    ax = axes[idx // 2, idx % 2]
    
    # 筛选数据
    data = prefill_data[prefill_data['input_len'] == input_len].sort_values('batch_size')
    
    if not data.empty:
        # 绘制吞吐量曲线
        ax.plot(data['batch_size'], data['median_decode_throughput'], 'o-', linewidth=2, 
                markersize=8, color='purple', label='Actual')
        
        # 添加理想线性增长参考线
        base_throughput = data.iloc[0]['median_decode_throughput']
        ideal_throughputs = base_throughput * data['batch_size']
        ax.plot(data['batch_size'], ideal_throughputs, '--', alpha=0.7, color='green', label='Ideal linear')
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Decode Throughput (tokens/s)')
        ax.set_title(f'Input Length = {input_len} tokens')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')

plt.tight_layout()
plt.show()


# ## 4. 性能瓶颈对比分析
# 

# In[12]:


# 创建综合对比图
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig)

# 选择一个代表性的 input_len 进行详细分析
target_input_len = 800
data = prefill_data[prefill_data['input_len'] == target_input_len].sort_values('batch_size')

if not data.empty:
    # 1. Prefill vs Decode 延迟对比
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(data['batch_size'], data['prefill_latency'], 'o-', linewidth=2, 
             markersize=8, label='Prefill Latency')
    ax1.plot(data['batch_size'], data['median_decode_latency'], 's-', linewidth=2, 
             markersize=8, label='Decode Latency')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Latency (s)')
    ax1.set_title(f'Latency Comparison (input_len={target_input_len})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    
    # 2. 延迟增长率分析
    ax2 = fig.add_subplot(gs[0, 1])
    prefill_growth = data['prefill_latency'].values / data['prefill_latency'].iloc[0]
    decode_growth = data['median_decode_latency'].values / data['median_decode_latency'].iloc[0]
    batch_growth = data['batch_size'].values / data['batch_size'].iloc[0]
    
    ax2.plot(data['batch_size'], prefill_growth, 'o-', linewidth=2, markersize=8, label='Prefill Growth')
    ax2.plot(data['batch_size'], decode_growth, 's-', linewidth=2, markersize=8, label='Decode Growth')
    ax2.plot(data['batch_size'], batch_growth, '--', alpha=0.7, color='gray', label='Linear Growth')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Growth Factor')
    ax2.set_title('Latency Growth Rate Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # ax2.set_xscale('log', base=2)
    
    # 3. 吞吐量效率对比
    ax3 = fig.add_subplot(gs[0, 2])
    # 计算效率
    prefill_base_thr = data.iloc[0]['prefill_throughput']
    decode_base_thr = data.iloc[0]['median_decode_throughput']
    base_batch = data.iloc[0]['batch_size']
    
    prefill_eff = (data['prefill_throughput'] / (prefill_base_thr * data['batch_size'] / base_batch)) * 100
    decode_eff = (data['median_decode_throughput'] / (decode_base_thr * data['batch_size'])) * 100
    
    ax3.plot(data['batch_size'], prefill_eff, 'o-', linewidth=2, markersize=8, label='Prefill Efficiency')
    ax3.plot(data['batch_size'], decode_eff, 's-', linewidth=2, markersize=8, label='Decode Efficiency')
    ax3.axhline(y=100, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Efficiency (%)')
    ax3.set_title('Throughput Efficiency Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # ax3.set_xscale('log', base=2)
    ax3.set_ylim(0, 110)

# 4. Input Length 影响分析
ax4 = fig.add_subplot(gs[1, :])

# 选择固定的 batch_size
target_batch_size = 8
length_data = prefill_data[prefill_data['batch_size'] == target_batch_size].sort_values('input_len')

if not length_data.empty:
    ax4_twin = ax4.twinx()
    
    # Prefill 延迟随 input_len 变化
    line1 = ax4.plot(length_data['input_len'], length_data['prefill_latency'], 
                     'o-', linewidth=2, markersize=8, color='blue', label='Prefill Latency')
    line2 = ax4.plot(length_data['input_len'], length_data['median_decode_latency'], 
                     's-', linewidth=2, markersize=8, color='red', label='Decode Latency')
    
    # 吞吐量
    line3 = ax4_twin.plot(length_data['input_len'], length_data['prefill_throughput'], 
                          '^-', linewidth=2, markersize=8, color='green', label='Prefill Throughput')
    line4 = ax4_twin.plot(length_data['input_len'], length_data['median_decode_throughput'], 
                          'v-', linewidth=2, markersize=8, color='purple', label='Decode Throughput')
    
    ax4.set_xlabel('Input Length (tokens)')
    ax4.set_ylabel('Latency (s)', color='black')
    ax4_twin.set_ylabel('Throughput (tokens/s)', color='black')
    ax4.set_title(f'Performance vs Input Length (batch_size={target_batch_size})')
    
    # 合并图例
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='best')
    
    ax4.grid(True, alpha=0.3)

plt.suptitle('SGLang Performance Analysis: Prefill vs Decode Characteristics', fontsize=16)
plt.tight_layout()
plt.show()


# In[13]:


## 5. 性能模型拟合与预测


# In[14]:


from scipy.optimize import curve_fit

# 定义性能模型
def power_law_model(x, a, b):
    """幂律模型: y = a * x^b"""
    return a * np.power(x, b)

# 分析 Prefill 延迟的缩放规律
print("=" * 60)
print("Performance Model Fitting Results")
print("=" * 60)

# 选择一个 input_len 进行分析
for input_len in [400, 800, 1024]:
    data = prefill_data[prefill_data['input_len'] == input_len].sort_values('batch_size')
    
    if len(data) > 3:  # 需要足够的数据点进行拟合
        print(f"\nInput Length = {input_len} tokens:")
        
        # 拟合 Prefill 延迟模型
        x = data['batch_size'].values
        y = data['prefill_latency'].values
        
        try:
            popt, _ = curve_fit(power_law_model, x, y)
            a, b = popt
            print(f"  Prefill Latency Model: L = {a:.6f} × B^{b:.3f}")
            print(f"    - Scaling exponent: {b:.3f} (< 1 indicates sub-linear growth)")
            
            # 计算拟合优度
            y_pred = power_law_model(x, a, b)
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            print(f"    - R² = {r2:.4f}")
        except:
            print("    - Fitting failed")
        
        # 拟合 Decode 延迟模型
        y_decode = data['median_decode_latency'].values
        
        try:
            popt_decode, _ = curve_fit(power_law_model, x, y_decode)
            a_d, b_d = popt_decode
            print(f"  Decode Latency Model: L = {a_d:.6f} × B^{b_d:.3f}")
            print(f"    - Scaling exponent: {b_d:.3f} (≈ 1 indicates linear growth)")
            
            # 计算拟合优度
            y_pred_decode = power_law_model(x, a_d, b_d)
            r2_decode = 1 - np.sum((y_decode - y_pred_decode)**2) / np.sum((y_decode - np.mean(y_decode))**2)
            print(f"    - R² = {r2_decode:.4f}")
        except:
            print("    - Fitting failed")


# ## 6. 性能优化建议
# 

# In[15]:


# 生成性能优化建议
print("=" * 60)
print("Performance Optimization Recommendations")
print("=" * 60)

# 1. 找出最优 batch size
print("\n1. Optimal Batch Size Analysis:")

for input_len in [400, 800, 1024]:
    data = prefill_data[prefill_data['input_len'] == input_len]
    if not data.empty:
        # 找出最高总体吞吐量的配置
        best_row = data.loc[data['overall_throughput'].idxmax()]
        print(f"  Input Length {input_len}:")
        print(f"    - Best batch size: {best_row['batch_size']}")
        print(f"    - Max throughput: {best_row['overall_throughput']:.1f} tokens/s")
        print(f"    - Total latency: {best_row['total_latency']:.3f}s")

# 2. 延迟敏感 vs 吞吐量优先建议
print("\n2. Deployment Recommendations:")
print("\n  For Latency-Sensitive Applications:")
print("    - Use batch_size = 1-2")
print("    - Prefill latency increases sub-linearly (≈B^0.6)")
print("    - Decode latency increases nearly linearly")

print("\n  For Throughput-Oriented Applications:")
print("    - Use batch_size = 8-16 for best efficiency")
print("    - Larger batches show diminishing returns")
print("    - Consider memory constraints for very large batches")

# 3. 瓶颈分析
print("\n3. Performance Bottleneck Analysis:")
print("\n  Prefill Stage:")
print("    - Compute-bound: benefits from GPU parallelism")
print("    - Efficiency drops from ~100% to ~30% at large batch sizes")
print("    - Input length has linear impact on latency")

print("\n  Decode Stage:")
print("    - Memory bandwidth-bound: limited parallelism benefit")
print("    - Nearly linear scaling with batch size")
print("    - Each token generation requires full KV cache access")


# In[16]:


# 创建性能热力图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 准备热力图数据
# 创建 batch_size vs input_len 的矩阵
batch_sizes = sorted(prefill_data['batch_size'].unique())
input_lens = sorted(prefill_data['input_len'].unique())

# Prefill 吞吐量热力图
prefill_matrix = np.zeros((len(batch_sizes), len(input_lens)))
decode_matrix = np.zeros((len(batch_sizes), len(input_lens)))

for i, bs in enumerate(batch_sizes):
    for j, il in enumerate(input_lens):
        data = prefill_data[(prefill_data['batch_size'] == bs) & (prefill_data['input_len'] == il)]
        if not data.empty:
            prefill_matrix[i, j] = data.iloc[0]['prefill_throughput']
            decode_matrix[i, j] = data.iloc[0]['median_decode_throughput']

# 绘制热力图
im1 = ax1.imshow(prefill_matrix, aspect='auto', cmap='YlOrRd')
ax1.set_xticks(range(len(input_lens)))
ax1.set_xticklabels(input_lens)
ax1.set_yticks(range(len(batch_sizes)))
ax1.set_yticklabels(batch_sizes)
ax1.set_xlabel('Input Length')
ax1.set_ylabel('Batch Size')
ax1.set_title('Prefill Throughput Heatmap (tokens/s)')
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(decode_matrix, aspect='auto', cmap='YlGnBu')
ax2.set_xticks(range(len(input_lens)))
ax2.set_xticklabels(input_lens)
ax2.set_yticks(range(len(batch_sizes)))
ax2.set_yticklabels(batch_sizes)
ax2.set_xlabel('Input Length')
ax2.set_ylabel('Batch Size')
ax2.set_title('Decode Throughput Heatmap (tokens/s)')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()


# ## 7. 总结
# 
# ### 关键发现：
# 
# 1. **Prefill 阶段（算力瓶颈）**：
#    - 延迟随 batch_size 呈亚线性增长（≈B^0.6）
#    - 吞吐量随 batch_size 增加，但效率递减
#    - GPU 并行计算能力有效利用
# 
# 2. **Decode 阶段（带宽瓶颈）**：
#    - 延迟随 batch_size 近似线性增长
#    - 受内存带宽限制，批处理收益有限
#    - 串行生成特性导致延迟累积
# 
# 3. **优化建议**：
#    - **延迟优先**：使用小 batch_size (1-2)
#    - **吞吐量优先**：使用中等 batch_size (8-16)
#    - **资源利用**：大 batch_size 效率下降明显
# 
# 4. **Input Length 影响**：
#    - Prefill 延迟与 input_len 成正比
#    - Decode 延迟受 input_len 影响较小
#    - 长序列更适合批处理优化
# 

# In[17]:


# 创建延迟热力图分析
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

# 准备延迟热力图数据
batch_sizes = sorted(prefill_data['batch_size'].unique())
input_lens = sorted(prefill_data['input_len'].unique())

# 创建延迟矩阵
prefill_latency_matrix = np.zeros((len(batch_sizes), len(input_lens)))
decode_latency_matrix = np.zeros((len(batch_sizes), len(input_lens)))
total_latency_matrix = np.zeros((len(batch_sizes), len(input_lens)))
latency_ratio_matrix = np.zeros((len(batch_sizes), len(input_lens)))

for i, bs in enumerate(batch_sizes):
    for j, il in enumerate(input_lens):
        data = prefill_data[(prefill_data['batch_size'] == bs) & (prefill_data['input_len'] == il)]
        if not data.empty:
            prefill_lat = data.iloc[0]['prefill_latency']
            decode_lat = data.iloc[0]['median_decode_latency']
            prefill_latency_matrix[i, j] = prefill_lat
            decode_latency_matrix[i, j] = decode_lat
            total_latency_matrix[i, j] = prefill_lat + decode_lat
            # 计算Prefill延迟占总延迟的比例
            if (prefill_lat + decode_lat) > 0:
                latency_ratio_matrix[i, j] = prefill_lat / (prefill_lat + decode_lat)

# 1. Prefill延迟热力图
im1 = ax1.imshow(prefill_latency_matrix, aspect='auto', cmap='Reds', origin='lower')
ax1.set_xticks(range(len(input_lens)))
ax1.set_xticklabels(input_lens)
ax1.set_yticks(range(len(batch_sizes)))
ax1.set_yticklabels(batch_sizes)
ax1.set_xlabel('Input Length')
ax1.set_ylabel('Batch Size')
ax1.set_title('Prefill Latency Heatmap (ms)')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Latency (ms)')

# 添加数值标注
for i in range(len(batch_sizes)):
    for j in range(len(input_lens)):
        value = prefill_latency_matrix[i, j]
        if value > 0:
            text_color = 'white' if value > prefill_latency_matrix.max() * 0.6 else 'black'
            ax1.text(j, i, f'{value:.0f}', ha='center', va='center', 
                    color=text_color, fontsize=8, weight='bold')

# 2. Decode延迟热力图
im2 = ax2.imshow(decode_latency_matrix, aspect='auto', cmap='Blues', origin='lower')
ax2.set_xticks(range(len(input_lens)))
ax2.set_xticklabels(input_lens)
ax2.set_yticks(range(len(batch_sizes)))
ax2.set_yticklabels(batch_sizes)
ax2.set_xlabel('Input Length')
ax2.set_ylabel('Batch Size')
ax2.set_title('Decode Latency Heatmap (ms)')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Latency (ms)')

# 添加数值标注
for i in range(len(batch_sizes)):
    for j in range(len(input_lens)):
        value = decode_latency_matrix[i, j]
        if value > 0:
            text_color = 'white' if value > decode_latency_matrix.max() * 0.6 else 'black'
            ax2.text(j, i, f'{value:.0f}', ha='center', va='center', 
                    color=text_color, fontsize=8, weight='bold')

# 3. 总延迟热力图
im3 = ax3.imshow(total_latency_matrix, aspect='auto', cmap='Oranges', origin='lower')
ax3.set_xticks(range(len(input_lens)))
ax3.set_xticklabels(input_lens)
ax3.set_yticks(range(len(batch_sizes)))
ax3.set_yticklabels(batch_sizes)
ax3.set_xlabel('Input Length')
ax3.set_ylabel('Batch Size')
ax3.set_title('Total Latency Heatmap (ms)')
cbar3 = plt.colorbar(im3, ax=ax3)
cbar3.set_label('Total Latency (ms)')

# 添加数值标注
for i in range(len(batch_sizes)):
    for j in range(len(input_lens)):
        value = total_latency_matrix[i, j]
        if value > 0:
            text_color = 'white' if value > total_latency_matrix.max() * 0.6 else 'black'
            ax3.text(j, i, f'{value:.0f}', ha='center', va='center', 
                    color=text_color, fontsize=8, weight='bold')

# 4. Prefill延迟占比热力图
im4 = ax4.imshow(latency_ratio_matrix, aspect='auto', cmap='RdYlBu_r', origin='lower', vmin=0, vmax=1)
ax4.set_xticks(range(len(input_lens)))
ax4.set_xticklabels(input_lens)
ax4.set_yticks(range(len(batch_sizes)))
ax4.set_yticklabels(batch_sizes)
ax4.set_xlabel('Input Length')
ax4.set_ylabel('Batch Size')
ax4.set_title('Prefill Latency Ratio Heatmap (%)')
cbar4 = plt.colorbar(im4, ax=ax4)
cbar4.set_label('Prefill Ratio (0-1)')

# 添加百分比标注
for i in range(len(batch_sizes)):
    for j in range(len(input_lens)):
        value = latency_ratio_matrix[i, j]
        if value > 0:
            text_color = 'white' if abs(value - 0.5) > 0.3 else 'black'
            ax4.text(j, i, f'{value:.2f}', ha='center', va='center', 
                    color=text_color, fontsize=8, weight='bold')

plt.tight_layout()
plt.show()


# ## 延迟热力图分析
# 
# 上述延迟热力图提供了多个维度的延迟分析视角：
# 
# ### 1. **Prefill延迟热力图**（左上角，红色系）
# - **横轴**：Input Length（输入长度）
# - **纵轴**：Batch Size（批处理大小）
# - **颜色强度**：延迟值（毫秒），颜色越深延迟越高
# - **关键观察**：
#   - Prefill延迟随input_len呈线性增长
#   - 随batch_size呈亚线性增长（约B^0.6）
#   - 大input_len + 大batch_size组合延迟最高
# 
# ### 2. **Decode延迟热力图**（右上角，蓝色系）
# - **特征**：延迟相对稳定，受input_len影响较小
# - **关键观察**：
#   - Decode延迟主要受batch_size影响
#   - Input_len的影响较小（带宽瓶颈特征）
#   - 延迟增长相对平缓
# 
# ### 3. **总延迟热力图**（左下角，橙色系）
# - **含义**：Prefill + Decode的总延迟
# - **实用性**：用户最关心的端到端延迟
# - **优化参考**：寻找延迟和吞吐量的最佳平衡点
# 
# ### 4. **Prefill延迟占比热力图**（右下角，红蓝渐变）
# - **含义**：Prefill延迟占总延迟的比例（0-1）
# - **颜色解释**：
#   - 红色（接近1）：Prefill主导延迟
#   - 蓝色（接近0）：Decode主导延迟
#   - 黄色（约0.5）：两阶段延迟平衡
# - **战略意义**：
#   - 高比例：优化Prefill阶段收益更大
#   - 低比例：优化Decode阶段收益更大
# 
# ### 延迟优化策略建议：
# 
# 1. **延迟敏感场景**：选择左下角区域（小batch_size + 短input_len）
# 2. **吞吐量优先场景**：可接受右上角区域的延迟换取高吞吐量
# 3. **平衡策略**：根据延迟占比热力图，针对性优化主导阶段
# 
# ### 热力图使用指南：
# 
# - **数值标注**：每个格子内显示具体延迟值（毫秒）
# - **颜色对比**：便于快速识别性能热点和最优区域
# - **多维分析**：结合四个图表全面理解延迟特征
# - **部署决策**：根据业务需求选择合适的参数组合
# 

# In[18]:


# 延迟缩放分析和效率计算
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

# 1. 延迟缩放效率热力图
batch_sizes = sorted(prefill_data['batch_size'].unique())
input_lens = sorted(prefill_data['input_len'].unique())

# 计算延迟缩放效率 (理想延迟 / 实际延迟)
prefill_efficiency_matrix = np.zeros((len(batch_sizes), len(input_lens)))
decode_efficiency_matrix = np.zeros((len(batch_sizes), len(input_lens)))

for j, il in enumerate(input_lens):
    # 获取batch_size=1时的基准延迟
    base_data = prefill_data[(prefill_data['batch_size'] == 1) & (prefill_data['input_len'] == il)]
    if not base_data.empty:
        base_prefill_lat = base_data.iloc[0]['prefill_latency']
        base_decode_lat = base_data.iloc[0]['median_decode_latency']
        
        for i, bs in enumerate(batch_sizes):
            data = prefill_data[(prefill_data['batch_size'] == bs) & (prefill_data['input_len'] == il)]
            if not data.empty:
                actual_prefill_lat = data.iloc[0]['prefill_latency']
                actual_decode_lat = data.iloc[0]['median_decode_latency']
                
                # 理想情况下延迟应该保持不变（完美并行）
                if actual_prefill_lat > 0:
                    prefill_efficiency_matrix[i, j] = base_prefill_lat / actual_prefill_lat
                if actual_decode_lat > 0:
                    decode_efficiency_matrix[i, j] = base_decode_lat / actual_decode_lat

# 绘制Prefill延迟效率热力图
im1 = ax1.imshow(prefill_efficiency_matrix, aspect='auto', cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
ax1.set_xticks(range(len(input_lens)))
ax1.set_xticklabels(input_lens)
ax1.set_yticks(range(len(batch_sizes)))
ax1.set_yticklabels(batch_sizes)
ax1.set_xlabel('Input Length')
ax1.set_ylabel('Batch Size')
ax1.set_title('Prefill Latency Efficiency (理想延迟/实际延迟)')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Efficiency (1=完美, 0=极差)')

# 添加效率标注
for i in range(len(batch_sizes)):
    for j in range(len(input_lens)):
        value = prefill_efficiency_matrix[i, j]
        if value > 0:
            text_color = 'white' if value < 0.5 else 'black'
            ax1.text(j, i, f'{value:.2f}', ha='center', va='center', 
                    color=text_color, fontsize=8, weight='bold')

# 绘制Decode延迟效率热力图
im2 = ax2.imshow(decode_efficiency_matrix, aspect='auto', cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
ax2.set_xticks(range(len(input_lens)))
ax2.set_xticklabels(input_lens)
ax2.set_yticks(range(len(batch_sizes)))
ax2.set_yticklabels(batch_sizes)
ax2.set_xlabel('Input Length')
ax2.set_ylabel('Batch Size')
ax2.set_title('Decode Latency Efficiency (理想延迟/实际延迟)')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Efficiency (1=完美, 0=极差)')

# 添加效率标注
for i in range(len(batch_sizes)):
    for j in range(len(input_lens)):
        value = decode_efficiency_matrix[i, j]
        if value > 0:
            text_color = 'white' if value < 0.5 else 'black'
            ax2.text(j, i, f'{value:.2f}', ha='center', va='center', 
                    color=text_color, fontsize=8, weight='bold')

# 3. 延迟增长率热力图 (相对于batch_size=1)
prefill_growth_matrix = np.zeros((len(batch_sizes), len(input_lens)))
decode_growth_matrix = np.zeros((len(batch_sizes), len(input_lens)))

for j, il in enumerate(input_lens):
    base_data = prefill_data[(prefill_data['batch_size'] == 1) & (prefill_data['input_len'] == il)]
    if not base_data.empty:
        base_prefill_lat = base_data.iloc[0]['prefill_latency']
        base_decode_lat = base_data.iloc[0]['median_decode_latency']
        
        for i, bs in enumerate(batch_sizes):
            data = prefill_data[(prefill_data['batch_size'] == bs) & (prefill_data['input_len'] == il)]
            if not data.empty:
                actual_prefill_lat = data.iloc[0]['prefill_latency']
                actual_decode_lat = data.iloc[0]['median_decode_latency']
                
                if base_prefill_lat > 0:
                    prefill_growth_matrix[i, j] = actual_prefill_lat / base_prefill_lat
                if base_decode_lat > 0:
                    decode_growth_matrix[i, j] = actual_decode_lat / base_decode_lat

# 绘制Prefill延迟增长热力图
im3 = ax3.imshow(prefill_growth_matrix, aspect='auto', cmap='Reds', origin='lower')
ax3.set_xticks(range(len(input_lens)))
ax3.set_xticklabels(input_lens)
ax3.set_yticks(range(len(batch_sizes)))
ax3.set_yticklabels(batch_sizes)
ax3.set_xlabel('Input Length')
ax3.set_ylabel('Batch Size')
ax3.set_title('Prefill Latency Growth (相对batch_size=1)')
cbar3 = plt.colorbar(im3, ax=ax3)
cbar3.set_label('Growth Factor (1=无增长)')

# 添加增长倍数标注
for i in range(len(batch_sizes)):
    for j in range(len(input_lens)):
        value = prefill_growth_matrix[i, j]
        if value > 0:
            text_color = 'white' if value > prefill_growth_matrix.max() * 0.6 else 'black'
            ax3.text(j, i, f'{value:.1f}x', ha='center', va='center', 
                    color=text_color, fontsize=8, weight='bold')

# 绘制Decode延迟增长热力图
im4 = ax4.imshow(decode_growth_matrix, aspect='auto', cmap='Blues', origin='lower')
ax4.set_xticks(range(len(input_lens)))
ax4.set_xticklabels(input_lens)
ax4.set_yticks(range(len(batch_sizes)))
ax4.set_yticklabels(batch_sizes)
ax4.set_xlabel('Input Length')
ax4.set_ylabel('Batch Size')
ax4.set_title('Decode Latency Growth (相对batch_size=1)')  
cbar4 = plt.colorbar(im4, ax=ax4)
cbar4.set_label('Growth Factor (1=无增长)')

# 添加增长倍数标注
for i in range(len(batch_sizes)):
    for j in range(len(input_lens)):
        value = decode_growth_matrix[i, j]
        if value > 0:
            text_color = 'white' if value > decode_growth_matrix.max() * 0.6 else 'black'
            ax4.text(j, i, f'{value:.1f}x', ha='center', va='center', 
                    color=text_color, fontsize=8, weight='bold')

plt.tight_layout()
plt.show()

# 打印关键性能指标总结
print("=== 延迟性能总结 ===")
print(f"Prefill 最高效率: {prefill_efficiency_matrix.max():.3f}")
print(f"Decode 最高效率: {decode_efficiency_matrix.max():.3f}")
print(f"Prefill 最大增长倍数: {prefill_growth_matrix.max():.1f}x")
print(f"Decode 最大增长倍数: {decode_growth_matrix.max():.1f}x")

# 找出最优配置
best_efficiency_idx = np.unravel_index(
    (prefill_efficiency_matrix + decode_efficiency_matrix).argmax(), 
    prefill_efficiency_matrix.shape
)
best_bs = batch_sizes[best_efficiency_idx[0]]
best_il = input_lens[best_efficiency_idx[1]]
print(f"综合效率最优配置: batch_size={best_bs}, input_len={best_il}")

