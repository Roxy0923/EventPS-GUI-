#!/usr/bin/env python3
"""
EventPS 评估报告生成器
自动从评估日志生成完整的评估报告，包括CSV、文本和可视化图表

使用方法:
    python3 generate_evaluation_report.py <log_file> [output_dir]

示例:
    # 自动创建输出目录
    python3 generate_evaluation_report.py logs/eval_ps_fcn_20251104_091804.log
    
    # 指定输出目录
    python3 generate_evaluation_report.py logs/eval_ps_fcn_20251104_091804.log eval_report/my_report
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import sys
from datetime import datetime

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 解析命令行参数 ==========
if len(sys.argv) < 2:
    print("错误: 缺少日志文件参数")
    print()
    print("使用方法:")
    print("  python3 generate_evaluation_report.py <log_file> [output_dir]")
    print()
    print("示例:")
    print("  python3 generate_evaluation_report.py logs/eval_ps_fcn_20251104_091804.log")
    print("  python3 generate_evaluation_report.py logs/eval_cnn_ps_all.log eval_report/my_report")
    sys.exit(1)

LOG_FILE = sys.argv[1]

# 检查日志文件是否存在
if not os.path.exists(LOG_FILE):
    print(f"错误: 日志文件不存在: {LOG_FILE}")
    sys.exit(1)

# 确定输出目录
if len(sys.argv) >= 3:
    OUTPUT_DIR = sys.argv[2]
else:
    # 自动创建带时间戳的输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 从日志文件名推断模型类型
    if 'cnn' in LOG_FILE.lower():
        OUTPUT_DIR = f'eval_report/cnn_report_{timestamp}'
    elif 'fcn' in LOG_FILE.lower() or 'ps_fcn' in LOG_FILE.lower():
        OUTPUT_DIR = f'eval_report/fcn_report_{timestamp}'
    else:
        OUTPUT_DIR = f'eval_report/report_{timestamp}'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("EventPS 评估报告生成器")
print("="*80)
print(f"日志文件: {LOG_FILE}")
print(f"输出目录: {OUTPUT_DIR}")
print("="*80)
print()

# ========== 读取评估日志 ==========
print("正在读取评估日志...")
with open(LOG_FILE, 'r') as f:
    content = f.read()

# ========== 自动检测模型类型 ==========
model_type = None
model_name = None
model_col_name = None

# 尝试提取模型名称
model_match = re.search(r'Loaded model: (ev_[\w_]+\.bin)', content)
if model_match:
    model_name = model_match.group(1)
    if 'cnn_ps' in model_name:
        model_type = 'CNN-PS'
        model_col_name = 'CNN-PS Error (°)'
    elif 'ps_fcn' in model_name:
        model_type = 'PS-FCN'
        model_col_name = 'PS-FCN Error (°)'

# 如果没有找到模型名称，从BENCHMARK关键字推断
if model_type is None:
    if 'BENCHMARK Event-CNN-PS' in content:
        model_type = 'CNN-PS'
        model_col_name = 'CNN-PS Error (°)'
        model_name = 'CNN-PS Model'
    elif 'BENCHMARK Event-PS-FCN' in content:
        model_type = 'PS-FCN'
        model_col_name = 'PS-FCN Error (°)'
        model_name = 'PS-FCN Model'
    else:
        print("警告: 无法识别模型类型，使用默认值")
        model_type = 'Unknown'
        model_col_name = 'Model Error (°)'
        model_name = 'Unknown Model'

print(f"✓ 检测到模型类型: {model_type}")
print(f"✓ 模型名称: {model_name}")
print()

# ========== 提取数据集结果 ==========
print("正在提取评估结果...")

datasets = []
dataset_names = {
    '000000': 'Ball',
    '000002': 'Bear',
    '000003': 'Buddha',
    '000004': 'Cat',
    '000005': 'Cow',
    '000006': 'Goblet',
    '000007': 'Harvest',
    '000008': 'Pot1',
    '000009': 'Pot2'
}

# 根据模型类型选择正确的正则表达式
if model_type == 'CNN-PS':
    model_pattern = r'BENCHMARK Event-CNN-PS n_pixels (\d+) ang_err_mean ([\d.]+)'
elif model_type == 'PS-FCN':
    model_pattern = r'BENCHMARK Event-PS-FCN n_pixels (\d+) ang_err_mean ([\d.]+)'
else:
    model_pattern = r'BENCHMARK Event-\w+-\w+ n_pixels (\d+) ang_err_mean ([\d.]+)'

for i in range(10):
    dataset_id = f'{i:06d}'
    if dataset_id == '000001':
        continue
    
    # 查找该数据集的所有结果
    pattern = f'Evaluating data/diligent/{dataset_id}/.*?✓ data/diligent/{dataset_id}/ completed'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        section = match.group(0)
        # 提取最后一个结果
        model_results = re.findall(model_pattern, section)
        ls_results = re.findall(r'BENCHMARK Event-LS-PS n_pixels (\d+) ang_err_mean ([\d.]+)', section)
        
        if model_results and ls_results:
            model_err = float(model_results[-1][1])
            ls_err = float(ls_results[-1][1])
            datasets.append({
                'Dataset ID': dataset_id,
                'Dataset Name': dataset_names.get(dataset_id, dataset_id),
                'Pixels': int(model_results[-1][0]),
                'LS-PS Error (°)': ls_err,
                model_col_name: model_err,
                'Improvement (%)': ((ls_err - model_err) / ls_err * 100)
            })

# ========== 创建DataFrame ==========
df = pd.DataFrame(datasets)

# 检查数据
if len(datasets) == 0:
    print("❌ 错误: 没有提取到任何数据！")
    print("   请检查日志文件格式是否正确。")
    sys.exit(1)

print(f"✓ 成功提取 {len(datasets)} 个数据集的评估结果")
print()

# 添加平均值行
avg_row = {
    'Dataset ID': 'Average',
    'Dataset Name': 'Average',
    'Pixels': '-',
    'LS-PS Error (°)': df['LS-PS Error (°)'].mean(),
    model_col_name: df[model_col_name].mean(),
    'Improvement (%)': df['Improvement (%)'].mean()
}
df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

# ========== 保存CSV ==========
print("正在生成报告文件...")
csv_path = os.path.join(OUTPUT_DIR, 'evaluation_results_diligent.csv')
df.to_csv(csv_path, index=False, float_format='%.2f')
print(f"  ✓ CSV: {csv_path}")

# ========== 保存文本报告 ==========
txt_path = os.path.join(OUTPUT_DIR, 'evaluation_results_diligent.txt')
with open(txt_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DiLiGenT Dataset Evaluation Results\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Log File: {LOG_FILE}\n")
    f.write("=" * 80 + "\n\n")
    f.write(df.to_string(index=False))
    f.write("\n\n")
    f.write("=" * 80 + "\n")
    f.write(f"Summary:\n")
    f.write(f"  Average LS-PS Error:  {df.iloc[:-1]['LS-PS Error (°)'].mean():.2f}°\n")
    f.write(f"  Average {model_type} Error: {df.iloc[:-1][model_col_name].mean():.2f}°\n")
    f.write(f"  Average Improvement:  {df.iloc[:-1]['Improvement (%)'].mean():.2f}%\n")
    f.write("=" * 80 + "\n")

print(f"  ✓ 文本报告: {txt_path}")

# ========== 生成综合分析图 ==========
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 各数据集误差对比
ax1 = axes[0, 0]
x = np.arange(len(df) - 1)
width = 0.35
bars1 = ax1.bar(x - width/2, df.iloc[:-1]['LS-PS Error (°)'], width, label='LS-PS', color='#FF6B6B', alpha=0.8)
bars2 = ax1.bar(x + width/2, df.iloc[:-1][model_col_name], width, label=model_type, color='#4ECDC4', alpha=0.8)

ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax1.set_ylabel('Angular Error (°)', fontsize=12, fontweight='bold')
ax1.set_title('Angular Error Comparison by Dataset', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df.iloc[:-1]['Dataset Name'], rotation=45, ha='right')
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}°', ha='center', va='bottom', fontsize=9)

# 2. 改进百分比
ax2 = axes[0, 1]
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df) - 1))
bars = ax2.barh(df.iloc[:-1]['Dataset Name'], df.iloc[:-1]['Improvement (%)'], color=colors, alpha=0.8)
ax2.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
ax2.set_title('Performance Improvement Over LS-PS', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')

# 3. 平均误差对比
ax3 = axes[1, 0]
avg_data = [df.iloc[:-1]['LS-PS Error (°)'].mean(), df.iloc[:-1][model_col_name].mean()]
colors_avg = ['#FF6B6B', '#4ECDC4']
bars = ax3.bar(['LS-PS', model_type], avg_data, color=colors_avg, alpha=0.8, width=0.6)
ax3.set_ylabel('Average Angular Error (°)', fontsize=12, fontweight='bold')
ax3.set_title('Overall Average Performance', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}°', ha='center', va='bottom', fontsize=16, fontweight='bold')

improvement = df.iloc[:-1]['Improvement (%)'].mean()
ax3.text(0.5, max(avg_data) * 0.5, f'Improvement:\n{improvement:.2f}%',
        ha='center', va='center', fontsize=18, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# 4. 散点图
ax4 = axes[1, 1]
ax4.scatter(df.iloc[:-1]['LS-PS Error (°)'], df.iloc[:-1][model_col_name],
           s=200, c=df.iloc[:-1]['Improvement (%)'], cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=1.5)

for i, row in df.iloc[:-1].iterrows():
    ax4.annotate(row['Dataset Name'], 
                (row['LS-PS Error (°)'], row[model_col_name]),
                fontsize=9, ha='right', va='bottom')

min_val = min(df.iloc[:-1]['LS-PS Error (°)'].min(), df.iloc[:-1][model_col_name].min()) - 2
max_val = max(df.iloc[:-1]['LS-PS Error (°)'].max(), df.iloc[:-1][model_col_name].max()) + 2
ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Equal Performance')

ax4.set_xlabel('LS-PS Error (°)', fontsize=12, fontweight='bold')
ax4.set_ylabel(f'{model_type} Error (°)', fontsize=12, fontweight='bold')
ax4.set_title(f'Error Scatter Plot (Below diagonal = {model_type} better)', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)
ax4.set_aspect('equal', adjustable='box')

cbar = plt.colorbar(ax4.collections[0], ax=ax4)
cbar.set_label('Improvement (%)', fontsize=10)

plt.tight_layout()
png_path = os.path.join(OUTPUT_DIR, 'evaluation_results_diligent.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ 综合分析图: {png_path}")

# ========== 生成简化对比图 ==========
fig2, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(df) - 1)
width = 0.35

bars1 = ax.bar(x - width/2, df.iloc[:-1]['LS-PS Error (°)'], width, 
              label='LS-PS (Classical)', color='#E74C3C', alpha=0.9)
bars2 = ax.bar(x + width/2, df.iloc[:-1][model_col_name], width, 
              label=f'{model_type} (Our Model)', color='#3498DB', alpha=0.9)

ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
ax.set_ylabel('Angular Error (°)', fontsize=14, fontweight='bold')
ax.set_title(f'DiLiGenT Dataset: {model_type} vs LS-PS Performance Comparison\nModel: {model_name}', 
            fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df.iloc[:-1]['Dataset Name'], rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

improvement_text = f'Average Improvement: {df.iloc[:-1]["Improvement (%)"].mean():.2f}%\n'
improvement_text += f'LS-PS Avg: {df.iloc[:-1]["LS-PS Error (°)"].mean():.2f}° → '
improvement_text += f'{model_type} Avg: {df.iloc[:-1][model_col_name].mean():.2f}°'

ax.text(0.02, 0.98, improvement_text, transform=ax.transAxes,
       fontsize=12, fontweight='bold', verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
simple_png_path = os.path.join(OUTPUT_DIR, 'evaluation_comparison_simple.png')
plt.savefig(simple_png_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ 简洁对比图: {simple_png_path}")

# ========== 复制日志文件到输出目录 ==========
import shutil
log_copy_path = os.path.join(OUTPUT_DIR, os.path.basename(LOG_FILE))
shutil.copy2(LOG_FILE, log_copy_path)
print(f"  ✓ 日志副本: {log_copy_path}")

print()
print("="*80)
print("✓ 报告生成完成！")
print("="*80)
print(f"输出目录: {OUTPUT_DIR}")
print()
print("生成的文件:")
print(f"  - evaluation_results_diligent.csv      # CSV数据表")
print(f"  - evaluation_results_diligent.txt      # 文本报告")
print(f"  - evaluation_results_diligent.png      # 综合分析图")
print(f"  - evaluation_comparison_simple.png     # 简洁对比图")
print(f"  - {os.path.basename(LOG_FILE)}         # 评估日志副本")
print()
print(f"平均误差: LS-PS {df.iloc[:-1]['LS-PS Error (°)'].mean():.2f}° → {model_type} {df.iloc[:-1][model_col_name].mean():.2f}°")
print(f"平均提升: {df.iloc[:-1]['Improvement (%)'].mean():.2f}%")
print("="*80)
