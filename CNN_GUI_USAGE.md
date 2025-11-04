# EventPS CNN-PS GUI 使用说明

## 可执行文件说明

### 编译配置
- **Features**: `loader_render` （仅启用渲染功能）
- **不包含**: `display_cv`, `display_gl` （避免与Python端OpenCV GUI冲突）
- **Python版本**: Python 3.10
- **位置**: `target/release/event_ps_eval` 和 `target/release/event_ps_train`

### GUI显示方式

#### Python端GUI（CNN模型预测结果）✅ 当前配置
```bash
# 配置文件设置
show_ls_ps = none  # 关闭Rust端GUI

# Python代码启用GUI
cv.imshow("normal_gt", ...)      # 真实法线
cv.imshow("buffer_show", ...)    # 事件缓冲区
cv.imshow("normal_pred", ...)    # CNN预测结果
cv.waitKey(1)
```

**优点**:
- 可以看到CNN模型的预测结果
- 显示更多细节（ground truth、buffer、prediction）
- 适合模型调试和评估

**运行方式**:
```bash
cd /home/c303/cxz/EventPS（GUI）/EventPS
source venv/bin/activate
./target/release/event_ps_eval data/diligent/000000/eval_cnn_ps.ini
```

#### Rust端GUI（LS-PS经典方法）
```bash
# 配置文件设置
show_ls_ps = cv  # 或 gl

# Python代码禁用GUI
# 注释掉所有 cv.imshow() 和 cv.waitKey()
```

**优点**:
- 实时显示LS-PS法线重建
- 性能更好

**缺点**:
- 看不到CNN模型结果
- 与Python GUI冲突（会段错误）

**切换方式**:
1. 重新编译启用 `display_cv`: `cargo build --release --features display_cv,loader_render`
2. 修改配置文件: `show_ls_ps = cv`
3. 禁用Python端GUI代码

---

## 评估结果

### 模型: ev_cnn_ps_019500.bin

#### DiLiGenT数据集评估结果

| Dataset | LS-PS Error | CNN-PS Error | Improvement |
|---------|-------------|--------------|-------------|
| Ball    | 10.99°      | 10.16°       | 7.52%       |
| Bear    | 18.73°      | 16.45°       | 12.14%      |
| Buddha  | 12.74°      | 11.67°       | 8.39%       |
| Cat     | 26.51°      | 17.32°       | **34.68%**  |
| Cow     | 18.43°      | 15.31°       | 16.91%      |
| Goblet  | 36.06°      | 23.76°       | **34.12%**  |
| Harvest | 13.78°      | 12.45°       | 9.61%       |
| Pot1    | 15.75°      | 14.49°       | 7.98%       |
| Pot2    | 24.61°      | 19.08°       | 22.47%      |
| **Average** | **19.73°** | **15.63°** | **20.78%** |

### 详细结果文件
- `evaluation_results_diligent.csv` - 数据表格
- `evaluation_results_diligent.txt` - 文本报告
- `evaluation_results_diligent.png` - 综合分析图（4合1）
- `evaluation_comparison_simple.png` - 简洁对比图

---

## 训练说明

### 训练命令
```bash
cd /home/c303/cxz/EventPS（GUI）/EventPS
source venv/bin/activate
./target/release/event_ps_train --cnn-ps-train python/cnn_ps_train.py
```

### 训练配置
- Learning Rate: 4e-4
- 优化器: Adam
- 学习率调度: MultiStepLR
- GUI显示: 自动启用

---

## 重要说明

### ⚠️ 不能同时启用的功能
**Rust端display_cv/display_gl** 和 **Python端cv.imshow()** 不能同时使用，会导致段错误！

### ✅ 当前配置（推荐）
- Rust: 只编译 `loader_render`
- Python: 启用 `cv.imshow()` 显示CNN结果
- 配置: `show_ls_ps = none`

### 如需重新编译
```bash
# 仅Python端GUI（当前配置）
cargo build --release --features loader_render

# 仅Rust端GUI（需同时禁用Python GUI）
cargo build --release --features display_cv,display_gl,loader_render
```

---

**日期**: 2025-10-21  
**模型版本**: ev_cnn_ps_019500.bin (19500次迭代)  
**性能**: 在DiLiGenT数据集上平均提升20.78%










