# 好心情rPPG系统结果值计算方式介绍

本文档详细介绍 rPPG（远程光电容积脉搏波描记法）系统中各种生理指标的计算方式、原理和意义。

## 目录
- [1. 心率计算](#1-心率计算)
- [2. HRV指标计算](#2-hrv指标计算)
  - [2.1 时域HRV指标](#21-时域hrv指标)
  - [2.2 频域HRV指标](#22-频域hrv指标)
- [3. HRV健康指数计算](#3-hrv健康指数计算)
- [4. 压力水平计算](#4-压力水平计算)
- [5. 数据处理流程](#5-数据处理流程)

## 1. 心率计算

心率（Heart Rate, HR）是通过使用快速傅里叶变换（FFT）分析脉搏信号的频率成分，找出主导频率并转换为每分钟心跳数（bpm）。

## 2. HRV指标计算

心率变异性（Heart Rate Variability, HRV）是指心跳间隔时间的变化。系统计算了以下HRV指标：

### 2.1 时域HRV指标

#### RMSSD (Root Mean Square of Successive Differences)

**定义**：相邻NN间隔差值的均方根。

**计算方式**：
1. 首先使用 `scipy.signal.find_peaks` 检测脉搏信号中的峰值位置
2. 计算相邻峰值之间的间隔（RR间隔），转换为毫秒单位
3. 计算相邻RR间隔差值的平方
4. 求这些平方值的平均值
5. 对平均值取平方根

```python
# 代码实现
rmssd = np.sqrt(np.mean(np.square(np.diff(nn_intervals))))
```

**意义**：主要反映副交感神经系统（迷走神经）的活动，对短期心率变异性敏感。

#### SDNN (Standard Deviation of NN intervals)

**定义**：所有NN间隔的标准差。

**计算方式**：计算所有RR间隔（转换为毫秒后）的标准差。

```python
# 代码实现
sdnn = np.std(nn_intervals)
```

**意义**：反映整体心率变异性，同时受交感神经和副交感神经活动的影响。

#### pNN50 (Percentage of NN50)

**定义**：相邻NN间隔差值超过50毫秒的百分比。

**计算方式**：
1. 计算相邻RR间隔差值的绝对值
2. 统计差值大于50毫秒的数量
3. 计算这个数量占总差值数量的百分比

```python
# 代码实现
nn_diffs = np.abs(np.diff(nn_intervals))
pnn50 = (np.sum(nn_diffs > 50) / len(nn_diffs)) * 100 if len(nn_diffs) > 0 else 0.0
```

**意义**：反映副交感神经系统活动，对迷走神经张力敏感。

### 2.2 频域HRV指标

#### LF (Low Frequency Power)

**定义**：低频段（0.04-0.15 Hz）的功率谱密度。

**计算方式**：
1. 对RR间隔序列进行预处理和线性插值，生成均匀采样的时间序列
2. 使用 Welch 方法计算功率谱密度
3. 计算0.04-0.15 Hz频率范围内的功率（通过梯形积分法）

```python
# 关键代码实现
lf_band = (0.04, 0.15)  # 低频带 (0.04-0.15 Hz)
lf_idx = np.logical_and(fxx >= lf_band[0], fxx <= lf_band[1])
lf = np.trapz(pxx[lf_idx], fxx[lf_idx]) if np.any(lf_idx) else 0.0
```

**意义**：反映交感神经系统活动，也受副交感神经系统影响，与血压调节有关。

#### HF (High Frequency Power)

**定义**：高频段（0.15-0.4 Hz）的功率谱密度。

**计算方式**：
1. 与LF计算类似，但频率范围为0.15-0.4 Hz
2. 使用梯形积分法计算该频段内的功率

```python
# 关键代码实现
hf_band = (0.15, 0.4)   # 高频带 (0.15-0.4 Hz)
hf_idx = np.logical_and(fxx >= hf_band[0], fxx <= hf_band[1])
hf = np.trapz(pxx[hf_idx], fxx[hf_idx]) if np.any(hf_idx) else 0.0
```

**意义**：主要反映副交感神经系统（迷走神经）活动，与呼吸频率相关。

#### LF/HF Ratio

**定义**：低频功率与高频功率的比值。

**计算方式**：LF值除以HF值（如果HF值大于0）。

```python
# 代码实现
lf_hf_ratio = lf / hf if hf > 0 else 0.0
```

**意义**：反映交感神经与副交感神经活动的平衡状态，比值升高通常表示压力增加或交感神经主导。

## 3. HRV健康指数计算

HRV健康指数是基于多项HRV指标综合计算的评分（0-100）。

**计算方式**：
1. 对每项HRV指标进行评分（0-3分）
   - RMSSD评分：>40ms得3分，>20ms得2分，>0得1分
   - SDNN评分：>50ms得3分，>30ms得2分，>0得1分
   - pNN50评分：>10%得3分，>5%得2分，>0得1分
   - HF评分：>500ms²得3分，>200ms²得2分，>0得1分
   - LF/HF比值评分：0.5-2.0得3分，<0.5或<4.0得2分，>0得1分
2. 计算总评分（0-15分）
3. 将总评分转换为0-100的健康指数
4. 限制结果在1-99范围内，避免极端值

```python
# 关键代码实现
total_score = rmssd_score + sdnn_score + pnn50_score + hf_score + lf_hf_ratio_score
health_index = (total_score / 15) * 100\health_index = int(max(1.0, min(99.0, health_index)))
```

**健康指数范围**：
- Excellent (优秀): >80
- Good (良好): >60
- Fair (一般): >40
- Poor (较差): ≤40

## 4. 压力水平计算

压力水平基于HRV指标计算，范围为1-99。

**计算方式**：
1. 计算时域评分（基于RMSSD和SDNN）
   - time_domain_score = 10 - ((rmssd / 50) + (sdnn / 60))
2. 计算频域评分（基于HF和LF/HF比值）
   - hf_normalized = hf / 500 if hf > 0 else 0
   - lf_hf_normalized = min(lf_hf_ratio / 5, 1) if lf_hf_ratio > 0 else 0
   - freq_domain_score = 10 - (hf_normalized * 5) + (lf_hf_normalized * 5)
3. 计算平均评分并转换到0-100范围
   - stress_score = ((time_domain_score + freq_domain_score) / 2) * 10
4. 限制结果在1-99范围内

```python
# 关键代码实现
stress_score = (time_domain_score + freq_domain_score) / 2
stress_score = stress_score * 10
stress_score = int(max(1.0, min(99.0, stress_score)))
```

**压力水平范围**：
- High (高压力): >70
- Medium (中等压力): >40
- Low (低压力): ≤40

## 5. 数据处理流程

整个数据处理流程包括以下步骤：

1. **视频帧提取**：
   - 从视频中读取帧，根据`FRAME_SUBSAMPLE_RATE`（当前为5）进行下采样
   - 使用LinkNet34分割模型提取人脸区域

2. **信号预处理**：
   - 计算人脸区域的RGB平均值作为原始信号
   - 使用Pulse类处理RGB信号，提取脉搏波形
   - 应用移动平均滤波平滑脉搏信号

3. **指标计算**：
   - 计算心率
   - 计算HRV指标（时域和频域）
   - 计算HRV健康指数
   - 计算压力水平

4. **结果输出**：
   - 返回包含所有计算结果和单位信息的字典

```python
# 主要处理流程（简化）
cap = cv2.VideoCapture(video_path)
# 提取人脸区域RGB信号
# ...
pulse_calculator = Pulse(framerate=effective_fs, signal_size=signal_size, batch_size=30)
pulse_signal = pulse_calculator.get_pulse(rgb_signal)
pulse_signal = moving_avg(pulse_signal, 6)

hr = pulse_calculator.get_rfft_hr(pulse_signal)
hrv_metrics = calculate_hrv_metrics(pulse_signal, video_fs)
hrv_health = calculate_hrv_health_index(hrv_metrics)
stress = get_stress_level(hrv_metrics)
```

## 参考范围说明

| 指标 | 良好范围 | 说明 |
|------|---------|------|
| RMSSD | >20 ms | 健康成年人通常在20-100 ms之间 |
| SDNN | >50 ms | 健康成年人通常在50-100 ms之间 |
| pNN50 | >5% | 健康成年人通常在5-20%之间 |
| LF | 因年龄和测试条件而异 | 通常在0-1000 ms²之间 |
| HF | 因年龄和测试条件而异 | 通常在0-500 ms²之间 |
| LF/HF Ratio | 0.5-2.0 | 反映自主神经系统平衡 |
| 健康指数 | >60 | 60以上表示良好或优秀 |
| 压力水平 | <40 | 40以下表示低压力 |

**注意**：以上参考范围仅供参考，实际健康评估应结合个体情况和专业医疗建议。