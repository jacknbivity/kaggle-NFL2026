# NFL Big Data Bowl 2026 - Prediction

> Kaggle 竞赛：[NFL Big Data Bowl 2026 - Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)

## 🏆 竞赛成绩

**第 21 名 / 1899 支队伍**（Top ~1.1%）

![Ranking](./RANK.png)

## 👤 个人负责部分

本仓库仅包含本人（[jacknbivity](https://github.com/jacknbivity)）负责的模型部分，涵盖两种深度学习轨迹预测方案。其余队友负责的数据清洗、特征工程方案整合、后处理优化等部分不在本仓库展开。

本人核心贡献：
- 🧠 设计并实现 **ST-GRU** 与 **ST-Transformer** 两种时空序列预测模型
- 🎯 提出 **Temporal Huber Loss** 时序加权损失函数，对远期预测施加时间衰减惩罚
- 🔀 **方向统一化** + **相对坐标变换** 预处理管线，消除进攻方向异构性
- 📐 **Frame Spatial Attention** + **Relative Position Bias** 帧内球员交互建模
- 🔢 **Fourier Feature Encoder** / **RBF Encoder** 位置高频编码
- ⚡ 全流程 **内存优化**（float32 量化、Polars 高效数据处理、多进程并行加载）
- 📊 **GroupKFold 多折交叉验证** + 多随机种子集成

---

本项目提供两种深度学习模型来预测 NFL 比赛中球员的移动轨迹：

| 模型 | 文件 | 说明 |
|------|------|------|
| **ST-GRU** | `519-ST-GRU.py` | 基于 SpatioTemporal GRU 的序列预测模型，融合空间注意力机制与傅里叶特征编码 |
| **ST-Transformer** | `519-STTransformer.py` | 基于 SpatioTemporal Transformer 的序列预测模型，更强的时空建模能力 |

## 环境要求

- Python 3.9+
- PyTorch 2.0+（支持 CUDA）
- 详见 [requirements.txt](requirements.txt)

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

将 Kaggle 竞赛数据放置在 `./nfl-big-data-bowl-2026-prediction/` 目录下，结构如下：

```
nfl-big-data-bowl-2026-prediction/
├── train/
│   ├── input_2023_w01.csv
│   ├── output_2023_w01.csv
│   ├── ...
│   ├── input_2023_w09.csv
│   └── output_2023_w09.csv
└── test/
    └── ...
```

### 训练模型

```bash
# 使用 ST-GRU 模型训练
python 519-ST-GRU.py --mode train

# 使用 ST-Transformer 模型训练
python 519-STTransformer.py --mode train
```

### 推理预测

```bash
# 使用 ST-GRU 模型推理
python 519-ST-GRU.py --mode infer

# 使用 ST-Transformer 模型推理
python 519-STTransformer.py --mode infer
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式：`train` 或 `infer` | `infer` |
| `--model_path` | 模型保存/加载路径 | `./saved_models_rnn` |
| `--use_cache` | 使用缓存的序列数据 | `True` |
| `--no_cache` | 强制重新构建序列 | — |

## 方法概述

### 特征工程

- 方向统一化：将所有右向进攻镜像为左向，便于模型学习
- 坐标变换：将绝对坐标转换为相对于球落点的相对坐标
- 高级特征：速度分量、加速度分量、动量、动能、空间位置特征等
- 时空序列构建：将每场比赛构建为 `[B, T, N, F]` 格式的时空序列

### 模型架构

#### ST-GRU（时空门控循环单元）
核心设计思路：**帧内空间注意力 → 帧间 GRU 时序建模 → 注意力池化 → 分段预测**

| 模块 | 说明 |
|------|------|
| `FourierFeatureEncoder` | 多频率傅里叶编码（bands=1,2,4,8,16），将低维坐标映射到高维正弦/余弦空间 |
| `RBFEncoder` | 径向基函数编码，对距离特征进行非线性展开 |
| `RelativePositionBias` | 相对位置偏置，增强空间注意力对球员间几何关系的感知 |
| `FrameSpatialAttention` | 帧内多头空间注意力，建模同一时刻所有球员的交互关系 |
| `SpatioTemporal_GRU` | 堆叠 GRU 进行帧间时序编码，配合 MultiheadAttention 池化 |
| `ResidualMLP` | 残差 MLP 预测头，分段输出 short/mid/long 三个时间范围 |
| `TemporalHuber` | 时序加权 Huber 损失：时间衰减 + L2 平滑正则 |

#### ST-Transformer（时空 Transformer）
核心设计思路：**输入嵌入 + 时间位置编码 → 空间 Transformer（帧内） → 时间 Transformer（帧间） → 线性预测头**

| 模块 | 说明 |
|------|------|
| `embedding` | 特征嵌入层，将输入映射到 hidden_dim |
| `temporal_pos_embedding` | 可学习时间位置编码 |
| `spatial_transformer` | nn.TransformerEncoder，GELU 激活，建模帧内球员空间交互 |
| `temporal_transformer` | nn.TransformerEncoder，建模时序依赖 |
| `head` | 线性层一次性输出 horizon×2 维位移预测 |

### 损失函数：Temporal Huber Loss

$$
L = \underbrace{\frac{\sum_{t} w_t \cdot \text{Huber}(y_t, \hat{y}_t)}{\sum w_t}}_{\text{时间衰减 Huber}} + \lambda \cdot \underbrace{\frac{\sum_{t} \|\Delta^2 \hat{y}_t\|^2}{\sum m_t}}_{\text{二阶平滑正则}}
$$

其中 $w_t = e^{-\alpha t}$ 为时间衰减权重，越远的帧惩罚越小（鼓励模型优先学好近期预测）；$\lambda$ 控制轨迹平滑度。

### 训练策略

- **交叉验证**：GroupKFold（按 game_id 分组），ST-GRU 5 折 / ST-Transformer 10 折
- **早停**：patience=30 epochs
- **优化器**：Adam，学习率 1e-3
- **批次**：Batch Size 128，窗口大小 8，隐藏维度 128
- **正则**：Dropout 0.1，LayerNorm，多随机种子集成
- **数据质量**：截断超过 50 帧（5 秒）的异常长 play

## 项目结构

```
.
├── 519-ST-GRU.py          # ST-GRU 模型（主模型）
├── 519-STTransformer.py   # ST-Transformer 模型
├── RANK.png              # 竞赛排名截图
├── requirements.txt       # Python 依赖
├── .gitignore            # Git 忽略规则
├── README.md             # 本文件
├── log/                  # 运行日志（自动生成）
│   └── YYYYMMDD_HHMMSS/  # 按时间戳组织的日志
├── saved_models_rnn/     # 模型保存目录
└── nfl-big-data-bowl-2026-prediction/  # 竞赛数据（需自行下载）
```

## 致谢

- [NFL Big Data Bowl 2026](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction) 竞赛主办方
- Kaggle 社区

## License

MIT License
