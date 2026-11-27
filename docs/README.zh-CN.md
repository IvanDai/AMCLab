# AMCLab

**面向自动调制分类（AMC）的轻量级 PyTorch 工具包。**

[English](../README.md) | 简体中文

AMCLab 是一个面向信号调制分类研究的开源 Python 工具包，将 I/Q 数据集加载、信号预处理与数据增强、内置基线模型以及 Keras 风格的训练和评估 API 整合到统一的实验流程中。

AMCLab 旨在减少重复编写训练代码的工作量，帮助研究者便捷地开发新模型、复现实验，并在一致的实验设置下比较不同模型架构。

## 主要特性

- **数据集：** 支持 RadioML 2016.10a/10b、RadioML 2018、RML22、HisarMod 2019 和合成 I/Q 数据；支持 RadioML 2018 HDF5 数据的懒加载。
- **信号变换：** 提供可组合的信号预处理和数据增强，包括功率归一化、随机相位旋转等。
- **基线模型：** 内置 16 类基线模型，覆盖卷积、循环、残差及混合架构，部分模型提供不同规模的变体。
- **模型训练：** Keras 风格的 `Trainer.fit()` 接口，支持检查点保存、早停和断点续训。
- **模型评估：** 支持总体准确率、高信噪比准确率、各 SNR 下的准确率、Macro-F1、Cohen's Kappa 和混淆矩阵。
- **实验复现：** 支持按调制类别和 SNR 分层划分数据集并保存划分索引、管理随机种子，以及自动选择 CUDA、MPS 或 CPU 设备。

## 安装与使用

项目配置要求 Python 3.9 及以上版本。请根据 Python 版本和硬件环境选择兼容的 PyTorch 版本。

### 方式一：作为 Python 包安装（推荐）

克隆仓库，并安装包含可选数据集依赖的 AMCLab：

```bash
git clone https://github.com/IvanDai/AMCLab.git
cd AMCLab
python -m pip install .
```

如果需要修改 AMCLab 源码，可改用可编辑安装：

```bash
python -m pip install -e .
```

也可以直接从 GitHub 安装（需要已安装 Git）：

```bash
python -m pip install "amclab @ git+https://github.com/IvanDai/AMCLab.git"
```

### 方式二：直接集成到自己的项目

将本仓库中的整个 `src/amclab/` 文件夹复制到自己的 Python 项目根目录：

```text
your_project/
├── amclab/
└── train.py
```

在自己的 Python 环境中安装所需的第三方依赖：

```bash
python -m pip install "torch>=2.0.0" "numpy>=1.24.0" "tqdm>=4.65.0" "h5py>=3.8.0" "scipy>=1.10.0"
```

从项目根目录运行脚本，即可直接导入本地 `amclab` 包，无需安装 AMCLab 本身，也无需修改 `sys.path`。分发源码时请保留 MIT 许可证声明。

两种方式使用相同的 Python API：

```python
import amclab

print(amclab.__version__)
print(amclab.list_models(canonical_only=True))
```

## 快速开始

使用合成数据训练并评估一个基线模型，无需下载数据集：

```python
from amclab import Evaluator, Trainer, get_model, load_dataset, seed_everything

seed_everything(42)

train_ds, val_ds, test_ds = load_dataset(
    name="synthetic",
    split=(0.6, 0.2, 0.2),
    seed=42,
    split_file="./runs/quickstart/splits_seed42.npz",
)

model = get_model(
    "mcldnn",
    num_classes=train_ds.num_classes,
    seq_length=train_ds.seq_length,
)

trainer = Trainer(model, device="auto", save_dir="./runs/quickstart")
trainer.fit(train_ds, val_ds, epochs=3, batch_size=32)

evaluator = Evaluator(model, device="auto")
metrics = evaluator.evaluate(
    test_ds,
    checkpoint="./runs/quickstart/checkpoints/best.pt",
    save_dir="./runs/quickstart/eval",
)
```

使用真实数据集时，修改对应的数据集名称和路径即可。训练、续训和评估阶段应复用同一个 `split_file`，确保训练集、验证集和测试集的划分保持一致。

## 示例

安装 AMCLab 后，在仓库根目录运行以下示例：

| 示例 | 说明 |
| --- | --- |
| [01 — 单模型实验](../examples/01_single_baseline_train_resume/) | 内置基线模型的训练、断点续训与评估。 |
| [02 — 自定义模型](../examples/02_custom_model/) | 自定义 PyTorch 模型及其不同变体的训练和评估。 |
| [03 — 批量基准测试](../examples/03_batch_train_and_eval/) | 批量训练基线模型、加载检查点进行评估并导出对比结果表。 |

```bash
python examples/01_single_baseline_train_resume/train_and_resume.py
python examples/02_custom_model/run_custom_model.py
python examples/03_batch_train_and_eval/01_batch_train.py
python examples/03_batch_train_and_eval/02_batch_eval_summary.py
```

所有示例默认使用合成数据。若要使用真实数据集，请修改示例脚本或共享 `config.py` 中的 `DATASET_NAME` 和 `DATASET_PATH`。实验结果默认保存在 `runs/` 目录下，仅在本地保留。

## 相关论文（Publications）

<!-- 在此补充相关论文。 -->

## 许可证与支持

AMCLab 采用 [MIT 许可证](../LICENSE) 开源。

如果 AMCLab 对你的研究或项目有所帮助，欢迎为本仓库点亮 ⭐ [Star](https://github.com/IvanDai/AMCLab)。感谢你的支持！

## 仓库结构

```text
AMCLab/
├── src/amclab/       # 核心 Python 包
│   ├── datasets/     # I/Q 数据集与数据划分
│   ├── transforms/   # 信号预处理与数据增强
│   ├── models/       # 基线模型
│   ├── engine/       # 训练、评估与指标
│   └── utils/        # 运行环境工具
├── examples/         # 训练与基准测试示例
├── docs/             # 补充文档
├── pyproject.toml    # Python 包配置
└── requirements.txt # 依赖列表
```
