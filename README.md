# AMCLab v0.2.2：面向自动调制分类（AMC）的极简 PyTorch 工具包

`AMCLab` 是一个定位为 **可 `import` 的 Python 第三方包**、采用 **Keras 风格 API** 的自动调制分类（Automatic Modulation Classification, AMC）通用框架。

## 1. 仓库结构

```text
AMCLab/
├── pyproject.toml                             # 标准 Python 包配置 (v0.2.2)
├── requirements.txt
├── README.md
├── src/amclab/
│   ├── datasets/                              # 纯 IQ 数据集加载器（含 split_file 持久化与 RML2018 真懒加载）
│   ├── transforms/                            # 可组合信号变换与数据增强（torchvision 风格）
│   ├── models/                                # 单模型单文件、零跨文件依赖（16 个经典 Baseline）
│   │   ├── cnn1.py                            # CNN1 (VT-CNN2)
│   │   ├── cnn2.py                            # CNN2
│   │   ├── ic_amcnet.py                       # IC-AMCNet
│   │   ├── mcnet.py                           # MCNet
│   │   ├── resnet.py                          # ResNet (small / base / large)
│   │   ├── densenet.py                        # DenseNet
│   │   ├── gru.py                             # GRU
│   │   ├── lstm.py                            # LSTM
│   │   ├── dae.py                             # DAE
│   │   ├── cldnn.py                           # CLDNN
│   │   ├── cldnn2.py                          # CLDNN2
│   │   ├── mcldnn.py                          # MCLDNN
│   │   ├── cgdnet.py                          # CGDNet
│   │   ├── pet_cgdnn.py                       # PET-CGDNN
│   │   ├── iqformer.py                        # IQFormer (small / base / large，内部自算 STFT)
│   │   └── fea_t.py                           # FEA-T (small / base)
│   ├── engine/
│   │   ├── trainer.py                         # Keras 风格 Trainer（fit / resume_from / extra_epochs）
│   │   ├── evaluator.py                       # Keras 风格 Evaluator（Overall / High-SNR / Per-SNR / F1 / Kappa）
│   │   └── metrics.py                         # 全量聚合指标计算
│   └── utils/
│       └── runtime.py                         # NVIDIA CUDA / Apple MPS / CPU 自动识别与随机种子管理
└── examples/
    ├── 01_single_baseline_train_resume/       # 范例 1：单个 Baseline 的训练、追加训练与评估
    │   └── train_and_resume.py
    ├── 02_custom_model/                       # 范例 2：自定义模型多文件组织与多变体训练
    │   ├── my_model.py
    │   └── run_custom_model.py
    └── 03_batch_train_and_eval/               # 范例 3：全量 Baseline 批量训练 + 批量测试汇总表
        ├── config.py                          # 统一配置数据集路径、共享 split_file 与模型列表
        ├── 01_batch_train.py                  # 代码 1：批量训练所有内置模型
        └── 02_batch_eval_summary.py           # 代码 2：批量加载权重测试并汇总到控制台与 CSV 表
```

## 2. 安装与使用

在项目根目录执行可编辑安装：

```bash
pip install -r requirements.txt
pip install -e .
```

安装完成后，运行任意范例（每个范例顶部都用醒目注释标出了 `DATASET_NAME` 与 `DATASET_PATH`，默认 `"synthetic"` 可零依赖直跑，改为真实路径即可跑 RML2016/2018/22/HisarMod）：

```bash
# 范例 1：单模型训练、追加训练 (extra_epochs) 与测试
python examples/01_single_baseline_train_resume/train_and_resume.py

# 范例 2：自定义模型多变体训练与测试
python examples/02_custom_model/run_custom_model.py

# 范例 3：批量训练全部 16 个内置 Baseline，再批量加载测试并汇总结果表
python examples/03_batch_train_and_eval/01_batch_train.py
python examples/03_batch_train_and_eval/02_batch_eval_summary.py
```

## 3. `split_file` 固化划分用法

调用 `load_dataset` 时传入 `split_file="./runs/splits_seed42.npz"`：
- **首次运行**：若该文件不存在，按 `seed` 执行 `(调制类别, SNR)` 严格分层划分并自动保存到 `split_file`。
- **后续运行**：若该文件已存在，直接读取并校验其中的 `train/val/test` 索引，确保不同模型、不同阶段（训练、追加训练、批量评估）严格使用同一批样本划分。
