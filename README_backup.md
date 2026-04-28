# BLGpose（Bridging Language and Geometry for 6D Pose）

本仓库包含一个面向 **6D Pose Estimation** 的训练/实验代码框架（以 `train.py` 为入口），整体基于 **PyTorch + PyTorch Lightning**，并提供了对应的配置与预训练模型目录。

> 当前目录名为 `blgpose/`，建议作为 GitHub 仓库根目录（或子模块）直接使用。

## 功能概览

- **训练入口**：`train.py`（支持选择数据集、配置文件、权重输出目录、断点恢复等）
- **配置管理**：`config/` 下提供默认配置与 MicKey 相关训练配置（如 `curriculum_learning_warm_up.yaml`）
- **预训练模型**：`pretrained_models/`（请按需放置/下载权重）

## 目录结构（简要）

- `train.py`：训练入口
- `config/`：训练与模型配置
- `lib/`：模型、数据与训练逻辑
- `pretrained_models/`：预训练权重（不一定随仓库提供）
- `resources/environment.yml`：Conda 环境定义（Python 3.8 / torch 2.0.1 等）
- `figures/`：图片/可视化资源

## 环境安装

本项目推荐使用 Conda 创建环境（仓库已提供 `resources/environment.yml`）。

```bash
conda env create -f resources/environment.yml
conda activate mickey
```

如果你不使用 Conda，也可以参考 `resources/environment.yml` 中的 pip 依赖手动安装（建议保持版本一致以避免不兼容问题）。

## 快速开始

### 1) 数据准备

`train.py` 支持的数据集选项：

- `Shapenet6D`
- `NOCS`（默认）
- `TOYL`

运行时需要指定数据根目录 `--dataset_root`。默认值为 `Oryon/data`，你可以按自己的机器路径修改，例如：

```bash
python train.py --dataset NOCS --dataset_root "E:/datasets/NOCS"
```

> 说明：数据的具体组织方式由 `lib/datasets/` 内的 `DataModule`/数据集实现决定；若你接入了自定义数据集，请在对应模块中适配。

### 2) 训练

使用默认配置（MicKey warm-up curriculum）进行训练：

```bash
python train.py ^
  --config config/MicKey/curriculum_learning_warm_up.yaml ^
  --dataset NOCS ^
  --dataset_root Oryon/data ^
  --experiment MicKey_default ^
  --path_weights weights
```

训练日志默认使用 TensorBoard。权重与日志会写入 `--path_weights` 指定目录下（由 PyTorch Lightning 的 logger/回调管理）。

### 3) 断点恢复

```bash
python train.py --resume "path/to/your.ckpt"
```

## 训练配置说明

- **默认配置基类**：`config/default.py`
- **示例配置**：`config/MicKey/*.yaml`

你可以通过修改 YAML（如 batch size、num workers、学习率、epoch 等）来调整训练策略。GPU 相关设置在配置里通常通过 `TRAINING.NUM_GPUS` 控制（示例配置中有 `[1]` 或 `2` 等写法）。

## 预训练权重

仓库包含 `pretrained_models/` 目录用于存放预训练模型文件。若你的实验依赖额外权重（例如视觉/分割/匹配器相关权重），请按代码中读取路径放置到对应位置。

## 常见问题（FAQ）

- **Windows 路径/转义问题**：建议在命令行中对路径使用引号（如 `"E:/datasets/NOCS"`），或统一使用正斜杠 `/`。
- **显存不足（OOM）**：优先减小 batch size、降低输入分辨率（相关参数在配置与 `train.py` 的数据参数中）、或减少 worker/开启混合精度（Lightning 的 `precision`）。

## 引用与致谢

如果你基于本仓库进行研究或复现，建议在论文/报告中注明来源与参考实现。若你希望我补上准确的 BibTeX，请把对应论文链接或仓库上游链接发我，我可以把引用信息补齐到这里。