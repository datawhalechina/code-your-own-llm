# 第二章 环境配置

本章将详细介绍如何配置运行项目所需的开发环境。我们将从硬件要求、软件依赖到环境搭建进行全面而详尽的讲解。为了保证任何人都能学习并且跑通项目的代码，体验亲自训练大语言模型中的各种细节，因此我们进行了大量的实验来适配各种可能的硬件条件，覆盖了 `Linux`, `Windows` 以及 `Mac` 三种主流的操作系统，并且通过 `TIER` 的分级和训练参数的调整，适配无卡（`CPU`）、单卡（`Single GPU`）和单机多卡（`Multi-GPU`）等多种计算资源的情况，让学习者几乎能做到硬件无感知。

## 1 硬件环境

项目设计为上可在单个 `8×H100` 节点上运行完整的训练流程，下也支持在较小规模的硬件上进行实验。根据个人的预算需求和实际情况，可以选择不同的硬件配置，我们也为主流操作系统测试了多种 `TIER`, 方便学习者进行更加灵活的选择。

### 1.1 Linux

我们推荐使用 `Linux` 作为首选的操作系统，因为在 `Linux` 中已经建立了非常成熟且良好的开发者生态，且深度学习的训练往往在 `Linux` 中有着最高的计算效率，无论是简单尝试，科研实验还是生产部署都有着不可替代的地位。

`Linux` 有着许多的发行版，接下来我们将以 `Ubuntu 24.04.2 LTS` 为例，并搭配 `NVIDIA` 的 `GPU` 来作为 `Linux` 下的示例环境。

首先使用以下命令检查并确认自己系统当前的发行版信息

```bash
lsb_release -a
```

然后，确认自己服务器的 `CPU` 和 `GPU` 信息来选择最合适的 `LINUX-TIER`

```bash
# CPU
lscpu
# NVIDIA-GPU
nvidia-smi -L
# AMD-GPU
rocm-smi
```
如[表2-1](#tab2-1)所示，在 `Linux` 环境中，我们为 `CPU`, 单卡和多卡都设置了不同的 `TIER`，学习者可以根据自己的实际情况选择最适合自己的进行后续环境配置和学习。<span id="tab2-1"> </span>

<div align="center">
	<p>表2-1 Linux操作系统下的Tier划分</p>
</div>

<div align="center">

| `Linux-Tier`      | 计算资源                 | 显存      | 适用模型参数量 | 卡时预算（￥/小时） |
| ----------------- | ------------------------ | --------- | -------------- | ------------------- |
| `Linux-Tier-Free` | `CPU`                    | $0$       | `d4`           | ￥$0$               |
| `Linux-Tier-1`    | `1 × RTX 3090/4090/5090` | $24/32GB$ | `d10`          | ￥$2$               |
| `Linux-Tier-2`    | `8 × A6000`              | $384GB$   | `d16`          | ￥$20$              |
| `Linux-Tier-3`    | `8 × A100/H100`          | $640GB$   | `d20`          | ￥$56/128$          |
| `Linux-Tier-4`    | `8 × H200`               | $1152GB$  | `d32`          | ￥$240$             |

</div>

### 1.2 Windows

虽然可以选择 `Windows` 作为项目学习的环境，但是可能存在不必要的显存开销和性能损失，推荐只有 `	Windows` 系统的学习者安装并使用 `WSL2` 并在其中参考[第1.1节](#1.1-Linux)选择并配置环境。

在 `Windows` 中，考虑到绝大部分情况为学习者个人的工作环境而非专业的服务器或工作站环境，我们只为是否有显卡区分了两种 `TIER`，如[表2-2](#tab2-2)所示，学习者可以根据自己的实际情况选择最适合自己的进行后续环境配置和学习。<span id="tab2-2"> </span>

<div align="center">
	<p>表2-2 Windows操作系统下的Tier划分</p>
</div>

<div align="center">

| `Windows-Tier`      | 计算资源                 | 显存      | 适用模型参数量 | 卡时预算 |
| ------------------- | ------------------------ | --------- | -------------- | -------- |
| `Windows-Tier-Free` | `CPU`                    | $0$       | `d4`           | $0$      |
| `Windows-Tier-1`    | `1 × RTX 3090/4090/5090` | $24/32GB$ | `d10`          | $0$      |

</div>

### 1.3 macOS

`macOS` 用户可以使用M系列芯片的 `MPS` 加速进行小规模实验，适用的规模和计算资源参考 `Windows-Tier-Free`。



### 1.4 存储要求

训练过程中需要存储预训练数据、模型检查点和中间结果，因此建议学习者至少准备以下存储空间：

- <strong>预训练数据</strong>：`~20GB`（`FineWeb` 部分数据集）
- <strong>模型检查点</strong>：`~10GB`（包含多个训练阶段的 `checkpoint`）
- <strong>临时文件</strong>：`~5GB`（分词器训练、评估结果等）
- <strong>总计</strong>：建议预留 `35GB` 的可用存储空间，默认的数据和模型的存储路径为 `~/.cache/nanochat/`

如果SSD固态硬盘的大小不够，可以适当减少模型的参数量和数据的下载量；同时可以尝试挂载 `NFS`、`PFS`、`CephFS` 或 `S3` 作为网络文件存储，并注意修改默认存储路径到挂载路径下。

以 `CephFS` 为例，挂载的命令如下：

```bash
cd /mnt
sudo mkdir $USER
sudo mount -t ceph $IP:$PORT:$PATH -o name=$NAME,secret=$SECRET== /mnt/$USER
```



## 2 软件环境

项目的软件环境基于`Python`，主要使用 `PyTorch` 作为深度学习框架，并单独使用 `Rust` 实现了高性能的 `BPE` 分词器。整个项目使用 `uv`，一个同样使用 `Rust` 实现的 `Python` 包和项目管理器，进行统一维护。

### 2.1 虚拟环境

可以选择使用 `conda` 或 `venv` 来管理虚拟环境，新建虚拟环境的目的是让项目之间的环境相互隔离，不会因为依赖冲突而难以维护。

- 使用 `conda` 新建虚拟环境并安装 `Python` 和 `uv`

```bash
conda create -n code python=3.10
pip install uv
# install the repo dependencies
uv sync --extra gpu
```

- 使用 `venv` 新建虚拟环境

```bash
# install uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
```

验证环境

```bash
python --version
pip show uv
```

`torch` 可用性验证

```python
import torch

# 检查 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 检查 CUDA 是否可用
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

# 检查 GPU 数量
if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")

# 检查 MPS 是否可用（macOS）
if hasattr(torch.backends, 'mps'):
    print(f"MPS 是否可用: {torch.backends.mps.is_available()}")
```

### 2.2 Rust工具链

然后安装 `Rust` 的编译环境

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

验证环境

```bash
rustc --version
cargo --version
```

### 2.3 其他操作系统备注

#### 2.3.1 Windows

`Windows` 中不支持直接使用 `pip` 来安装 `uv`, 需要独立安装并将其添加到环境变量中

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
scoop install rust
```

验证环境

```powershell
uv --verison
rustc --version
```

#### 2.3.2 macOS

`macOS` 用户可以使用M系列芯片的 `MPS` 加速进行小规模实验。

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install rust
```

验证环境

```
rustc --version
```



## 3 云计算平台

如果本地没有足够的 `GPU` 资源，但是希望能够训练更高的 `TIER`，可以考虑使用云算力平台进行训练。国内的云算力平台综合性价比和稳定性推荐 [`AutoDL`](https://www.autodl.com/)。

### 3.1 AutoDL

`AutoDL` 提供按需付费的 `GPU` 实例，最多同时租赁单节点八卡：

1. 注册 [`AutoDL`](https://www.autodl.com/) 账号
2. 选择适合的实例和基础镜像并启动
3. 创建 SSH 密钥对
4. 通过 SSH 连接到实例
5. 按照[第2节](#2-软件环境)安装依赖配置环境

### 3.2 其他云计算平台

其他提供 `GPU` 的 `AI` 训练的云计算平台包括：

- <strong>[Google Cloud Platform](https://cloud.google.com/)</strong>：提供 A100 和 H100 实例
- <strong>[Amazon Web Services](https://aws.amazon.com/)</strong>：提供 P4/P5 实例
- <strong>[Microsoft Azure](https://azure.microsoft.com/)</strong>：提供 ND 系列实例
- <strong>[Runpod](https://www.runpod.io/)</strong>：提供性价比较高的 GPU 租赁服务
