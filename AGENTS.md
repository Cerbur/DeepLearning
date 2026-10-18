# Agent 指南与项目规范

本文件记录本项目（深度学习环境与后续开发）的全局规范与指引，所有 Agent 在本工作区执行任务时必须严格遵守。

---

## 1. 包管理规范（重要）

- **优先使用 Conda**：本项目为深度学习项目，为避免 CUDA、C++ 运行时库及底层二进制依赖冲突，**后续所有依赖包的安装必须优先使用 `conda install`，严禁默认使用 `pip install`**。
- **避免混用产生包冲突**：特别是涉及到 PyTorch、NumPy、SciPy、OpenCV、CUDA 相关工具包等核心底层库，必须通过 `conda` 进行版本管理与安装。
- **Pip 使用限制**：仅在目标库确实未收录于 Conda 官方源、清华镜像源或 conda-forge 时，方可谨慎使用 `pip install`，且安装时不得覆盖或升级已有的底层二进制包（如 torch、numpy、cuda 相关库）。

---

## 2. 深度学习环境信息

- **Conda 安装路径**：`C:\Users\10511\miniconda3`
- **专用虚拟环境**：`pytorch`
- **环境激活命令**：
  ```bash
  conda activate pytorch
  ```
- **环境核心版本**：
  - **Python**：`3.10` (3.10.22)
  - **PyTorch**：`2.0.1`
  - **CUDA Runtime**：`11.8` (`pytorch-cuda=11.8`)
  - **Torchvision**：`0.15.2`
  - **Torchaudio**：`2.0.2`
  - **NumPy**：`1.26.4` (兼容 PyTorch 2.0，保持 <2.0 版本)
- **已安装的基础工具库**：
  - **Pandas**：`2.3.3`
  - **Matplotlib**：`3.10.9`
  - **Seaborn**：`0.13.2`
  - **Scikit-learn**：`1.7.2`
  - **JupyterLab**：`4.6.4`
  - **ipykernel**：`7.3.0`（已注册 Jupyter 内核：`Python (PyTorch 2.0)`）
- **硬件检测**：
  - GPU：NVIDIA GeForce RTX 4070 SUPER (12GB VRAM)
  - CUDA 加速状态：`torch.cuda.is_available() == True`

---

## 3. 镜像源配置

Conda 配置文件位于 `~/.condarc`（`C:\Users\10511\.condarc`），已配置为清华大学开源软件镜像站（TUNA）：
- `defaults`（main, r, msys2）：清华源
- `pytorch` channel：清华镜像源 (`https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/`)
- `conda-forge` channel：清华镜像源 (`https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/`)

安装新包时命令示例：
```bash
conda install -n pytorch <package_name>
# 或在已激活环境内
conda install <package_name>
```
