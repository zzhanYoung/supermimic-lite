# SuperMimic Lite

[English](README.md) | **简体中文**

这是面向 Unitree G1 动作跟踪的 **SuperMimic 非官方复现（unofficial reproduction）**，基于 [mjlab](https://github.com/mujocolab/mjlab) 和 MuJoCo Warp。它通过仿真器采集的状态转移学习动力学模型，再利用模型中的想象轨迹（imagined rollouts）训练策略。

本代码库**仅在 LAFAN1 上进行了最基本的测试**，因此命名为 **Lite**。奖励项、权重和其他训练设置与原论文存在差异。下列结果反映的是本实现的表现，并非对原论文实验设置的完整复现。

## LAFAN1 实验结果

实验使用 40 段重定向到 G1 的动作，采样率为 50 Hz：

- **SuperMimic Lite：** 1 × NVIDIA L40S，**4,096 个并行环境**。
- **PPO 基线：** 8 × NVIDIA RTX 4090，**每张 GPU 8,192 个并行环境**，共 65,536 个。

以下完整动作评估于 2026-10-27 完成，使用 40 个并行环境，每段动作均从第 0 帧开始，随机种子为 42。两种方法采用相同的评估流程，关闭随机化以及重置、观测和动作噪声。动作执行到末尾且未触发失败终止，即记为成功。

| 方法 | Checkpoint | 日志记录的训练时间 | 真实环境步数 | 成功动作数 | 成功率 |
| --- | --- | ---: | ---: | ---: | ---: |
| PPO | `model_39999.pt` | 814.9 分钟（13.58 小时） | 62,914,560,000 | 24/40 | 60.0% |
| **SuperMimic Lite** | `model_13500.pt` | **211.5 分钟（3.52 小时）** | **138,371,072** | **39/40** | **97.5%** |

图中的训练曲线截至上述评估使用的 checkpoint，分别对应 PPO 的 40,000 次更新和 SuperMimic 的 13,500 次更新。真实环境步数统计所有 GPU、所有环境的仿真状态转移；SuperMimic 的计数包含初始采样，不包含模型中的想象轨迹。SuperMimic 的平均动作完成度为 97.61%，唯一失败的动作是 `fallAndGetUp1_subject1`。

![LAFAN1 训练曲线：横轴为日志记录的训练时间](docs/results/lafan1_train-time.png)

两组实验使用的硬件和训练配置不同。日志记录的训练时间包括采样、学习及计入日志的编译时间，不包括启动准备、初始采样、checkpoint 保存和停机时间。PPO 的训练失败比例为累计统计；SuperMimic Lite 则统计负责记录日志的进程最近完成的 100 个 episode。这是在训练动作库上的基础对比，并非独立测试集上的基准评测。

## 安装

先安装 [uv](https://docs.astral.sh/uv/)，然后在仓库根目录执行以下命令。支持 Python 3.11–3.13，仓库默认使用 3.13。

```bash
uv sync --locked --extra cu128 --no-dev
```

安装时会自动获取固定版本的官方 mjlab 依赖。若使用 CPU，将 `--extra cu128` 替换为 `--extra cpu`，并在下方运行命令中选择 `--device cpu`。使用 `uv run --no-sync` 可保持已安装的 Torch 版本及其 CPU/CUDA 配置。

## 从 Hugging Face 下载 LAFAN1

**本仓库不提供 LAFAN1，也不分发上述实验使用的处理后动作文件。** 可从 [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) 下载公开的 G1 重定向数据。其中 `g1/` 目录包含 30 Hz 的 CSV 文件，每帧存有根节点位置、按 `xyzw` 排列的根节点四元数和 29 个关节角。该上游重定向数据是独立的数据来源；上方历史实验结果不代表下述转换流程所生成数据上的表现。

1. 使用 [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) 仅下载 G1 的 CSV 文件：

   ```bash
   uvx --from huggingface-hub hf download lvhaidong/LAFAN1_Retargeting_Dataset \
     --repo-type dataset --include 'g1/*.csv' \
     --local-dir motions/lafan1_hf
   ```

2. 在本地将其转换为跟踪任务所需的 50 Hz NPZ 格式。转换使用 CPU，对轨迹重采样并计算各刚体的运动学信息：

   ```bash
   uv run --no-sync supermimic data csv \
     --input-dir motions/lafan1_hf/g1 \
     --output-dir motions/lafan1_g1_rtgt_50hz \
     --input-fps 30 --output-fps 50
   ```

3. 按下一节说明，将生成的目录设为 `motion_folder`。下载的 CSV 和生成的 NPZ 均放在 Git 忽略的 `motions/` 目录中。使用数据时请遵守上游数据集的条款。

首次使用仓库时，请先转换数据，再创建资源配置。若已有配置，请选择不在 `protected_roots` 内的输出目录。恢复训练时应使用原来的动作文件。

## 配置资源路径

Checkpoint 和原始日志保留在本地，不纳入 Git；后续将另行安排 checkpoint 的 Hugging Face 托管。

```bash
cp assets.example.toml .supermimic.local.toml
```

编辑 `motion_folder`，使其指向 50 Hz 跟踪 NPZ 文件所在的目录。回放、评估或恢复训练时，还需要设置 `checkpoint` 和 `run_dir`；运行目录中必须包含 `params/env.yaml` 和 `params/agent.yaml`。从头训练只需配置动作目录。

路径相对于 TOML 文件解析，也可以指向仓库外的目录。命令行参数会覆盖配置中的路径。本地配置文件已被 Git 忽略；`protected_roots` 用于防止输出覆盖需要保留的资源目录。

## 训练

```bash
# 单张 GPU；默认使用 4,096 个环境，训练 10,000 次学习迭代。
uv run --no-sync supermimic train --device cuda:0

# 两张 GPU，每张 GPU 使用 4,096 个环境。
uv run --no-sync supermimic train --gpu-ids '[0,1]'
```

一次迭代指一次完整的学习更新，包括世界模型、策略和 Critic 的更新。`--max-iterations 10000` 设置的是目标总迭代数。训练会将 checkpoint、配置快照和 TensorBoard 日志写入 `checkpoints/`；可用 `--output-dir` 指定新的运行目录。首次 CUDA 学习更新包含编译时间。

更多选项可通过 `uv run --no-sync supermimic train --help-full` 查看，例如 `--env.scene.num-envs 1024`。

### 恢复训练

```bash
uv run --no-sync supermimic train --resume \
  --checkpoint /path/to/run/model_5000.pt \
  --max-iterations 10000 \
  --output-dir checkpoints/resumed_run
```

从第 5,000 次迭代恢复、目标设为 10,000 时，会再运行 5,000 次迭代。恢复训练需要原动作数据集、配置快照，以及所有匹配的 `model_N.rankR.replay.pt` 回放缓冲区分片。环境数量、训练进程数量（rank 数）和回放缓冲区维度必须与保存时一致。

## 回放与评估

```bash
# 交互式查看器，需要显示环境。
uv run --no-sync supermimic play --motion dance1_subject1

# 无窗口回放。
uv run --no-sync supermimic play --headless --steps 100 --device cuda:0

# 从每段动作的首帧开始评估。
uv run --no-sync supermimic eval --num-envs 40 --device cuda:0 \
  --output-dir logs/eval/my_run
```

`--motion` 使用不含 `.npz` 后缀的文件名选择动作。可通过 `--checkpoint`、`--motion-folder` 和 `--run-dir` 覆盖配置中的输入路径。回放和评估只需要主 checkpoint 与配置快照，无需回放缓冲区分片。请仅加载可信来源的 checkpoint。

评估会生成 `results.json`、`summary.csv` 和 `per_motion.csv`。输出目录必须尚不存在或为空。

## 绘制训练曲线

绘制一次训练运行的曲线：

```bash
uv run --no-sync supermimic plot --source /path/to/run --output-dir logs/plots
```

对比图包含奖励、episode 长度、失败比例和刚体位置误差四个子图。数据可用时，图例会标注日志记录的训练时间和真实环境步数。使用 JSON 数据源配置时，可将 `max_iteration` 设为最后纳入绘图的迭代索引，索引从 0 开始；对于 SuperMimic 的 `model_N.pt`，应设为 `N - 1`，使曲线与该 checkpoint 对齐。

## 许可证与致谢

代码采用 [Apache-2.0](LICENSE) 许可证，由 [zzhanYoung](https://github.com/zzhanYoung) 维护。SuperMimic Lite 是非官方复现，原始研究工作归功于 SuperMimic 作者。

任务配置、动作处理工具和环境步进实现部分衍生自 mjlab；mjlab 由 MuJoCo Lab 贡献者维护，采用 Apache-2.0 许可证。框架和机器人资源从官方上游安装。MuJoCo 类型声明生成脚本保留在 `scripts/` 中，其生成的 `typings/` 仅用于本地开发，不纳入 Git 或发布归档。

Checkpoint、动作数据和原始日志不包含在源码发行包中；它们的来源和分发条款独立于代码许可证。LAFAN1 的相关条款见[上游数据集说明](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)。

## 模型输入与预测

以下说明对应当前 G1 实现。Actor 使用短时间的反馈历史和未来参考动作；Critic 和世界模型还使用训练时可从仿真器获得的额外信息。

| 组件 | 输入选择 | 输出与训练方式 |
| --- | --- | --- |
| **Actor（策略）** | 10 帧反馈历史，每帧 99 维：躯干相对参考的朝向（6）、IMU 角速度（3）、投影重力（3）、相对关节位置（29）、关节速度（29）和上一时刻动作（29）；另有 10 帧未来参考，共 730 维。 | 将未来参考编码为 32 维潜变量，与 990 维反馈历史拼接，预测 29 维关节位置动作；随后按照任务的动作配置缩放并加上偏移。 |
| **Critic（价值网络）** | 914 维特权信息：消除航向影响的根节点与关节状态（71）、关节／刚体／根节点跟踪误差（80）、Context（33）和未来参考（730）。 | 预测标量价值，以想象轨迹计算的 lambda-return 为训练目标，目标不参与梯度回传。 |
| **World Model State（世界模型状态）** | 物理状态共 71 维：根节点位置（3）、四元数（4）、关节位置（29）、根节点线速度与角速度（6）、关节速度（29）。模型同时接收当前动作（29）和 Context（33）。 | 预测 71 维状态增量，加到当前物理状态上，再归一化得到的四元数。当前动作被写入下一状态的“上一时刻动作”字段。 |
| **Context（环境上下文）** | 从仿真器读取的足部共用摩擦系数（1）、关节编码器偏差（29）和躯干质心偏移（3）。 | 作为世界模型和 Critic 的输入，在每条想象轨迹内保持固定；不通过学习得到的估计器预测。 |

**参考动作与反馈。** 10 帧未来目标直接读取动作片段的 `t+1` 至 `t+10` 帧，超出片段末尾时取最后一帧。每帧包含 73 维特征：在当前参考航向坐标系下的根节点位移和 6D 朝向、在各未来帧根节点坐标系下的根节点线速度与角速度、关节位置，以及乘以 0.1 的关节速度。Actor 的关节位置观测包含编码器偏差，但不直接接收特权 Context 输入。

**状态预测。** 存储的状态共 100 维，包括 71 维物理状态和 29 维上一时刻动作；模型只预测物理状态。根节点位置相对于环境原点，根节点四元数和线速度使用世界坐标轴，角速度使用根节点自身坐标轴。世界模型在回放缓冲区的 8 步窗口上进行自回归拟合，损失使用归一化的几何状态误差，包括旋转误差。

**想象轨迹中的学习。** Actor 通过 32 步可微分想象轨迹优化；更新 Actor 时，世界模型和 Critic 的参数保持冻结。正向运动学重建各刚体状态，奖励和软终止信号通过解析计算得到。未来参考目标来自数据集，并随轨迹步进向前推进。Critic 使用 lambda-return，默认 `gamma=0.99`、`lambda=0.95`。

相关实现：[输入选择](src/supermimic/learning/adapter.py)、[网络结构](src/supermimic/learning/models.py)、[状态积分](src/supermimic/learning/dynamics.py)、[解析计算模块](src/supermimic/learning/heads.py)和[训练更新](src/supermimic/learning/algorithm.py)。
