# pigdet 实验环境

实际环境位置：`D:\Python_Resources\Envs\pigdet`。Python：`3.11.17`。

激活后进入对应项目：

```powershell
conda activate pigdet
Set-Location -LiteralPath 'D:\Projects\Pig_Detection_test'
python -c "import sys, torch; print(sys.executable); print(torch.cuda.is_available())"
```

如按名称激活失败，使用完整路径：`conda activate D:\Python_Resources\Envs\pigdet`。

| 组件 | 实际版本 |
|---|---|
| torch | `2.5.1+cu121` |
| torchvision | `0.20.1+cu121` |
| numpy | `1.26.4` |
| opencv-python | `4.11.0.86` |
| ultralytics | `8.4.183` |
| yolox | `0.3.0` |
| pycocotools | `2.0.11` |
| onnx | `1.23.2` |
| onnxsim | `0.4.36` |

实际设备：`NVIDIA GeForce RTX 2050`，显存约 `4.0 GiB`；PyTorch 内置 CUDA runtime 为 `12.1`。驱动支持的最高 CUDA 版本与包内 runtime 不要求数字相同。本项目不要求另行安装同版本完整 CUDA Toolkit 才能使用预编译 PyTorch wheel。

环境检测：**已通过**。六种检测方法均完成无预训练权重、小尺寸合成输入的 CUDA 前向；COCO 完全匹配合成样例 AP 接近 1。 `pip check` 通过。详细结果见 `verification.json`。

这些仅证明环境与主要代码路径能够工作，不代表已完成数据转换、正式训练、精度评测或大分辨率显存测试。本机默认batch=4；服务器已完成的100轮实验使用RTX 3090，不能把服务器验证当作本机4GB显存验收。

## 已保存的环境记录

- `requirements-lock.txt`：本次安装全部 pip 版本；YOLOX 本地 editable 路径也会记录。
- `environment.yml`：Conda 环境导出，包含 pip 包。
- `conda-explicit.txt`：Conda 原生包精确列表，不包含全部 pip 包。
- `pip-check.txt`：依赖一致性检查。
- `verification.json`：验证范围、版本及设备。
- `sources.json`：第三方源码来源与提交。

## 从零重建的顺序

当前环境已经创建，不需要重复执行创建命令。重建到一个尚不存在的新目录时，按以下顺序操作，并保持自己的目标路径一致：

```powershell
# 以下 prefix 为原环境位置，已经存在时不要再次创建或覆盖。
conda create --prefix 'D:\Python_Resources\Envs\pigdet' python=3.11 pip -y
conda activate 'D:\Python_Resources\Envs\pigdet'
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r 'D:\Projects\Pig_Detection_test\requirements.txt'
```

GPU 轮子来源：[PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。主 `requirements.txt` 需要先安装上面的 CUDA PyTorch 对；不要在未指定 CUDA wheel 来源的干净环境中仅凭 pip 默认索引重建。

YOLOX 源码实际位于 `tools\third_party\YOLOX`，使用官方仓库固定提交。为兼容 Python 3.11 的现成 Windows wheel，将其 `requirements.txt` 中 `onnx-simplifier==0.4.10` 改为 `onnxsim==0.4.36`，补丁保存于本目录。未修改检测算法逻辑。

源码已通过 editable 方式安装。重建时先克隆固定提交、应用依赖补丁，再执行：

```powershell
python -m pip install --no-build-isolation --no-deps -e 'D:\Projects\Pig_Detection_test\tools\third_party\YOLOX'
```


## 再次验证

```powershell
conda activate pigdet
Set-Location -LiteralPath 'D:\Projects\Pig_Detection_test'
python tools/check_environment.py
```

环境激活后设置了 `PYTHONNOUSERSITE=1`，避免混用用户级包；`TORCH_HOME` 指向 D 盘模型目录、`HF_HOME` 指向 D 盘缓存，`YOLO_CONFIG_DIR` 指向本项目 `experiments/environment/ultralytics`。
