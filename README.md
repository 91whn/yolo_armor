# YOLO26n 装甲板模型视频推理与可视化

## 1. 项目完成的功能
本项目使用参考材料中提供的 YOLO26n 装甲板模型 `best.pt`，完成以下功能：
- **加载模型权重**：读取本地 `best.pt` 模型进行推理。
- **读取视频**：读取自己拍摄的装甲板视频。
- **视频逐帧推理**：调用 Ultralytics 框架对视频每一帧进行推理。
- **目标可视化**：在检测到目标时，显示检测框、类别、置信度，以及模型预测的 **4个关键点**（左上、左下、右下、右上）。
- **保存结果视频**：将绘制后的每一帧写入输出视频文件，内容与原视频对应，可正常播放。
- **关键点来源说明**：关键点数据严格来自模型的 `result.keypoints` 输出，**绝对没有使用检测框的四个角代替**。关键点使用不同颜色、连线、编号和名称进行标记，确保清晰可见且能看出所属目标。

## 2. Python版本、主要依赖、使用CPU还是GPU
- **Python 版本**：推荐 Python 3.8 或以上版本（本脚本在 Python 3.8/3.9 测试通过）。
- **主要依赖**：
  - `ultralytics >= 8.0.0` （YOLO26 推理框架）
  - `torch >= 1.10.0` （CPU 或 GPU 版本均可）
  - `torchvision >= 0.11.0`
  - `opencv-python >= 4.5.0`
  - `numpy >= 1.20.0`
- **CPU/GPU 支持**：脚本默认为 **CPU** 运行（参数 `--device cpu`），也支持通过 `--device 0` 切换为 **GPU** 运行。本次作业不要求实时推理，CPU 下虽然较慢但可以完整跑完。

## 3. 模型权重的准备方式与放置位置
- 参考材料中已提供 YOLO26n 装甲板模型权重文件 `best.pt`。
- 请将 `best.pt` 放置在项目根目录下（即与 `infer.py` 同级目录）。

## 4. 依赖安装步骤、实际运行命令和结果保存位置
### 依赖安装步骤：
```bash
# 1. 创建并激活虚拟环境（推荐，避免污染系统环境）
python -m venv venv
# Windows 激活: venv\Scripts\activate
# Linux/Mac 激活: source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 如果使用 GPU 推理，请安装对应版本的 torch（以 CUDA 11.8 为例）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

## 5. 视频拍摄地点、时长及场景说明
- **拍摄地点**：[例如：宿舍桌面 / 实验室]
- **视频时长**：[例如：11秒（总共331帧，30fps）]
- **场景说明**：[例如：固定机位拍摄，装甲板在画面中平移/旋转，模拟比赛识别场景]
- **视频分辨率**：[例如：1280x720]

## 6. 能展示检测框和关键点的结果截图
*(图片已放入项目 screenshots 文件夹)*

![检测框和关键点截图1](./screenshots/result1.jpg)
*图1：模型检测到装甲板目标，绿色框为检测框，彩色圆点+连线为模型预测的4个关键点。*

![检测框和关键点截图2](./screenshots/result2.jpg)
*图2：同一视频不同帧的推理结果，展示关键点跟随目标运动。*

![检测框和关键点截图3](./screenshots/result3.jpg)
*图3：另一帧的推理结果，关键点并非来自检测框角点，且标注清晰。*

## 7. 实际推理实现与遇到的问题及解决方案
- **实际推理实现**：直接运行 `python infer.py --model best.pt --source armor_video.mp4 --output output.mp4` 即可。
- **遇到的问题及解决**：
  1. 出现 `No such file or directory`：发现Windows隐藏了文件后缀，导致文件名变成 `infer.py.txt`，通过命令行 `ren infer.py.txt infer.py` 修复。
  2. 出现 `模型文件不存在: best.pt`：由于原文件名有空格（`best .pt`），通过 `ren "best .pt" best.pt` 修复。
  3. 视频名字重复（`armor_video.mp4.mp4`）：同样通过 `ren` 命令去除了多余后缀。
         



