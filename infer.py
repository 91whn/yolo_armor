"""
YOLO26n 装甲板模型视频推理脚本
功能：加载模型 -> 视频逐帧推理 -> 绘制检测框/类别/置信度/4个关键点 -> 保存结果视频
"""

import argparse
import os
import sys

import cv2
from ultralytics import YOLO

# ============================================================
# 配置区域（可通过命令行参数覆盖）
# ============================================================
DEFAULT_MODEL_PATH = "best.pt"          # 模型权重路径
DEFAULT_VIDEO_PATH = "input.mp4"        # 输入视频路径
DEFAULT_OUTPUT_PATH = "output.mp4"      # 输出视频路径
DEFAULT_CONF = 0.25                     # 置信度阈值
DEFAULT_DEVICE = "cpu"                  # 推理设备: "cpu" 或 "0"(GPU)
DEFAULT_IMGSZ = 640                     # 推理输入尺寸

# 装甲板4个关键点名称（按标注顺序，使用英文避免OpenCV中文乱码）
KEYPOINT_NAMES = ["LU", "LD", "RD", "RU"]  # Left-Up, Left-Down, Right-Down, Right-Up

# 每个关键点的颜色 (BGR)
KEYPOINT_COLORS = [
    (0, 0, 255),    # 左上 - 红色
    (0, 255, 0),    # 左下 - 绿色
    (255, 0, 0),    # 右下 - 蓝色
    (0, 255, 255),  # 右上 - 黄色
]


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26n 装甲板模型视频推理与关键点可视化")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="模型权重路径")
    parser.add_argument("--source", type=str, default=DEFAULT_VIDEO_PATH, help="输入视频路径")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="输出视频路径")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="置信度阈值")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="推理设备: 'cpu' 或 '0'(GPU)")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="推理输入尺寸")
    return parser.parse_args()


def draw_keypoints(frame, keypoints_xy, keypoints_conf=None, conf_thresh=0.3):
    """绘制来自模型预测的关键点，不依赖检测框角点"""
    h, w = frame.shape[:2]
    num_kpts = len(keypoints_xy)
    pts = []
    
    # 绘制关键点连线（按装甲板四点顺序闭合连线）
    for i in range(num_kpts):
        x, y = int(keypoints_xy[i][0]), int(keypoints_xy[i][1])
        if x <= 0 and y <= 0:
            continue
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        pts.append((x, y))

    if len(pts) >= 2:
        for i in range(len(pts)):
            pt1 = pts[i]
            pt2 = pts[(i + 1) % len(pts)]
            cv2.line(frame, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

    # 绘制关键点圆点 + 编号标签
    for i in range(num_kpts):
        x, y = int(keypoints_xy[i][0]), int(keypoints_xy[i][1])
        if x <= 0 and y <= 0:
            continue
        if x < 0 or y < 0 or x >= w or y >= h:
            continue

        color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
        cv2.circle(frame, (x, y), 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 8, (255, 255, 255), 1, cv2.LINE_AA)

        label = f"{i}:{KEYPOINT_NAMES[i]}" if i < len(KEYPOINT_NAMES) else f"{i}"
        if keypoints_conf is not None and i < len(keypoints_conf):
            c = keypoints_conf[i]
            if c >= conf_thresh:
                label += f" {c:.2f}"

        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return frame


def draw_boxes_and_labels(frame, boxes_data, class_names):
    """绘制检测框、类别名称和置信度"""
    for box in boxes_data:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()

    # 1. 加载模型
    if not os.path.exists(args.model):
        print(f"[错误] 模型文件不存在: {args.model}")
        sys.exit(1)
    print(f"[信息] 加载模型: {args.model}")
    model = YOLO(args.model)

    # 2. 读取视频
    if not os.path.exists(args.source):
        print(f"[错误] 视频文件不存在: {args.source}")
        sys.exit(1)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {args.source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[信息] 视频: {args.source}  {width}x{height}  {fps:.1f}fps  {total_frames}帧")

    # 3. 初始化视频写入器
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[错误] 无法创建输出视频: {args.output}")
        cap.release()
        sys.exit(1)
    print(f"[信息] 输出视频将保存至: {args.output}")

    class_names = model.names if hasattr(model, "names") else {}
    frame_count = 0

    # 4. 逐帧推理
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.predict(source=frame, conf=args.conf, device=args.device, imgsz=args.imgsz, verbose=False)
        result = results[0]
        annotated_frame = frame.copy()

        # 绘制检测框
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            annotated_frame = draw_boxes_and_labels(annotated_frame, boxes, class_names)

        # 绘制关键点（严格来自模型预测）
        keypoints = result.keypoints
        if keypoints is not None and keypoints.xy is not None:
            xy_list = keypoints.xy.cpu().numpy()
            conf_list = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None

            for obj_idx in range(xy_list.shape[0]):
                kpt_xy = xy_list[obj_idx]
                kpt_conf = conf_list[obj_idx] if conf_list is not None else None
                annotated_frame = draw_keypoints(annotated_frame, kpt_xy, keypoints_conf=kpt_conf, conf_thresh=0.3)

        # 写入输出视频
        writer.write(annotated_frame)

        # 打印进度
        if frame_count % 50 == 0:
            print(f"[进度] {frame_count}/{total_frames} 帧")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n[完成] 共处理 {frame_count} 帧，推理结果视频已保存至: {args.output}")


if __name__ == "__main__":
    main()