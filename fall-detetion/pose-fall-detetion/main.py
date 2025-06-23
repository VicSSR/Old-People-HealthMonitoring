import threading
import cv2
import time
import os
import numpy as np
import requests
from tracker import BYTETracker, Object
from pose import Yolov8Pose
from server import start_server, send_data_to_frontend  # 导入服务器启动函数
import multiprocessing

# 模型路径
YOLOV8_PT = "../runs/pose/train/weights/best.pt"
SAVE_PATH = "../outputs"

# 全局变量，用于跟踪服务器状态
websocket_initialized = False

# 在文件顶部添加
from server import socketio, app  # 确保导入核心对象

# 修改 init_websocket 函数
def init_websocket():
    """初始化WebSocket连接"""
    global websocket_initialized
    
    if not websocket_initialized:
        print("正在初始化WebSocket连接...")
        
        # 在后台线程中启动服务器
        server_thread = threading.Thread(target=start_server)
        server_thread.daemon = True
        server_thread.start()
        
        # 等待服务器启动
        time.sleep(2)
        
        # 测试服务器是否就绪
        try:
            response = requests.get('http://localhost:5000/health')
            if response.status_code == 200 and response.json().get('status') == 'ok':
                print("WebSocket服务器已就绪")
                websocket_initialized = True
            else:
                print("WebSocket服务器健康检查失败")
        except requests.ConnectionError:
            print("无法连接到WebSocket服务器")

def process_camera(camera_index: int = 0):
    """
    处理摄像头实时输入
    """
    # 初始化WebSocket
    init_websocket()
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"无法打开摄像头 (索引: {camera_index})")
        return

    # 获取摄像头属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {frame_width}x{frame_height}")

    # 初始化跟踪器
    tracker = BYTETracker(frame_rate=144, track_buffer=50)

    fps = 0
    FallenId = 583
    frame_count = 0
    last_time = time.time()
    fall_time = time.time()
    speed_error = False
    while True:
        # 计算FPS
        current_time = time.time()
        elapsed_time = current_time - last_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
        else:
            fps = 0
        
        # 重置计数器和时间
        if elapsed_time >= 1.0:
            frame_count = 0
            last_time = current_time

        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break

        # 目标检测
        results = yolov8_pose.detect_yolov8(frame)
        
        if results:
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy().tolist()[0]
                    width = x2 - x1
                    height = y2 - y1
                    prob = box.conf.cpu().numpy().tolist()[0]
                    cls_id = box.cls.cpu().numpy().tolist()[0]
                    
                    if cls_id == 0:  # 类别0是人
                        detections.append(Object((x1, y1, width, height), prob))

                # 更新跟踪器
                tracks = tracker.update(detections)

                # 绘制跟踪结果并进行跌倒检测
                for track in tracks:
                    tlbr = track.tlbr
                    track_id = track.track_id
                    speed = track.velocity

                    # 计算三维矢量速度
                    vx = speed[0]
                    vy = speed[1]
                    vz = speed[3]
                    speed_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)

                    cv2.putText(frame, f"FPS: {fps:.0f}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # 计算实际速度
                    speed_magnitude_mps = speed_magnitude * fps
                    
                    # 绘制跟踪ID
                    cv2.putText(frame, f"ID: {track_id:.0f}", (int(tlbr[0]), int(tlbr[1]) - 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    
                        

                    # 速度报警
                    if speed_magnitude_mps > 300:
                        speed_error = True
                        fall_time = time.time()
                        speed_text = f"SpeedError: {speed_magnitude_mps:.2f}"
                        cv2.putText(frame, speed_text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        if time.time()-fall_time>10:  
                            
                            speed_error = False

                    # print(time.time()-fall_time)
                    
                    # 获取关键点
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        keypoints = result.keypoints.xy.cpu().numpy().tolist()[0]
                        confidences = result.keypoints.conf.cpu().numpy().tolist()[0]
                        fall = yolov8_pose.fall_estimate(keypoints, confidences)
                        
                        # 跌倒报警
                        if fall and (fall_time == 0 or time.time()-fall_time>10 ):
                            # 准备发送到前端的数据
                            data = {
                                "altumcareFallenId": str(FallenId),
                                "actionType": "SpeedError" if speed_error else "Fallen",
                                "alertId": "684aba0a400053263775b308",
                                "serialNumber": "23CFAE9B98BC82EE",
                                "groupId": "5662",
                                "userId": "6378",
                                "roomId": "16365",
                                "roomName": "bisai",
                                "personId": "-2",
                                "personName": "people",
                                "timeStamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "trackId": track_id
                            }
                            fall_time = time.time()
                            # 使用 WebSocket 发送数据到前端
                            if send_data_to_frontend(data):
                                print(f"成功发送跌倒警报 (ID: {track_id})")
                            else:
                                print(f"发送跌倒警报失败 (ID: {track_id})")
                            
                            # 更新状态
                            if speed_error and time.time()-fall_time>50:
                                
                                speed_error = False
                            else:
                                FallenId += 1
                            
                            cv2.putText(frame, 'Fall Detected', (0, 75), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 显示图像
        cv2.namedWindow("Camera Detection Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Detection Result", 1980, 1080)
        result = cv2.resize(frame, (1980, 1080), interpolation=cv2.INTER_AREA)
        cv2.imshow("Camera Detection Result", result)

        frame_count += 1

        # 按 'q' 键退出
        if cv2.waitKey(1) == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    主函数
    """
    global yolov8_pose
    
    # 创建模型实例
    yolov8_pose = Yolov8Pose(YOLOV8_PT)
    
    import argparse
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLOv8-Pose人体姿态估计')
    parser.add_argument('input_type', type=str, help='输入类型: image, video 或 camera')
    parser.add_argument('input_path', type=str, nargs='?', default="0", help='输入文件路径或摄像头索引')
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(SAVE_PATH, exist_ok=True)

    # 根据输入类型处理
    if args.input_type == 'camera':
        try:
            camera_index = int(args.input_path)
            process_camera(camera_index)
        except ValueError:
            print("摄像头索引必须是整数！")
            exit(-1)
    else:
        print("目前仅支持 camera 模式")
        exit(-1)

if __name__ == '__main__':
    main()