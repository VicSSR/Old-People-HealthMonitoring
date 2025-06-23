from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import time

# 创建全局应用实例
app = Flask(__name__)
# 关键修改：添加 async_mode 和 logger 配置
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   async_mode='threading',
                   logger=True,  # 启用详细日志
                   engineio_logger=True)  # 启用 Engine.IO 日志

# WebSocket 连接事件
@socketio.on('connect')
def handle_connect():
    print('客户端已连接')
    # 发送测试消息验证连接
    emit('server_message', {'data': '连接成功！'})

# 添加根路由用于测试
@app.route('/')
def index():
    return "WebSocket 服务器运行中"

# 添加健康检查端点
@app.route('/health')
def health_check():
    return jsonify({"status": "ok", "service": "websocket-server"})

def send_data_to_frontend(data):
    """发送数据到前端（确保服务器已启动）"""
    global socketio
    if socketio:
        try:
            socketio.emit('fall_data', data)
            print(f"已发送数据到前端: {data}")
            return True
        except Exception as e:
            print(f"发送数据时出错: {str(e)}")
            return False
    else:
        print("SocketIO 未初始化，无法发送数据")
        return False

def start_server():
    """启动WebSocket服务器"""
    print("正在启动WebSocket服务器...")
    try:
        # 关键修改：使用 socketio.run 而不是 app.run
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5000, 
                    debug=False, 
                    use_reloader=False,
                    allow_unsafe_werkzeug=True)
        print("WebSocket服务器已启动在端口 5000")
    except Exception as e:
        print(f"启动服务器失败: {str(e)}")

if __name__ == '__main__':
    start_server()