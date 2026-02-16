"""
MNIST 手写数字推理服务

提供两种方式:
1. Web 界面: 打开浏览器绘制数字
2. API 接口: POST 请求上传图片
"""
import os
import io
import base64
import torch
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify, render_template_string
from model import MNISTClassifier


app = Flask(__name__)

# 全局变量: 模型和预处理
model = None
device = None
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def load_model():
    """加载训练好的模型"""
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MNISTClassifier().to(device)

    # 加载模型权重
    model_path = os.path.join(os.path.dirname(__file__), 'mnist_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: {model_path} not found, using untrained model")
        print("Please run train.py first to train and save the model")

    model.eval()


def preprocess_image(image_data):
    """
    预处理图像

    Args:
        image_data: PIL Image 或 base64 编码的图像

    Returns:
        预处理后的 tensor, shape=(1, 1, 28, 28)
    """
    if isinstance(image_data, str):
        # Base64 解码
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
    else:
        image = image_data

    # 转为灰度图
    image = image.convert('L')

    # 注意: Web 画布已经是黑底白字，和 MNIST 格式一致，无需反转
    # 如果输入是白底黑字（如扫描文档），则需要反转

    # 预处理
    tensor = transform(image).unsqueeze(0)  # 添加 batch 维度

    return tensor.to(device)


def predict(tensor):
    """
    预测数字

    Args:
        tensor: 预处理后的图像 tensor

    Returns:
        predicted_digit: 预测的数字 (0-9)
        probabilities: 各类别的概率
    """
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted = torch.argmax(probabilities, dim=1)

    return predicted.item(), probabilities[0].cpu().tolist()


# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>MNIST 手写数字识别</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            text-align: center;
            max-width: 500px;
            width: 100%;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 24px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .canvas-container {
            border: 3px solid #ddd;
            border-radius: 10px;
            display: inline-block;
            margin-bottom: 20px;
            background: #000;
        }
        #canvas {
            cursor: crosshair;
            display: block;
        }
        .buttons {
            margin-bottom: 20px;
        }
        button {
            padding: 12px 30px;
            margin: 0 5px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-predict {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-predict:hover { transform: scale(1.05); }
        .btn-clear {
            background: #f0f0f0;
            color: #333;
        }
        .btn-clear:hover { background: #e0e0e0; }
        .result {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            display: none;
        }
        .prediction {
            font-size: 72px;
            font-weight: bold;
            color: #667eea;
            line-height: 1;
        }
        .confidence {
            color: #666;
            margin-top: 10px;
        }
        .probabilities {
            margin-top: 20px;
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 5px;
        }
        .prob-item {
            width: 40px;
            text-align: center;
        }
        .prob-bar {
            height: 60px;
            background: #e0e0e0;
            border-radius: 5px;
            position: relative;
            overflow: hidden;
        }
        .prob-fill {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: height 0.3s;
        }
        .prob-label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .prob-value {
            font-size: 10px;
            color: #999;
        }
        .api-info {
            margin-top: 30px;
            padding: 15px;
            background: #f0f0f0;
            border-radius: 10px;
            font-size: 12px;
            color: #666;
            text-align: left;
        }
        .api-info code {
            background: #e0e0e0;
            padding: 2px 6px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>MNIST 手写数字识别</h1>
        <p class="subtitle">在下方画布中写入 0-9 的数字</p>

        <div class="canvas-container">
            <canvas id="canvas" width="280" height="280"></canvas>
        </div>

        <div class="buttons">
            <button class="btn-predict" onclick="predict()">识别</button>
            <button class="btn-clear" onclick="clearCanvas()">清空</button>
        </div>

        <div class="result" id="result">
            <div class="prediction" id="prediction">-</div>
            <div class="confidence">置信度: <span id="confidence">0</span>%</div>
            <div class="probabilities" id="probabilities"></div>
        </div>

        <div class="api-info">
            <strong>API 接口:</strong><br>
            POST /api/predict<br>
            Body: {"image": "base64_encoded_image"}
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;

        // 初始化画布
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // 鼠标事件
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);

        // 触摸事件
        canvas.addEventListener('touchstart', handleTouchStart);
        canvas.addEventListener('touchmove', handleTouchMove);
        canvas.addEventListener('touchend', stopDrawing);

        function startDrawing(e) {
            isDrawing = true;
            draw(e);
        }

        function draw(e) {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
        }

        function stopDrawing() {
            isDrawing = false;
            ctx.beginPath();
        }

        function handleTouchStart(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        }

        function handleTouchMove(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        }

        function clearCanvas() {
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('result').style.display = 'none';
        }

        function predict() {
            const imageData = canvas.toDataURL('image/png');

            fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image: imageData})
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').style.display = 'block';
                document.getElementById('prediction').textContent = data.prediction;
                document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1);

                // 显示概率分布
                const probsContainer = document.getElementById('probabilities');
                probsContainer.innerHTML = '';
                data.probabilities.forEach((prob, i) => {
                    const item = document.createElement('div');
                    item.className = 'prob-item';
                    item.innerHTML = `
                        <div class="prob-bar">
                            <div class="prob-fill" style="height: ${prob * 100}%"></div>
                        </div>
                        <div class="prob-label">${i}</div>
                        <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
                    `;
                    probsContainer.appendChild(item);
                });
            })
            .catch(error => {
                alert('识别失败: ' + error);
            });
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Web 界面"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API 接口"""
    try:
        data = request.get_json()
        image_data = data.get('image', '')

        # 预处理
        tensor = preprocess_image(image_data)

        # 预测
        predicted, probabilities = predict(tensor)

        return jsonify({
            'success': True,
            'prediction': predicted,
            'confidence': probabilities[predicted],
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    print("=" * 60)
    print("MNIST 手写数字识别 Web 服务")
    print("=" * 60)

    # 加载模型
    load_model()

    print(f"\n设备: {device}")
    print("\n服务启动中...")
    print("  Web 界面: http://localhost:5001")
    print("  API 接口: POST http://localhost:5001/api/predict")
    print("\n按 Ctrl+C 停止服务")
    print("-" * 60)

    app.run(host='0.0.0.0', port=5001, debug=False)
