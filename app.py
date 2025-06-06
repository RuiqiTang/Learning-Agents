import os
from flask import Flask, request, render_template, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
import json
from openai import OpenAI
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Configure folders
UPLOAD_FOLDER = 'assets'
FLASHCARDS_FOLDER = 'flashcards'

# Configure OpenAI client for DeepSeek
client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY', ''),
    base_url="https://api.deepseek.com"
)

# Ensure directories exist
for folder in [UPLOAD_FOLDER, FLASHCARDS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'md', 'markdown'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_flashcards_stream(markdown_content):
    """Generate flashcards with streaming progress"""
    try:
        prompt = \
        """
        你是一个专业的数学和机器学习教育专家。请将以下Markdown笔记内容转换为FlashCards格式。

        要求：
        1. 识别并保留所有LaTeX数学公式，确保其格式正确
        2. 根据知识点的重要性和考试出现概率进行拆分
        3. 每个知识点应该是完整的概念，包含必要的上下文
        4. 对于数学证明或推导，应该将过程拆分为合理的步骤
        5. 每个FlashCard包含：
           - question: 问题（可以是概念解释、证明步骤、公式推导等）
           - answer: 详细答案（包含完整的解释和相关公式）
           - importance: 重要性 (1-5，5最重要)
           - probability: 考试概率 (1-5，5最可能)
        6. 返回JSON数组格式，确保所有LaTeX公式都正确保留

        示例输出格式：
        [
            {{
                "question": "什么是高斯过程(GP)的定义？",
                "answer": "高斯过程(GP)是一个随机过程，其任何有限次实现集合都具有联合多元正态分布。数学表示为：$f(x) \\sim GP(\\mu(x), k(x, x'))$，其中：\\n- $\\mu(x)$ 是均值函数\\n- $k(x, x')$ 是协方差函数",
                "importance": 5,
                "probability": 4
            }}
        ]

        笔记内容：
        {content}
        """.format(content=markdown_content)

        # 发送开始消息
        yield json.dumps({
            'type': 'status',
            'data': '正在生成闪卡...'
        })

        # 使用stream=True来获取实时响应
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的数学和机器学习教育专家，精通将复杂的数学概念转换为清晰的FlashCards。请确保保留所有LaTeX公式，并确保输出格式严格符合JSON规范。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=4000,
            stream=False
        )
        print("="*10,"response loaded","="*10)

        # 合并所有内容
        content = response.choices[0].message.content

        # 处理JSON响应
        try:
            # 清理内容
            content = content.strip()
            if content.startswith('```json'):
                content = content[len("```json"):].strip()
            if content.endswith('```'):
                content = content[:-len("```")].strip()
            content = content.lstrip('\ufeff')
            content = content.replace('\\', '\\\\')
            
            flashcards = json.loads(content)
            if not isinstance(flashcards, list):
                raise ValueError("Response is not a list")
            
            # 发送成功结果
            yield json.dumps({
                'type': 'result',
                'data': flashcards
            })

        except Exception as e:
            # 发送错误信息
            yield json.dumps({
                'type': 'error',
                'data': str(e)
            })
            
    except Exception as e:
        # 发送错误信息
        yield json.dumps({
            'type': 'error',
            'data': str(e)
        })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read markdown content
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        # 返回流式响应
        return Response(
            stream_with_context(generate_flashcards_stream(markdown_content)),
            mimetype='text/event-stream'
        )
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/flashcards/<filename>')
def get_flashcards(filename):
    try:
        flashcards_path = os.path.join(FLASHCARDS_FOLDER, filename)
        with open(flashcards_path, 'r', encoding='utf-8') as f:
            flashcards = json.load(f)
        return jsonify(flashcards)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    app.run(debug=True) 