import os
from flask import Flask, request, render_template, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
import json
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import glob

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

def find_existing_flashcards(filename):
    """Find existing flashcards file for the given markdown filename"""
    base_name = os.path.splitext(filename)[0]
    pattern = os.path.join(FLASHCARDS_FOLDER, f"{base_name}_*.json")
    files = glob.glob(pattern)
    
    if files:
        # 返回最新的文件
        latest_file = max(files, key=os.path.getctime)
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f), os.path.basename(latest_file)
    
    return None, None

def save_flashcards(original_filename, flashcards):
    """Save flashcards to a JSON file"""
    base_name = os.path.splitext(original_filename)[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{base_name}_{timestamp}_flashcards.json"
    filepath = os.path.join(FLASHCARDS_FOLDER, filename)
    
    # 在保存前处理数据
    processed_flashcards = []
    for card in flashcards:
        # 处理换行符：确保文本中的\n被保留为实际的换行符
        processed_card = {
            'question': card['question'].strip().replace('\\n', '\n'),
            'answer': card['answer'].strip().replace('\\n', '\n'),
            'importance': card['importance'],
            'probability': card['probability'],
            'learning_state': card['learning_state']
        }
        processed_flashcards.append(processed_card)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(processed_flashcards, f, ensure_ascii=False, indent=2)
    
    return filename

def generate_flashcards_stream(markdown_content, original_filename):
    """Generate flashcards with streaming progress"""
    try:
        # 首先检查是否存在现有的闪卡文件
        existing_flashcards, existing_filename = find_existing_flashcards(original_filename)
        if existing_flashcards:
            yield json.dumps({
                'type': 'result',
                'data': existing_flashcards,
                'filename': existing_filename,
                'message': '从现有文件加载闪卡'
            })
            return

        # 如果没有现有文件，生成新的闪卡
        yield json.dumps({
            'type': 'status',
            'data': '正在生成闪卡...'
        })

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

        content = response.choices[0].message.content

        try:
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
            
            # 为每个闪卡添加学习状态
            for card in flashcards:
                card['learning_state'] = {
                    'review_count': 0,
                    'last_review': None,
                    'next_review': None,
                    'ease_factor': 2.5,
                    'interval': 0
                }
                # 处理反斜杠，恢复LaTeX公式
                if '\\\\' in card['question']:
                    card['question'] = card['question'].replace('\\\\', '\\')
                if '\\\\' in card['answer']:
                    card['answer'] = card['answer'].replace('\\\\', '\\')
                # 处理换行符
                if '\\n' in card['question']:
                    card['question'] = card['question'].replace('\\n', '\n')
                if '\\n' in card['answer']:
                    card['answer'] = card['answer'].replace('\\n', '\n')
            
            saved_filename = save_flashcards(original_filename, flashcards)
            
            yield json.dumps({
                'type': 'result',
                'data': flashcards,
                'filename': saved_filename,
                'message': '成功生成新的闪卡'
            })

        except Exception as e:
            yield json.dumps({
                'type': 'error',
                'data': str(e)
            })
            
    except Exception as e:
        yield json.dumps({
            'type': 'error',
            'data': str(e)
        })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/practice/<filename>')
def practice(filename):
    try:
        filepath = os.path.join(FLASHCARDS_FOLDER, filename)
        if not os.path.exists(filepath):
            return render_template('error.html', message='找不到闪卡文件')
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()  # 移除任何多余的空白字符
            if content.endswith('%'):  # 移除可能的多余字符
                content = content[:-1]
            flashcards = json.loads(content)
        
        # 处理从文件读取的数据
        for card in flashcards:
            # 确保问题和答案是字符串并处理换行符
            card['question'] = str(card['question']).strip().replace('\\n', '\n')
            card['answer'] = str(card['answer']).strip().replace('\\n', '\n')
            # 处理LaTeX公式
            if '\\\\' in card['question']:
                card['question'] = card['question'].replace('\\\\', '\\')
            if '\\\\' in card['answer']:
                card['answer'] = card['answer'].replace('\\\\', '\\')
        
        return render_template('practice.html', flashcards=flashcards, filename=filename)
    except Exception as e:
        print(f"Error in practice route: {str(e)}")  # 添加服务器端日志
        return render_template('error.html', message=str(e))

@app.route('/api/update_card_state', methods=['POST'])
def update_card_state():
    try:
        data = request.json
        filename = data['filename']
        card_index = data['card_index']
        new_state = data['new_state']
        
        filepath = os.path.join(FLASHCARDS_FOLDER, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            flashcards = json.load(f)
        
        flashcards[card_index]['learning_state'] = new_state
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(flashcards, f, ensure_ascii=False, indent=2)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

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
        
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        return Response(
            stream_with_context(generate_flashcards_stream(markdown_content, filename)),
            mimetype='text/event-stream'
        )
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True) 