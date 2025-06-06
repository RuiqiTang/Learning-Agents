import os
from flask import Flask, request, render_template, jsonify
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

def create_flashcards(markdown_content):
    """Convert markdown content to flashcards using DeepSeek API"""
    try:
        prompt = """
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
            {
                "question": "什么是高斯过程(GP)的定义？",
                "answer": "高斯过程(GP)是一个随机过程，其任何有限次实现集合都具有联合多元正态分布。数学表示为：$f(x) \\sim GP(\\mu(x), k(x, x'))$，其中：\\n- $\\mu(x)$ 是均值函数\\n- $k(x, x')$ 是协方差函数",
                "importance": 5,
                "probability": 4
            }
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
            temperature=0.3,  # 降低温度以获得更确定性的输出
            max_tokens=4000,  # 增加最大token以处理长文本
            stream=False
        )
        
        # 获取响应内容
        content = response.choices[0].message.content
        print("="*10,"content: ","="*10,"\n", content)
        
        # 确保返回的是有效的JSON字符串
        try:
            # 尝试清理内容中的格式问题
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            flashcards = json.loads(content)
            if not isinstance(flashcards, list):
                raise ValueError("Response is not a list")
            return flashcards
        except json.JSONDecodeError:
            # 如果不是有效的JSON，尝试提取JSON部分
            import re
            json_match = re.search(r'\[\s*{.*}\s*\]', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    raise ValueError("Could not parse JSON from response")
            raise ValueError("Invalid JSON response")
            
    except Exception as e:
        print(f"Error creating flashcards: {str(e)}")
        print(f"API Response: {content if 'content' in locals() else 'No content'}")
        return []

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
        
        # Create flashcards
        flashcards = create_flashcards(markdown_content)
        
        if not flashcards:
            return jsonify({'error': 'Failed to create flashcards'}), 500
        
        # Save flashcards
        base_name = os.path.splitext(filename)[0]
        flashcards_path = os.path.join(FLASHCARDS_FOLDER, f"{base_name}_flashcards.json")
        with open(flashcards_path, 'w', encoding='utf-8') as f:
            json.dump(flashcards, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Flashcards created successfully',
            'flashcards': flashcards
        })
    
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