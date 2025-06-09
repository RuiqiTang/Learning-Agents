import os
from flask import Flask, request, render_template, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
import json
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import glob
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Configure folders
UPLOAD_FOLDER = 'assets'
FLASHCARDS_FOLDER = 'flashcards'

# Configure OpenAI client for DeepSeek
api_key = os.getenv('DEEPSEEK_API_KEY')
if not api_key:
    logger.error("DEEPSEEK_API_KEY not found in environment variables")
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"  # 更新为正确的API端点
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
        
        # 确保LaTeX公式被正确保存
        def normalize_latex(text):
            # 确保LaTeX公式中的反斜杠是单个的
            text = text.replace('\\\\', '\\')
            return text
        
        processed_card['question'] = normalize_latex(processed_card['question'])
        processed_card['answer'] = normalize_latex(processed_card['answer'])
        
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
            content = f.read().strip()
            if content.endswith('%'):
                content = content[:-1]
            flashcards = json.loads(content)
        
        # 处理从文件读取的数据
        for card in flashcards:
            # 确保问题和答案是字符串并处理换行符
            card['question'] = str(card['question']).strip().replace('\\n', '\n')
            card['answer'] = str(card['answer']).strip().replace('\\n', '\n')
            
            # 清理和标准化公式
            card['question'] = normalize_latex_formula(card['question'])
            card['answer'] = normalize_latex_formula(card['answer'])
        
        return render_template('practice.html', flashcards=flashcards, filename=filename)
    except Exception as e:
        print(f"Error in practice route: {str(e)}")
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

def normalize_latex_formula(text):
    """规范化LaTeX公式，确保格式统一且正确"""
    import re

    def clean_formula(formula):
        """清理单个公式内容"""
        # 移除多余的空格
        formula = formula.strip()
        
        # 统一处理希腊字母和数学符号
        greek_letters = {
            'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'δ': '\\delta',
            'ε': '\\epsilon', 'θ': '\\theta', 'λ': '\\lambda', 'μ': '\\mu',
            'π': '\\pi', 'ρ': '\\rho', 'σ': '\\sigma', 'τ': '\\tau',
            'φ': '\\phi', 'ω': '\\omega', 'Γ': '\\Gamma', 'Δ': '\\Delta',
            'Θ': '\\Theta', 'Λ': '\\Lambda', 'Ξ': '\\Xi', 'Π': '\\Pi',
            'Σ': '\\Sigma', 'Φ': '\\Phi', 'Ψ': '\\Psi', 'Ω': '\\Omega'
        }
        
        # 替换Unicode希腊字母为LaTeX命令
        for greek, latex in greek_letters.items():
            formula = formula.replace(greek, latex)
        
        # 处理数学运算符和函数
        math_operators = {
            'min': '\\min', 'max': '\\max', 'exp': '\\exp', 'log': '\\log',
            'sin': '\\sin', 'cos': '\\cos', 'tan': '\\tan', 'sum': '\\sum',
            'prod': '\\prod', 'int': '\\int', 'rightarrow': '\\rightarrow',
            'leftarrow': '\\leftarrow', 'leftrightarrow': '\\leftrightarrow',
            'leq': '\\leq', 'geq': '\\geq', 'neq': '\\neq',
            'approx': '\\approx', 'sim': '\\sim', 'equiv': '\\equiv',
            'propto': '\\propto', 'infty': '\\infty', 'partial': '\\partial'
        }
        
        # 在单词边界处替换数学运算符
        for op, latex in math_operators.items():
            formula = re.sub(r'\b' + re.escape(op) + r'\b', latex, formula)
        
        # 处理上下标
        formula = re.sub(r'([a-zA-Z0-9])_([a-zA-Z0-9]+)', r'\1_{\\text{\2}}', formula)
        formula = re.sub(r'([a-zA-Z0-9])\^([a-zA-Z0-9]+)', r'\1^{\\text{\2}}', formula)
        
        # 确保分数格式正确
        formula = re.sub(r'\\frac\s*{([^}]+)}\s*{([^}]+)}', r'\\frac{\1}{\2}', formula)
        
        # 处理多重反斜杠
        formula = formula.replace('\\\\', '\\')
        
        return formula

    def process_text_segment(text):
        """处理文本段落，识别并转换数学表达式"""
        # 识别单个字母变量并转换为数学模式
        text = re.sub(r'(?<![\\$a-zA-Z])([x-zX-Z])(?![\\$a-zA-Z0-9])', r'$\1$', text)
        
        # 处理简单的上下标表达式
        text = re.sub(r'([a-zA-Z])_([a-zA-Z0-9])', r'$\1_{\2}$', text)
        text = re.sub(r'([a-zA-Z])\^([a-zA-Z0-9])', r'$\1^{\2}$', text)
        
        return text

    try:
        if not isinstance(text, str):
            return text

        # 1. 首先处理已有的数学公式
        def replace_math_formula(match):
            formula = match.group(1)
            return f"${clean_formula(formula)}$"

        def replace_display_formula(match):
            formula = match.group(1)
            return f"$${clean_formula(formula)}$$"

        # 处理行内公式
        text = re.sub(r'\$([^\$]+)\$', replace_math_formula, text)
        # 处理行间公式
        text = re.sub(r'\$\$([^\$]+)\$\$', replace_display_formula, text)
        
        # 2. 处理普通文本中的数学表达式
        segments = text.split('$')
        for i in range(0, len(segments), 2):
            segments[i] = process_text_segment(segments[i])
        text = '$'.join(segments)
        
        # 3. 清理和最终格式化
        # 移除重复的数学模式标记
        text = re.sub(r'\$\s*\$', '', text)
        text = re.sub(r'\$(\s*\$\s*\$\s*)\$', r'$$\1$$', text)
        
        # 确保公式周围有适当的空格
        text = re.sub(r'([^\s])\$', r'\1 $', text)
        text = re.sub(r'\$([^\s])', r'$ \1', text)
        
        return text
    except Exception as e:
        logger.error(f"Error in normalize_latex_formula: {str(e)}")
        return text  # 如果处理失败，返回原始文本

def format_deepseek_prompt():
    """生成用于DeepSeek的数学公式格式化提示"""
    return """
    在回答中，请严格遵循以下数学公式格式规范：
    
    1. 行内公式：
       - 使用单个美元符号：$formula$
       - 示例：$x$, $\\alpha$, $f(x)$
       - 变量、参数等单个符号也要使用数学模式：$x$ 而不是 x
    
    2. 行间公式：
       - 使用双美元符号：$$formula$$
       - 示例：$$\\min\\left(1, \\frac{\\pi(x')}{\\pi(x)}\\right)$$
       - 重要的多行公式或推导过程使用行间公式
    
    3. 数学符号规范：
       - 希腊字母：$\\alpha$, $\\beta$, $\\pi$ 等
       - 数学函数：$\\min$, $\\max$, $\\exp$, $\\log$ 等
       - 关系运算符：$\\leq$, $\\geq$, $\\neq$, $\\approx$ 等
       - 上下标使用花括号：$x_{t}$, $x^{2}$, $x_{i,j}$ 等
       - 分式：$\\frac{numerator}{denominator}$
    
    4. 格式化规则：
       - 所有数学符号和变量都必须在数学模式中
       - 公式前后要有适当的空格
       - 不要使用 Unicode 数学符号，使用 LaTeX 命令
       - 避免使用 \\[...\\] 或 \\(...\\) 格式
    """

@app.route('/api/ask_deepseek', methods=['POST'])
def ask_deepseek():
    try:
        logger.debug("Received request to /api/ask_deepseek")
        data = request.json
        question = data.get('question')
        context = data.get('context')
        
        if not question or not context:
            logger.error("Missing question or context in request")
            return jsonify({
                'success': False,
                'error': 'Missing question or context'
            }), 400
        
        # 构建提示词
        prompt = f"""
        作为一个专业的数学和机器学习教育专家，请基于以下内容回答问题。
        请确保回答准确、清晰，并保持与原文一致的专业水平。
        
        {format_deepseek_prompt()}

        上下文内容：
        {context}

        问题：
        {question}
        """
        
        logger.debug("Calling DeepSeek API")
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的数学和机器学习教育专家，精通解释复杂的数学概念。" + format_deepseek_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000,
                stream=False
            )
            
            logger.debug("Received response from DeepSeek API")
            answer = response.choices[0].message.content.strip()
            
            # 清理和标准化公式
            processed_answer = normalize_latex_formula(answer)
            logger.debug(f"Processed answer: {processed_answer}")
            
            return jsonify({
                'success': True,
                'answer': processed_answer
            })
        except Exception as api_error:
            logger.error(f"DeepSeek API error: {str(api_error)}")
            return jsonify({
                'success': False,
                'error': f"DeepSeek API error: {str(api_error)}"
            }), 500
            
    except Exception as e:
        logger.error(f"Error in ask_deepseek: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/update_card', methods=['POST'])
def update_card():
    try:
        data = request.json
        filename = data['filename']
        card_index = data['card_index']
        update_type = data['update_type']  # 'append' 或 'new'
        new_content = data['content']
        
        # 清理和标准化公式
        new_content = normalize_latex_formula(new_content)
        
        filepath = os.path.join(FLASHCARDS_FOLDER, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            flashcards = json.load(f)
        
        if update_type == 'append':
            # 在现有卡片中添加新内容
            card = flashcards[card_index]
            card['answer'] = card['answer'].strip() + '\n\n补充内容：\n' + new_content.strip()
        else:  # 'new'
            # 创建新卡片
            new_card = {
                'question': f"补充问题：{data['question']}",
                'answer': new_content.strip(),
                'importance': flashcards[card_index]['importance'],
                'probability': flashcards[card_index]['probability'],
                'learning_state': {
                    'review_count': 0,
                    'last_review': None,
                    'next_review': None,
                    'ease_factor': 2.5,
                    'interval': 0
                }
            }
            flashcards.append(new_card)
        
        # 保存更新后的闪卡
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(flashcards, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'flashcards': flashcards
        })
    except Exception as e:
        logger.error(f"Error in update_card: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True) 