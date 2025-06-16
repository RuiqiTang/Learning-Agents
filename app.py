import os
from flask import Flask, request, render_template, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
import json
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import glob
import logging
import re
from database import Database
from config import UPLOAD_FOLDER, FLASHCARDS_FOLDER
from algo.supermemo import get_next_review_date

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
db = Database()

# 初始化数据库
try:
    # 重新初始化数据库
    db.init_db()
    logger.info("Database initialized successfully")
    
    # 导入现有的闪卡数据
    imported_count = db.import_from_json()
    logger.info(f"Imported {imported_count} flashcards from existing JSON files")
except Exception as e:
    logger.error(f"Error during startup: {str(e)}")
    raise

# Load environment variables
load_dotenv()

# Configure OpenAI client for DeepSeek
api_key = os.getenv('DEEPSEEK_API_KEY')
if not api_key:
    logger.error("DEEPSEEK_API_KEY not found in environment variables")
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"
)

# Configure app
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
    """Save flashcards to database and JSON file"""
    try:
        # 保存到数据库
        for card in flashcards:
            flashcard_data = {
                'source_file': original_filename,  # 直接使用原始文件名，不加时间戳
                'question': card['question'],
                'answer': card['answer'],
                'importance': card.get('importance', 3),
                'probability': card.get('probability', 3)
            }
            
            # 如果有学习状态，添加相关字段
            if 'learning_state' in card and card['learning_state']:
                flashcard_data.update({
                    'last_review': card['learning_state'].get('last_review'),
                    'next_review': card['learning_state'].get('next_review'),
                    'ease_factor': card['learning_state'].get('ease_factor', 2.5)
                })
            
            # 保存到数据库
            db.save_flashcard(flashcard_data)
        
        # 同时保存到JSON文件（保持向后兼容）
        base_name = os.path.splitext(original_filename)[0]
        filename = f"{base_name}_flashcards.json"  # 移除时间戳
        filepath = os.path.join(FLASHCARDS_FOLDER, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(flashcards, f, ensure_ascii=False, indent=2)
        
        return filename
    except Exception as e:
        logger.error(f"Error saving flashcards: {str(e)}")
        raise

class DateTimeEncoder(json.JSONEncoder):
    """处理datetime对象的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)

def generate_flashcards_stream(markdown_content, original_filename):
    """Generate flashcards with streaming progress"""
    try:
        yield json.dumps({
            'type': 'status',
            'data': '正在获取闪卡...'
        })

        # 获取基础文件名（去除扩展名和时间戳）
        base_filename = os.path.splitext(original_filename)[0]
        # 如果文件名包含时间戳（格式如 _YYYYMMDD_HHMMSS），去除它
        base_filename = re.sub(r'_\d{8}_\d{6}$', '', base_filename)

        # 直接从数据库获取闪卡，使用基础文件名进行模糊匹配
        flashcards = db.get_flashcards_by_source(base_filename)
        
        if not flashcards:
            # 如果数据库中没有找到闪卡，则生成新的闪卡
            yield json.dumps({
                'type': 'status',
                'data': '未找到现有闪卡，正在生成新的闪卡...'
            })
            
            # 这里保留原有的生成闪卡的代码
            # ... 原有的闪卡生成代码 ...
            
        else:
            # 获取所有相关的源文件
            source_files = set(card['source_file'] for card in flashcards)
            
            # 处理从数据库获取的闪卡
            for card in flashcards:
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
                
                # 转换datetime对象为字符串
                if 'created_at' in card:
                    card['created_at'] = card['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                if 'updated_at' in card:
                    card['updated_at'] = card['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
                if 'learning_state' in card:
                    if card['learning_state'].get('last_review'):
                        card['learning_state']['last_review'] = card['learning_state']['last_review'].strftime('%Y-%m-%d %H:%M:%S')
                    if card['learning_state'].get('next_review'):
                        card['learning_state']['next_review'] = card['learning_state']['next_review'].strftime('%Y-%m-%d %H:%M:%S')
            
            # 构建详细的消息
            message = f'成功加载 {len(flashcards)} 张闪卡'
            if len(source_files) > 1:
                message += f'（来自 {len(source_files)} 个相关文件：{", ".join(source_files)}）'
            
            yield json.dumps({
                'type': 'result',
                'data': flashcards,
                'filename': original_filename,
                'message': message
            }, cls=DateTimeEncoder)

    except Exception as e:
        yield json.dumps({
            'type': 'error',
            'data': str(e)
        })

@app.route('/')
def index():
    # 获取assets目录中的所有Markdown文件
    markdown_files = []
    for ext in ALLOWED_EXTENSIONS:
        markdown_files.extend(glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], f'*.{ext}')))
    markdown_files = [os.path.basename(f) for f in markdown_files]
    
    # 获取数据库中的文件
    db_files = db.get_available_files()
    
    return render_template('index.html', 
                         markdown_files=markdown_files,
                         db_files=db_files)

@app.route('/api/list_files', methods=['GET'])
def list_files():
    try:
        # 获取assets目录中的所有Markdown文件
        markdown_files = []
        for ext in ALLOWED_EXTENSIONS:
            markdown_files.extend(glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], f'*.{ext}')))
        markdown_files = [os.path.basename(f) for f in markdown_files]
        
        return jsonify({
            'success': True,
            'files': markdown_files
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/select_file', methods=['POST'])
def select_file():
    try:
        data = request.json
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        return Response(
            stream_with_context(generate_flashcards_stream(markdown_content, filename)),
            mimetype='text/event-stream'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/practice/<filename>')
def practice(filename):
    try:
        # 获取基础文件名（去除扩展名和时间戳）
        base_filename = os.path.splitext(filename)[0]
        # 如果文件名包含时间戳（格式如 _YYYYMMDD_HHMMSS），去除它
        base_filename = re.sub(r'_\d{8}_\d{6}$', '', base_filename)
        
        # 获取指定文件的闪卡
        flashcards = db.get_due_flashcards(source_file=base_filename)
        
        if not flashcards:
            return render_template('practice.html', flashcards=[], filename=filename, message='当前没有需要复习的卡片')
        
        # 处理从数据库读取的数据
        for card in flashcards:
            # 确保问题和答案是字符串并处理格式
            card['question'] = format_answer(str(card['question']).strip())
            card['answer'] = format_answer(str(card['answer']).strip())
        
        return render_template('practice.html', flashcards=flashcards, filename=filename)
    except Exception as e:
        logger.error(f"Error in practice route: {str(e)}")
        return render_template('error.html', message=str(e))

@app.route('/api/update_card_state', methods=['POST'])
def update_card_state():
    try:
        data = request.json
        card_id = data['card_id']
        review_data = data['review_data']
        
        # 使用SuperMemo算法计算下次复习时间
        current_interval = review_data.get('interval', 0)  # 如果是新卡片，interval为0
        ease_factor = review_data.get('ease_factor', 2.5)  # 默认难度系数为2.5
        difficulty = review_data.get('difficulty', 'good')  # 默认难度为good
        
        next_review_date, new_interval, new_ease_factor = get_next_review_date(
            difficulty=difficulty,
            current_interval=current_interval,
            ease_factor=ease_factor
        )
        
        # 使用当前时间作为复习时间
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        review_data.update({
            'review_date': current_time,
            'next_review': next_review_date.strftime('%Y-%m-%d %H:%M:%S'),
            'interval': new_interval,
            'ease_factor': new_ease_factor
        })
        
        # 保存复习记录到数据库（同时会更新热力图）
        db.save_review_record(card_id, review_data)
        
        # 清理重复的闪卡
        cleaned_count = db.cleanup_flashcards()
        logger.info(f"Cleaned up {cleaned_count} duplicate flashcards after review")
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error in update_card_state: {str(e)}")
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
            'Σ': '\\Sigma', 'Φ': '\\Phi', 'Ψ': '\\Psi', 'Ω': '\\Omega',
            # Unicode数学符号映射
            '𝜋': '\\pi', '𝑥': 'x', '𝑟': 'r', '𝑞': 'q', '𝑝': 'p',
            '′': "'", '⋅': '\\cdot', '∼': '\\sim'
        }
        
        # 替换Unicode希腊字母和数学符号为LaTeX命令
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
        formula = re.sub(r'([a-zA-Z0-9])\s*_\s*([a-zA-Z0-9\']+)', r'\1_{\2}', formula)
        formula = re.sub(r'([a-zA-Z0-9])\s*\^\s*([a-zA-Z0-9\']+)', r'\1^{\2}', formula)
        
        # 确保分数格式正确
        formula = re.sub(r'\\frac\s*{([^}]+)}\s*{([^}]+)}', r'\\frac{\1}{\2}', formula)
        
        # 处理多重反斜杠
        formula = formula.replace('\\\\', '\\')
        
        # 移除公式内多余的空格
        formula = re.sub(r'\s+', ' ', formula)
        formula = re.sub(r'(?<=\W)\s+|\s+(?=\W)', '', formula)
        
        return formula

    def merge_adjacent_formulas(text):
        """合并相邻的数学公式"""
        # 合并相邻的行内公式
        text = re.sub(r'\$\s*\$', '', text)  # 移除空的数学模式
        text = re.sub(r'\$\s*,\s*\$', ',', text)  # 合并被逗号分隔的公式
        text = re.sub(r'\$\s*([|,])\s*\$', r'\1', text)  # 合并被特殊字符分隔的公式
        text = re.sub(r'\$([^$]+?)\$\s*\$([^$]+?)\$', r'$\1\2$', text)  # 合并相邻的数学模式
        return text

    try:
        if not isinstance(text, str):
            return text

        # 预处理：移除多余的LaTeX命令
        text = re.sub(r'\\\\', r'\\', text)
        
        # 处理行内公式
        def replace_math_formula(match):
            formula = match.group(1)
            return f"${clean_formula(formula)}$"

        def replace_display_formula(match):
            formula = match.group(1)
            return f"$${clean_formula(formula)}$$"

        # 处理行内公式
        text = re.sub(r'\$([^\$]+?)\$', replace_math_formula, text)
        # 处理行间公式
        text = re.sub(r'\$\$([^\$]+?)\$\$', replace_display_formula, text)
        
        # 合并相邻的数学公式
        text = merge_adjacent_formulas(text)
        
        # 确保公式周围有适当的空格
        text = re.sub(r'([^\s])\$', r'\1 $', text)
        text = re.sub(r'\$([^\s])', r'$ \1', text)
        
        # 最终清理
        text = re.sub(r'\s+', ' ', text)  # 规范化空格
        text = text.replace(' ,', ',').replace(' .', '.')  # 修正标点符号
        
        return text
    except Exception as e:
        logger.error(f"Error in normalize_latex_formula: {str(e)}")
        return text  # 如果处理失败，返回原始文本

def format_answer(answer):
    """格式化答案文本，处理数学公式和格式化"""
    try:
        # 预处理文本
        text = answer.strip()
        if text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()
        text = text.lstrip('\ufeff')
        
        # 处理数学公式和格式化
        def clean_math_text(text):
            # 统一处理Unicode数学符号
            unicode_math = {
                '𝑥': 'x', '𝑟': 'r', '𝑞': 'q', '𝑝': 'p',
                '𝜋': '\\pi', '𝜃': '\\theta', '𝜇': '\\mu',
                '′': "'", '⋅': '\\cdot', '∼': '\\sim',
                '∆': '\\Delta', 'α': '\\alpha', 'β': '\\beta',
                'γ': '\\gamma', 'δ': '\\delta', 'ε': '\\epsilon',
                'ζ': '\\zeta', 'η': '\\eta', 'λ': '\\lambda',
                'σ': '\\sigma', 'τ': '\\tau', 'φ': '\\phi',
                'ω': '\\omega', 'Γ': '\\Gamma', 'Δ': '\\Delta',
                'Θ': '\\Theta', 'Λ': '\\Lambda', 'Ξ': '\\Xi',
                'Π': '\\Pi', 'Σ': '\\Sigma', 'Φ': '\\Phi',
                'Ψ': '\\Psi', 'Ω': '\\Omega'
            }
            for unicode_char, latex_char in unicode_math.items():
                text = text.replace(unicode_char, latex_char)
            
            # 处理数学公式
            text = re.sub(r'\\Delta\s+H', '\\Delta H', text)  # 修复Delta H的间距
            text = re.sub(r'\$\((.*?)\)\$', r'$\1$', text)  # 移除公式中多余的括号
            text = re.sub(r'\(\s*x\s*,\s*r\s*\)', '(x,r)', text)  # 标准化坐标对
            text = re.sub(r'\$\s*,\s*\$', ', ', text)  # 修复被错误分割的公式
            
            # 处理绝对值符号
            text = text.replace('|', '\\|')  # 将普通竖线替换为LaTeX绝对值符号
            
            # 确保公式中的空格正确
            text = re.sub(r'(?<=\w)(?=[\\$])', ' ', text)  # 在公式前添加空格
            text = re.sub(r'(?<=[\\$])(?=\w)', ' ', text)  # 在公式后添加空格
            
            # 处理多行公式
            text = re.sub(r'\n\s*(?=[\\$])', ' ', text)  # 合并跨行的公式
            text = re.sub(r'(?<=[\\$])\s*\n', ' ', text)  # 合并跨行的公式
            
            # 处理段落格式
            text = re.sub(r'\n{3,}', '\n\n', text)  # 减少多余的空行
            text = re.sub(r'(?<=。)\s*\n\s*(?=\S)', '\n\n', text)  # 在句子之间添加适当的空行
            
            # 处理特殊的LaTeX命令
            text = re.sub(r'\\begin{equation}', '$$', text)
            text = re.sub(r'\\end{equation}', '$$', text)
            text = re.sub(r'\\begin{align}', '$$', text)
            text = re.sub(r'\\end{align}', '$$', text)
            
            # 处理常见的数学符号
            text = re.sub(r'\\left\|', '\\|', text)
            text = re.sub(r'\\right\|', '\\|', text)
            text = re.sub(r'\\mathbb{R}', '\\mathbb{R}', text)
            text = re.sub(r'\\mathcal{N}', '\\mathcal{N}', text)
            
            return text.strip()
        
        # 应用清理和格式化
        formatted_text = clean_math_text(text)
        return formatted_text
        
    except Exception as e:
        logger.error(f"Error in format_answer: {str(e)}")
        return answer.strip()  # 如果处理失败，返回原始文本

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
                请确保回答准确、清晰，并保持与原文一致的专业水平，不要求生成json等格式化数据，而是自然语言。
                
                根据数学格式化规则：
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
                        "content": "你是一个专业的数学和机器学习教育专家，精通将复杂的数学概念转换为清晰的FlashCards。请确保保留所有LaTeX公式。"
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
            
            logger.debug("Received response from DeepSeek API")
            answer = response.choices[0].message.content.strip()
            
            # 格式化答案
            processed_answer = answer
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
        card_id = data['card_id']
        update_type = data['update_type']  # 'append' 或 'new'
        new_content = data['content']
        question = data.get('question', '')
        
        if update_type == 'append':
            # 获取当前卡片
            connection = db.connect()
            try:
                with connection.cursor() as cursor:
                    # 获取原卡片内容
                    sql = "SELECT * FROM flashcards WHERE id = %s"
                    cursor.execute(sql, (card_id,))
                    card = cursor.fetchone()
                    
                    if not card:
                        raise ValueError("Card not found")
                    
                    # 更新答案内容
                    new_answer = card['answer'] + '\n\n补充内容：\n' + new_content
                    sql = "UPDATE flashcards SET answer = %s WHERE id = %s"
                    cursor.execute(sql, (new_answer, card_id))
                    connection.commit()
                    
                    # 获取更新后的卡片
                    cursor.execute("SELECT * FROM flashcards WHERE id = %s", (card_id,))
                    updated_card = cursor.fetchone()
                    
                    return jsonify({
                        'success': True,
                        'message': '已补充到当前卡片',
                        'card': updated_card
                    })
            finally:
                connection.close()
        else:  # 'new'
            # 创建新卡片
            connection = db.connect()
            try:
                with connection.cursor() as cursor:
                    # 获取原卡片的一些属性作为参考
                    sql = "SELECT source_file, importance, probability FROM flashcards WHERE id = %s"
                    cursor.execute(sql, (card_id,))
                    ref_card = cursor.fetchone()
                    
                    if not ref_card:
                        raise ValueError("Reference card not found")
                    
                    # 创建新卡片
                    sql = """
                        INSERT INTO flashcards 
                        (source_file, question, answer, importance, probability)
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        ref_card['source_file'],
                        question,
                        new_content,
                        ref_card['importance'],
                        ref_card['probability']
                    ))
                    connection.commit()
                    
                    # 获取新创建的卡片
                    new_card_id = cursor.lastrowid
                    cursor.execute("SELECT * FROM flashcards WHERE id = %s", (new_card_id,))
                    new_card = cursor.fetchone()
                    
                    return jsonify({
                        'success': True,
                        'message': '已创建新卡片',
                        'card': new_card
                    })
            finally:
                connection.close()
            
    except Exception as e:
        logger.error(f"Error in update_card: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/heatmap', methods=['GET'])
def get_heatmap():
    try:
        logger.debug("Fetching heatmap data...")
        data = db.get_heatmap_data()
        logger.debug(f"Raw heatmap data: {data}")
        
        formatted_data = [{
            'date': row['date'].strftime('%Y-%m-%d'),
            'count': row['review_count']
        } for row in data]
        
        logger.debug(f"Formatted heatmap data: {formatted_data}")
        
        return jsonify({
            'success': True,
            'data': formatted_data
        })
    except Exception as e:
        logger.error(f"Error getting heatmap data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/test/review_records', methods=['GET'])
def test_review_records():
    """测试路由：检查数据库中的复习记录"""
    try:
        connection = db.connect()
        with connection.cursor() as cursor:
            # 检查复习记录表
            cursor.execute("SELECT COUNT(*) as count FROM review_records")
            review_count = cursor.fetchone()['count']
            
            # 检查最近的复习记录
            cursor.execute("""
                SELECT review_date, difficulty, ease_factor, review_interval
                FROM review_records
                ORDER BY review_date DESC
                LIMIT 5
            """)
            recent_reviews = cursor.fetchall()
            
            # 检查热力图数据
            cursor.execute("SELECT COUNT(*) as count FROM heatmap")
            heatmap_count = cursor.fetchone()['count']
            
            # 检查最近的热力图数据
            cursor.execute("""
                SELECT date, review_count
                FROM heatmap
                ORDER BY date DESC
                LIMIT 5
            """)
            recent_heatmap = cursor.fetchall()
            
            return jsonify({
                'success': True,
                'review_records_count': review_count,
                'recent_reviews': [{
                    'review_date': row['review_date'].strftime('%Y-%m-%d %H:%M:%S'),
                    'difficulty': row['difficulty'],
                    'ease_factor': float(row['ease_factor']),
                    'review_interval': row['review_interval']
                } for row in recent_reviews],
                'heatmap_records_count': heatmap_count,
                'recent_heatmap': [{
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'count': row['review_count']
                } for row in recent_heatmap]
            })
    except Exception as e:
        logger.error(f"Error in test route: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    finally:
        if connection:
            connection.close()

if __name__ == '__main__':
    app.run(debug=True) 