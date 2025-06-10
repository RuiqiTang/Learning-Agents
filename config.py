import os
from dotenv import load_dotenv
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': 'learning_agents',
    'charset': 'utf8mb4',
    'cursorclass': 'DictCursor',  # 使用字典游标
    'autocommit': True,  # 自动提交
    'connect_timeout': 5  # 连接超时时间
}

# 记录数据库配置（不包含敏感信息）
logger.info(f"Database host: {DB_CONFIG['host']}")
logger.info(f"Database user: {DB_CONFIG['user']}")
logger.info(f"Database name: {DB_CONFIG['database']}")

# Folders configuration
UPLOAD_FOLDER = 'assets'
FLASHCARDS_FOLDER = 'flashcards'

# Ensure directories exist
for folder in [UPLOAD_FOLDER, FLASHCARDS_FOLDER]:
    os.makedirs(folder, exist_ok=True) 