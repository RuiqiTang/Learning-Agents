import pymysql
from pymysql.cursors import DictCursor
from config import DB_CONFIG, FLASHCARDS_FOLDER
import logging
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.config = DB_CONFIG.copy()
        # 确保使用字典游标
        self.config['cursorclass'] = DictCursor

    def connect(self):
        try:
            connection = pymysql.connect(**self.config)
            logger.info("Successfully connected to database")
            return connection
        except pymysql.Error as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise

    def init_db(self):
        """Initialize database and create tables if they don't exist"""
        connection = None
        try:
            # 首先尝试创建数据库（如果不存在）
            config_without_db = self.config.copy()
            db_name = config_without_db.pop('database')
            
            temp_conn = pymysql.connect(**config_without_db)
            with temp_conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            temp_conn.close()
            
            # 然后连接到数据库并创建表
            connection = self.connect()
            with connection.cursor() as cursor:
                # Create flashcards table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS flashcards (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        source_file VARCHAR(255) NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        importance INT NOT NULL DEFAULT 3,
                        probability INT NOT NULL DEFAULT 3,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_source_file (source_file)
                    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
                """)
                
                # Create review_records table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS review_records (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        flashcard_id INT NOT NULL,
                        review_date DATETIME NOT NULL,
                        next_review DATETIME,
                        difficulty VARCHAR(20),
                        ease_factor FLOAT NOT NULL DEFAULT 2.5,
                        review_interval INT NOT NULL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (flashcard_id) REFERENCES flashcards(id) ON DELETE CASCADE,
                        INDEX idx_flashcard_review (flashcard_id, review_date),
                        INDEX idx_next_review (next_review)
                    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
                """)
                
                connection.commit()
                logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()

    def save_flashcard(self, flashcard_data):
        """Save a flashcard to database"""
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                sql = """
                    INSERT INTO flashcards 
                    (source_file, question, answer, importance, probability, 
                     review_count, last_review, next_review, last_difficulty, ease_factor)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    flashcard_data['source_file'],
                    flashcard_data['question'],
                    flashcard_data['answer'],
                    flashcard_data['importance'],
                    flashcard_data['probability'],
                    flashcard_data.get('review_count', 0),
                    flashcard_data.get('last_review'),
                    flashcard_data.get('next_review'),
                    flashcard_data.get('last_difficulty'),
                    flashcard_data.get('ease_factor', 2.5)
                ))
                connection.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving flashcard: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()

    def get_flashcards_by_source(self, source_file):
        """Get all flashcards for a specific source file"""
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                sql = "SELECT * FROM flashcards WHERE source_file = %s ORDER BY created_at"
                cursor.execute(sql, (source_file,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting flashcards: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()

    def update_flashcard_review(self, flashcard_id, review_data):
        """Update flashcard review information"""
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                sql = """
                    UPDATE flashcards 
                    SET review_count = %s,
                        last_review = %s,
                        next_review = %s,
                        last_difficulty = %s,
                        ease_factor = %s
                    WHERE id = %s
                """
                cursor.execute(sql, (
                    review_data['review_count'],
                    review_data['last_review'],
                    review_data['next_review'],
                    review_data['last_difficulty'],
                    review_data['ease_factor'],
                    flashcard_id
                ))
                connection.commit()
        except Exception as e:
            logger.error(f"Error updating flashcard review: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()

    def get_available_files(self):
        """Get list of all unique source files"""
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                sql = "SELECT DISTINCT source_file FROM flashcards ORDER BY source_file"
                cursor.execute(sql)
                return [row['source_file'] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting available files: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()

    def import_from_json(self):
        """Import existing flashcards from JSON files"""
        connection = None
        try:
            connection = self.connect()
            
            # 获取所有JSON文件
            json_files = [f for f in os.listdir(FLASHCARDS_FOLDER) if f.endswith('_flashcards.json')]
            imported_count = 0
            
            for json_file in json_files:
                filepath = os.path.join(FLASHCARDS_FOLDER, json_file)
                source_file = json_file.replace('_flashcards.json', '.md')
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        flashcards = json.load(f)
                    
                    for card in flashcards:
                        # 首先保存闪卡基本信息
                        flashcard_data = {
                            'source_file': source_file,
                            'question': card['question'],
                            'answer': card['answer'],
                            'importance': card.get('importance', 3),
                            'probability': card.get('probability', 3)
                        }
                        
                        # 保存闪卡并获取ID
                        with connection.cursor() as cursor:
                            sql = """
                                INSERT INTO flashcards 
                                (source_file, question, answer, importance, probability)
                                VALUES (%s, %s, %s, %s, %s)
                            """
                            cursor.execute(sql, (
                                flashcard_data['source_file'],
                                flashcard_data['question'],
                                flashcard_data['answer'],
                                flashcard_data['importance'],
                                flashcard_data['probability']
                            ))
                            flashcard_id = cursor.lastrowid
                            
                            # 如果有学习状态，创建复习记录
                            if 'learning_state' in card and card['learning_state']:
                                state = card['learning_state']
                                if state.get('last_review'):
                                    review_data = {
                                        'review_date': state['last_review'],
                                        'next_review': state.get('next_review'),
                                        'difficulty': state.get('last_difficulty', 'normal'),
                                        'ease_factor': state.get('ease_factor', 2.5),
                                        'interval': state.get('interval', 0)
                                    }
                                    
                                    # 插入复习记录
                                    sql = """
                                        INSERT INTO review_records 
                                        (flashcard_id, review_date, next_review, difficulty, ease_factor, review_interval)
                                        VALUES (%s, %s, %s, %s, %s, %s)
                                    """
                                    cursor.execute(sql, (
                                        flashcard_id,
                                        review_data['review_date'],
                                        review_data['next_review'],
                                        review_data['difficulty'],
                                        review_data['ease_factor'],
                                        review_data['interval']
                                    ))
                        
                        connection.commit()
                        imported_count += 1
                    
                    logger.info(f"Successfully imported {len(flashcards)} cards from {source_file}")
                except Exception as e:
                    logger.error(f"Error importing file {json_file}: {str(e)}")
                    continue
            
            logger.info(f"Total imported flashcards: {imported_count}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error during import: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()

    def save_review_record(self, flashcard_id, review_data):
        """Save a review record"""
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                sql = """
                    INSERT INTO review_records 
                    (flashcard_id, review_date, next_review, difficulty, ease_factor, review_interval)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    flashcard_id,
                    review_data['review_date'],
                    review_data['next_review'],
                    review_data['difficulty'],
                    review_data['ease_factor'],
                    review_data['interval']
                ))
                connection.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving review record: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()

    def get_review_history(self, flashcard_id):
        """Get review history for a flashcard"""
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                sql = """
                    SELECT * FROM review_records 
                    WHERE flashcard_id = %s 
                    ORDER BY review_date DESC
                """
                cursor.execute(sql, (flashcard_id,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting review history: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()

    def get_next_reviews(self, limit=10):
        """Get cards due for review"""
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                sql = """
                    SELECT f.*, r.next_review, r.ease_factor
                    FROM flashcards f
                    LEFT JOIN review_records r ON f.id = r.flashcard_id
                    WHERE r.next_review <= NOW() OR r.next_review IS NULL
                    ORDER BY r.next_review ASC
                    LIMIT %s
                """
                cursor.execute(sql, (limit,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting next reviews: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()

    def flashcard_exists(self, source_file, question):
        """判断卡片是否已存在（source_file+question唯一）"""
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                sql = "SELECT id FROM flashcards WHERE source_file=%s AND question=%s"
                cursor.execute(sql, (source_file, question))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking flashcard existence: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()

    def get_due_flashcards(self, now=None):
        """获取到期需要复习的卡片"""
        if now is None:
            now = datetime.now()
            
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                sql = """
                    SELECT f.*, COALESCE(r.next_review, f.created_at) as next_review,
                           COALESCE(r.ease_factor, 2.5) as ease_factor,
                           COALESCE(r.review_interval, 0) as review_interval
                    FROM flashcards f
                    LEFT JOIN (
                        SELECT flashcard_id, next_review, ease_factor, review_interval
                        FROM review_records r1
                        WHERE review_date = (
                            SELECT MAX(review_date)
                            FROM review_records r2
                            WHERE r2.flashcard_id = r1.flashcard_id
                        )
                    ) r ON f.id = r.flashcard_id
                    WHERE COALESCE(r.next_review, f.created_at) <= %s
                    ORDER BY COALESCE(r.next_review, f.created_at) ASC
                """
                cursor.execute(sql, (now,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting due flashcards: {str(e)}")
            raise
        finally:
            if connection:
                connection.close() 