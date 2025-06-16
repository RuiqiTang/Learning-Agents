import pymysql
from pymysql.cursors import DictCursor
from config import DB_CONFIG, FLASHCARDS_FOLDER
import logging
import os
import json
from datetime import datetime, timedelta

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
                        question LONGTEXT NOT NULL,
                        answer LONGTEXT NOT NULL,
                        importance INT NOT NULL DEFAULT 3,
                        probability INT NOT NULL DEFAULT 3,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_source_file (source_file)
                    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
                """)
                
                # 修改现有表的字段类型
                cursor.execute("""
                    ALTER TABLE flashcards 
                    MODIFY question LONGTEXT NOT NULL,
                    MODIFY answer LONGTEXT NOT NULL
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
                
                # Create heatmap table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS heatmap (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        date DATE NOT NULL,
                        review_count INT NOT NULL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        UNIQUE INDEX idx_date (date)
                    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
                """)
                
                connection.commit()
                logger.info("Database tables initialized successfully")

                # 初始化热力图数据
                self.init_heatmap_data()
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()

    def init_heatmap_data(self):
        """从review_records表初始化热力图数据"""
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                # 获取所有复习记录的日期
                sql = """
                    SELECT DATE(review_date) as date, COUNT(*) as count
                    FROM review_records
                    GROUP BY DATE(review_date)
                """
                cursor.execute(sql)
                review_dates = cursor.fetchall()

                # 更新热力图数据
                for record in review_dates:
                    sql = """
                        INSERT INTO heatmap (date, review_count)
                        VALUES (%s, %s)
                        ON DUPLICATE KEY UPDATE
                        review_count = VALUES(review_count),
                        updated_at = CURRENT_TIMESTAMP
                    """
                    cursor.execute(sql, (record['date'], record['count']))
                
                connection.commit()
                logger.info(f"Initialized heatmap data with {len(review_dates)} days")
        except Exception as e:
            logger.error(f"Error initializing heatmap data: {str(e)}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                connection.close()

    def save_flashcard(self, flashcard_data):
        """保存闪卡到数据库"""
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                # 检查卡片是否存在
                exists, existing_card = self.flashcard_exists(
                    flashcard_data['source_file'], 
                    flashcard_data['question']
                )
                
                if not exists:
                    # 插入新卡片
                    sql = """
                        INSERT INTO flashcards 
                        (source_file, question, answer, importance, probability)
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        flashcard_data['source_file'],
                        flashcard_data['question'],
                        flashcard_data['answer'],
                        flashcard_data.get('importance', 3),
                        flashcard_data.get('probability', 3)
                    ))
                    flashcard_id = cursor.lastrowid
                else:
                    # 更新现有卡片
                    sql = """
                        UPDATE flashcards 
                        SET answer = %s,
                            importance = %s,
                            probability = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """
                    cursor.execute(sql, (
                        flashcard_data['answer'],
                        flashcard_data.get('importance', 3),
                        flashcard_data.get('probability', 3),
                        existing_card['id']
                    ))
                    flashcard_id = existing_card['id']
                
                # 如果有学习状态，创建复习记录
                if 'last_review' in flashcard_data and flashcard_data['last_review']:
                    sql = """
                        INSERT INTO review_records 
                        (flashcard_id, review_date, next_review, difficulty, ease_factor, review_interval)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        flashcard_id,
                        flashcard_data['last_review'],
                        flashcard_data.get('next_review'),
                        'normal',  # 默认难度
                        flashcard_data.get('ease_factor', 2.5),
                        0  # 初始间隔
                    ))
                
                connection.commit()
                return flashcard_id
                
        except Exception as e:
            logger.error(f"Error saving flashcard: {str(e)}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                connection.close()

    def get_flashcards_by_source(self, source_file):
        """获取指定源文件的所有闪卡，支持模糊匹配文件名，并按问题去重选择最新版本"""
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                # 首先获取所有匹配的卡片中每个问题的最新版本
                sql = """
                    WITH LatestCards AS (
                        SELECT question, MAX(updated_at) as latest_update
                        FROM flashcards 
                        WHERE source_file LIKE %s
                        GROUP BY question
                    )
                    SELECT f.id, f.question, f.answer, f.importance, f.probability,
                           f.created_at, f.updated_at, f.source_file
                    FROM flashcards f
                    INNER JOIN LatestCards lc 
                        ON f.question = lc.question 
                        AND f.updated_at = lc.latest_update
                    ORDER BY f.updated_at DESC, f.id
                """
                # 添加通配符进行模糊匹配
                search_pattern = f"{source_file}%"
                cursor.execute(sql, (search_pattern,))
                cards = cursor.fetchall()
                
                # 获取每个卡片的最新学习状态
                for card in cards:
                    sql = """
                        SELECT review_date, next_review, difficulty, 
                               ease_factor, review_interval
                        FROM review_records
                        WHERE flashcard_id = %s
                        ORDER BY review_date DESC
                        LIMIT 1
                    """
                    cursor.execute(sql, (card['id'],))
                    learning_state = cursor.fetchone()
                    
                    # 添加学习状态到卡片数据中
                    card['learning_state'] = {
                        'review_count': 0,  # 默认值
                        'last_review': learning_state['review_date'] if learning_state else None,
                        'next_review': learning_state['next_review'] if learning_state else None,
                        'ease_factor': learning_state['ease_factor'] if learning_state else 2.5,
                        'interval': learning_state['review_interval'] if learning_state else 0
                    }
                    
                return cards
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

    def cleanup_flashcards(self):
        """清理重复的闪卡，根据question和answer去重，保留最新版本"""
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                # 使用CTE找出每个(question, answer)组合的最新记录
                sql = """
                    WITH LatestCards AS (
                        SELECT id, question, answer,
                               ROW_NUMBER() OVER (
                                   PARTITION BY question, answer
                                   ORDER BY updated_at DESC
                               ) as rn
                        FROM flashcards
                    ),
                    DuplicateCards AS (
                        SELECT id
                        FROM LatestCards
                        WHERE rn > 1
                    )
                    DELETE FROM flashcards
                    WHERE id IN (SELECT id FROM DuplicateCards)
                """
                cursor.execute(sql)
                deleted_count = cursor.rowcount
                connection.commit()
                logger.info(f"Cleaned up {deleted_count} duplicate flashcards")
                return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up flashcards: {str(e)}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                connection.close()

    def save_review_record(self, flashcard_id, review_data):
        """Save a review record and increment heatmap count"""
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                # 保存复习记录
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
                
                # 获取复习记录的日期
                review_date = datetime.strptime(review_data['review_date'], '%Y-%m-%d %H:%M:%S').date()
                
                # 更新热力图数据 - 增加当天的计数
                sql = """
                    INSERT INTO heatmap (date, review_count)
                    VALUES (%s, 1)
                    ON DUPLICATE KEY UPDATE
                    review_count = review_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                """
                cursor.execute(sql, (review_date,))
                
                # 清理重复的闪卡
                self.cleanup_flashcards()
                
                connection.commit()
                logger.info(f"Saved review record and incremented heatmap count for date {review_date}")
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving review record: {str(e)}")
            if connection:
                connection.rollback()
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
                # 首先检查是否存在完全相同的卡片
                sql = """
                    SELECT id, answer, importance, probability 
                    FROM flashcards 
                    WHERE source_file=%s AND question=%s
                """
                cursor.execute(sql, (source_file, question))
                result = cursor.fetchone()
                
                if result is None:
                    return False, None
                
                return True, result
        except Exception as e:
            logger.error(f"Error checking flashcard existence: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()

    def get_due_flashcards(self, source_file=None, now=None):
        """获取到期需要复习的卡片
        Args:
            source_file: 源文件名前缀，用于筛选特定文件的卡片
            now: 当前时间，用于判断是否到期
        """
        if now is None:
            now = datetime.now()
            
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                # 准备查询参数和条件
                params = []
                source_file_condition = ""
                if source_file:
                    source_file_condition = "AND source_file LIKE %s"
                    params.append(f"{source_file}%")
                
                # 将当前时间添加到参数列表
                params.append(now)
                
                # 构建SQL查询
                sql = f"""
                    WITH LatestCards AS (
                        SELECT f.id, f.source_file, f.question, f.answer, f.updated_at,
                               ROW_NUMBER() OVER (
                                   PARTITION BY f.source_file, f.question
                                   ORDER BY f.updated_at DESC
                               ) as rn
                        FROM flashcards f
                        WHERE 1=1
                        {source_file_condition}
                    )
                    SELECT 
                        f.*,
                        r.next_review,
                        r.ease_factor,
                        r.review_interval,
                        r.review_date as last_review
                    FROM LatestCards f
                    LEFT JOIN (
                        SELECT r1.*
                        FROM review_records r1
                        INNER JOIN (
                            SELECT flashcard_id, MAX(review_date) as max_review_date
                            FROM review_records
                            GROUP BY flashcard_id
                        ) r2 ON r1.flashcard_id = r2.flashcard_id 
                        AND r1.review_date = r2.max_review_date
                    ) r ON f.id = r.flashcard_id
                    WHERE f.rn = 1
                    AND (
                        r.next_review IS NULL  -- 从未复习过的卡片
                        OR r.next_review <= %s  -- 已经到期的卡片
                    )
                    ORDER BY 
                        CASE 
                            WHEN r.next_review IS NULL THEN f.updated_at  -- 新卡片按更新时间排序
                            ELSE r.next_review  -- 复习卡片按到期时间排序
                        END ASC
                """
                
                # 执行查询
                cursor.execute(sql, params)
                cards = cursor.fetchall()
                
                logger.info(f"Found {len(cards)} cards due for review from source {source_file}")
                return cards
                
        except Exception as e:
            logger.error(f"Error getting due flashcards: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()

    def update_heatmap(self, date=None):
        """更新指定日期的复习记录数量"""
        if date is None:
            date = datetime.now().date()
        
        logger.debug(f"Updating heatmap for date: {date}")
        
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                # 获取指定日期的复习记录数量
                sql = """
                    SELECT COUNT(*) as count
                    FROM review_records
                    WHERE DATE(review_date) = %s
                """
                cursor.execute(sql, (date,))
                result = cursor.fetchone()
                review_count = result['count']
                logger.debug(f"Found {review_count} reviews for date {date}")
                
                # 更新或插入heatmap记录
                sql = """
                    INSERT INTO heatmap (date, review_count)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE
                    review_count = VALUES(review_count),
                    updated_at = CURRENT_TIMESTAMP
                """
                cursor.execute(sql, (date, review_count))
                connection.commit()
                logger.debug(f"Updated heatmap record for date {date} with count {review_count}")
                
                return review_count
        except Exception as e:
            logger.error(f"Error updating heatmap: {str(e)}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                connection.close()

    def get_heatmap_data(self, start_date=None, end_date=None):
        """获取指定日期范围的heatmap数据"""
        connection = None
        try:
            connection = self.connect()
            with connection.cursor() as cursor:
                # 首先更新今天的数据
                today_count = self.update_heatmap(datetime.now().date())
                logger.debug(f"Today's review count: {today_count}")
                
                # 获取所有热力图数据
                sql = """
                    SELECT date, review_count
                    FROM heatmap
                    ORDER BY date ASC
                """
                cursor.execute(sql)
                results = cursor.fetchall()
                logger.debug(f"Found {len(results)} days with review data")
                return results
        except Exception as e:
            logger.error(f"Error getting heatmap data: {str(e)}")
            raise
        finally:
            if connection:
                connection.close() 