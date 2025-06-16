from datetime import datetime, timedelta

def calculate_next_interval(difficulty: str, current_interval: int, ease_factor: float) -> tuple[int, float]:
    """
    计算下一次复习的间隔和新的难度系数
    
    Args:
        difficulty: 当前复习的难度评价 ('again', 'hard', 'good', 'easy')
        current_interval: 当前的复习间隔（天）
        ease_factor: 当前的难度系数 (默认2.5)
    
    Returns:
        tuple: (新的间隔天数, 新的难度系数)
    """
    # 确保ease_factor不会太小
    ease_factor = max(1.3, ease_factor)
    
    if current_interval == 0:
        # 首次学习或重新学习
        if difficulty == 'again':
            return 1, max(1.3, ease_factor - 0.3)
        elif difficulty == 'hard':
            return 2, max(1.3, ease_factor - 0.15)
        elif difficulty == 'good':
            return 4, ease_factor
        else:  # easy
            return 7, ease_factor + 0.15
    else:
        # 复习已学习的卡片
        if difficulty == 'again':
            # 重置间隔
            return max(1, int(current_interval * 0.2)), max(1.3, ease_factor - 0.3)
        elif difficulty == 'hard':
            # 稍微增加间隔
            return max(2, int(current_interval * 1.2)), max(1.3, ease_factor - 0.15)
        elif difficulty == 'good':
            # 正常增加间隔
            return int(current_interval * ease_factor), ease_factor
        else:  # easy
            # 大幅增加间隔
            return int(current_interval * ease_factor * 1.3), min(3.0, ease_factor + 0.15)

def get_next_review_date(difficulty: str, current_interval: int, ease_factor: float) -> tuple[datetime, int, float]:
    """
    计算下一次复习的具体日期
    
    Args:
        difficulty: 复习的难度评价
        current_interval: 当前的复习间隔（天）
        ease_factor: 当前的难度系数
    
    Returns:
        tuple: (下次复习日期, 新的间隔天数, 新的难度系数)
    """
    new_interval, new_ease_factor = calculate_next_interval(difficulty, current_interval, ease_factor)
    if difficulty in ['again', 'hard']:
        next_review = datetime.now() + timedelta(minutes=5)
        # 这里 new_interval 仍然返回原算法的天数，但实际下次复习时间是5分钟后
    else:
        next_review = datetime.now() + timedelta(days=new_interval)
    return next_review, new_interval, new_ease_factor
