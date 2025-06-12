# Learning-Agents

一个智能的闪卡学习系统，结合了间隔重复算法和 AI 辅助学习功能。

## 功能特点

### 1. 智能闪卡系统
- 支持从 Markdown 文件自动生成闪卡
- 使用 SuperMemo 算法进行间隔重复学习
- 自动追踪学习进度和复习时间
- 智能去重，保持卡片最新版本

### 2. AI 辅助学习
- 集成 DeepSeek AI 进行实时问答
- 支持对每张卡片进行深入提问
- 可以将 AI 回答补充到现有卡片或创建新卡片

### 3. 学习数据可视化
- GitHub 风格的学习热力图
- 清晰的学习进度展示
- 复习历史记录追踪

### 4. 数学公式支持
- 完整支持 LaTeX 数学公式
- 自动格式化和规范化公式
- 支持行内和行间公式

## 技术栈

- 后端：Python + Flask
- 数据库：MySQL
- 前端：HTML + CSS + JavaScript
- 数学渲染：MathJax
- Markdown 渲染：Marked.js
- AI 集成：DeepSeek API

## 项目结构

```
Learning-Agents/
├── algo/
│   └── supermemo.py      # SuperMemo 算法实现
├── static/
│   └── assets/           # 静态资源文件
├── templates/
│   ├── index.html        # 主页面
│   └── practice.html     # 复习页面
├── app.py               # Flask 应用主文件
├── database.py          # 数据库操作
└── config.py           # 配置文件
```

## 安装和配置

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/Learning-Agents.git
cd Learning-Agents
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置数据库：
- 创建 MySQL 数据库
- 修改 `config.py` 中的数据库配置

4. 配置 AI：
- 在 `.env` 文件中设置 `DEEPSEEK_API_KEY`

5. 运行应用：
```bash
python app.py
```

## 使用说明

### 1. 创建闪卡
- 将 Markdown 文件放入 `assets` 目录
- 在网页界面上传或选择文件
- 系统会自动生成闪卡

### 2. 复习卡片
- 系统根据 SuperMemo 算法安排复习计划
- 根据难度评价（Again/Hard/Good/Easy）调整间隔
- 可以随时向 AI 提问，加深理解

### 3. 查看进度
- 在主页查看学习热力图
- 跟踪每日学习情况
- 查看整体学习进度

## 数据库结构

### flashcards 表
- 存储闪卡基本信息
- 自动去重，保持最新版本
- 记录创建和更新时间

### review_records 表
- 记录复习历史
- 存储间隔和难度系数
- 追踪下次复习时间

### heatmap 表
- 记录每日学习数据
- 用于生成热力图
- 实时更新学习记录

## 开发说明

### SuperMemo 算法
- 基于 SM-2 算法实现
- 根据用户反馈动态调整间隔
- 自适应学习难度

### AI 集成
- 使用 DeepSeek API
- 支持上下文相关的问答
- 保持数学公式格式

## 注意事项

1. 数据库备份
- 定期备份数据库
- 保护学习进度数据

2. API 密钥安全
- 不要泄露 DeepSeek API 密钥
- 使用环境变量管理密钥

3. 数学公式
- 使用标准 LaTeX 语法
- 注意公式换行和对齐

## 许可证

MIT License