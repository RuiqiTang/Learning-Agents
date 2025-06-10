# Learning-Agents
Share my learning process through AI agents

<!-- 1. 热力图，我的爱
2. 从pdf/手写笔记/图片中自动制卡
3. 对于每个卡片，有一个revision版本，可以选择：
   （1） 补充进现有卡片
   （2） 变成新的卡片 -->

编写程序，实现：
1. 建造一个本地的可视化html网页，可以upload pdf或者多个图片，将图片存储到本地文件夹的assets中
2. 将上传的assets转变为markdown格式（注意我的笔记多是数学笔记，需要ocr to latex），存储在notes_sparse文件夹下，将中间过程存储在ocr_output下
---

0610修改
编写程序，实现：
1. 在上传文件界面，修改界面显示两个按钮，一个是“上传文件”，另一个是“从当前文件库选取”，如果选择后一个，则从assets中进行选取
2. 将每个文件对应的多张flashcards输出到数据库learning_agents中的flashcards表，要求记录来自文件（比如test_notes）、卡片正面、卡片反面、上次复习难度（简单、困难等）、下次到期时间等字段