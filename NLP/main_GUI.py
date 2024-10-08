# main_gui.py
import sys
import os
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFileDialog, QRadioButton, QButtonGroup, QMessageBox, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
# 导入已有的分词函数
import jieba
# from llama3_2_segment import llama_segment  # 预留，暂未实现
from segment_algorithm.parallel_segment_algorithm import parallel_segment, read_dictionary
from segment_algorithm.origin_segment_algorithm import calculate_metrics

from wordcloud import WordCloud
from io import BytesIO
from collections import Counter

class SegmenterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        # 预加载词典，避免重复加载
        self.dictionary, self.max_word_length = read_dictionary('Data/new_wordlist.Dic')
        # 用于存储 jieba 分词结果，供评估使用
        self.jieba_words = []
        # 用于存储当前的分词结果
        self.current_words = []

    def init_ui(self):
        self.setWindowTitle('中文分词和词云生成工具')

        # 文本输入区
        self.input_label = QLabel('输入文本：')
        self.text_edit = QTextEdit()

        # 打开文件按钮
        self.open_file_button = QPushButton('打开文件')
        self.open_file_button.clicked.connect(self.load_file)

        # 分词算法选择
        self.algorithm_label = QLabel('选择分词算法：')
        self.jieba_radio = QRadioButton('jieba分词')
        self.llama_radio = QRadioButton('llama3.2分词（预留）')
        self.custom_radio = QRadioButton('并行双向最大匹配分词')
        self.jieba_radio.setChecked(True)  # 默认选中 jieba

        self.algorithm_group = QButtonGroup()
        self.algorithm_group.addButton(self.jieba_radio)
        self.algorithm_group.addButton(self.llama_radio)
        self.algorithm_group.addButton(self.custom_radio)

        # 开始分词按钮
        self.segment_button = QPushButton('开始分词')
        self.segment_button.clicked.connect(self.start_segmentation)

        # 生成词云按钮
        self.wordcloud_button = QPushButton('生成词云')
        self.wordcloud_button.clicked.connect(self.generate_wordcloud)
        self.wordcloud_button.setEnabled(False)  # 初始状态下禁用，待分词后启用

        # 分词结果显示区
        self.result_label = QLabel('分词结果（部分）：')
        self.result_edit = QTextEdit()
        self.result_edit.setReadOnly(True)

        # 评估指标显示区
        self.metrics_label = QLabel('评估指标：')
        self.metrics_edit = QTextEdit()
        self.metrics_edit.setReadOnly(True)

        # 词云图显示区域
        self.wordcloud_label = QLabel('词云图：')
        self.wordcloud_image = QLabel()
        self.wordcloud_image.setAlignment(Qt.AlignCenter)
        self.wordcloud_image.setFixedHeight(400)  # 设置固定高度，可根据需要调整

        # 为词云图添加滚动条（如果词云图尺寸较大，可以滚动查看）
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.wordcloud_image)

        # 布局设置
        main_layout = QVBoxLayout()

        # 输入区布局
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.open_file_button)

        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.text_edit)

        # 算法选择布局
        algo_layout = QHBoxLayout()
        algo_layout.addWidget(self.algorithm_label)
        algo_layout.addWidget(self.jieba_radio)
        algo_layout.addWidget(self.llama_radio)
        algo_layout.addWidget(self.custom_radio)
        main_layout.addLayout(algo_layout)

        # 开始分词和生成词云按钮
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.segment_button)
        button_layout.addWidget(self.wordcloud_button)
        main_layout.addLayout(button_layout)

        # 分词结果显示
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(self.result_edit)

        # 评估指标显示
        main_layout.addWidget(self.metrics_label)
        main_layout.addWidget(self.metrics_edit)

        # 词云图显示
        main_layout.addWidget(self.wordcloud_label)
        main_layout.addWidget(self.scroll_area)

        self.setLayout(main_layout)

    def load_file(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, '打开文本文件', '', 'Text Files (*.txt);;All Files (*)', options=options
        )
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
                self.text_edit.setText(text)

    def start_segmentation(self):
        self.wordcloud_button.setEnabled(False)  # 分词前禁用生成词云按钮
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, '警告', '请输入文本或加载文本文件。')
            return

        # 根据选择的算法进行分词
        if self.jieba_radio.isChecked():
            start_time = time.time()
            words = list(jieba.cut(text, cut_all=False, HMM=True))
            end_time = time.time()
            self.jieba_words = words  # 保存 jieba 分词结果供评估使用
            # 显示分词结果和耗时
            self.display_results(words, end_time - start_time)
            self.metrics_edit.setText('')  # 清空评估指标

        elif self.llama_radio.isChecked():
            QMessageBox.information(self, '提示', 'llama3.2分词功能尚未实现。')
            # 预留代码，待实现后填入
            # start_time = time.time()
            # words = llama_segment(text)
            # end_time = time.time()
            # self.display_results(words, end_time - start_time)
            # self.metrics_edit.setText('')  # 清空评估指标

        elif self.custom_radio.isChecked():
            # 需要先有 jieba 分词结果作为标准
            if not self.jieba_words:
                # 先使用 jieba 分词
                self.jieba_words = list(jieba.cut(text, cut_all=False, HMM=True))

            start_time = time.time()
            # 使用并行双向最大匹配算法分词
            num_workers = 4  # 设置进程数
            words = self.custom_segmentation(text, num_workers)
            end_time = time.time()
            # 评估分词结果
            metrics = self.evaluate_segmentation(words, self.jieba_words)
            # 显示分词结果、耗时和评估指标
            self.display_results(words, end_time - start_time)
            self.display_metrics(metrics)

        else:
            QMessageBox.warning(self, '警告', '请选择一种分词算法。')
            return

        # 保存当前的分词结果
        self.current_words = words
        # 分词完成后启用“生成词云”按钮
        self.wordcloud_button.setEnabled(True)

    def custom_segmentation(self, text, num_workers):
        # 调用并行分词函数
        forward_words, backward_words, bidirectional_words = parallel_segment(
            text, self.dictionary, self.max_word_length, num_workers=num_workers
        )
        # 这里选择双向分词结果
        return bidirectional_words

    def evaluate_segmentation(self, segmented_words, standard_words):
        # 将标准结果（jieba分词结果）转换为 Counter
        standard_counter = Counter(standard_words)
        correct_count = 0
        for word in segmented_words:
            if word in standard_counter and standard_counter[word] > 0:
                correct_count += 1
                standard_counter[word] -= 1  # 减少对应词的剩余次数
        total_standard_words = len(standard_words)
        precision, recall, f1_score = calculate_metrics(
            correct_count, total_standard_words, segmented_words
        )
        metrics = {
            '正确词数': correct_count,
            '分词结果数': len(segmented_words),
            '准确率': f'{precision * 100:.2f}%',
            '召回率': f'{recall * 100:.2f}%',
            'F1值': f'{f1_score * 100:.2f}%'
        }
        return metrics

    def display_results(self, words, elapsed_time):
        # 显示部分分词结果（前100个词）
        result_text = ' '.join(words[:100])
        self.result_edit.setText(result_text)
        # 显示耗时
        self.result_label.setText(f'分词结果（部分，耗时 {elapsed_time:.2f} 秒）：')
        # 清空词云图显示区域
        self.wordcloud_image.clear()
        self.wordcloud_label.setText('词云图：')

    def display_metrics(self, metrics):
        # 显示评估指标
        metrics_text = (
            f"正确词数：{metrics['正确词数']}\n"
            f"分词结果数：{metrics['分词结果数']}\n"
            f"准确率：{metrics['准确率']}\n"
            f"召回率：{metrics['召回率']}\n"
            f"F1值：{metrics['F1值']}"
        )
        self.metrics_edit.setText(metrics_text)

    def generate_wordcloud(self):
        if not hasattr(self, 'current_words') or not self.current_words:
            QMessageBox.warning(self, '警告', '请先进行分词。')
            return

        # 将词列表拼接成字符串，空格分隔
        text = ' '.join(self.current_words)

        # 加载停用词（可选）
        stopwords = self.load_stopwords('word_cloud_generate/cn_stopwords.txt')

        # 设置字体路径（确保路径正确，Windows系统下通常是以下路径）
        font_path = 'C:/Windows/Fonts/simhei.ttf'
        if not os.path.exists(font_path):
            QMessageBox.warning(self, '警告', f'字体文件 {font_path} 不存在，请检查路径。')
            return

        # 生成词云
        try:
            wc = WordCloud(
                font_path=font_path,
                width=800,
                height=600,
                background_color='white',
                stopwords=stopwords,
                collocations=False  # 防止词云中词重复
            )
            wc.generate(text)

            # 将词云图像转换为 QPixmap 并在界面中显示
            image = wc.to_image()
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            qimage = QImage.fromData(buffer.getvalue())
            pixmap = QPixmap.fromImage(qimage)
            self.wordcloud_image.setPixmap(pixmap.scaled(
                self.wordcloud_image.width(),
                self.wordcloud_image.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.wordcloud_label.setText('词云图（根据当前分词结果生成）：')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'生成词云时发生错误：{str(e)}')

    def load_stopwords(self, filepath):
        stopwords = set()
        if not os.path.exists(filepath):
            QMessageBox.warning(self, '警告', f'停用词文件 {filepath} 不存在，将不使用停用词。')
            return stopwords

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
        return stopwords

def main():
    app = QApplication(sys.argv)
    gui = SegmenterGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
