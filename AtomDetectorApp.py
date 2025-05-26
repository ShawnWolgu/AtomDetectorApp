import sys
import os
import numpy as np
import cv2
import pandas as pd
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QSlider, QFileDialog, QWidget, QStatusBar, 
                            QSpinBox, QLineEdit, QGroupBox, QDoubleSpinBox, QCheckBox)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

class AtomDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("原子探测器")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化变量
        self.original_image = None
        self.displayed_image = None
        self.displayed_image_backup = None
        self.mask = None
        self.last_point = None
        self.drawing_mode = False
        self.erasing_mode = False
        self.delete_center_mode = False
        self.min_atom_size = 3  # 最小原子尺寸
        self.max_atom_size = 7  # 最大原子尺寸
        self.centers = []
        self.last_point = None
        self.centers = []
        self.raw_image = None  # 存储最原始的图像
        self.show_points = True  # 默认显示坐标点

            # 晶格分析相关变量
        self.a_coordinates = None  # A坐标点列表
        self.b_coordinates = None  # B坐标点列表
        self.lattice_centers = None  # 计算出的晶格中心点
        self.polarization_vectors = None  # 计算出的极化矢量
        self.vector_scale = 1.0
        # 设置UI
        self.setup_ui()
    
    def setup_ui(self):
        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # 左侧图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.mousePressEvent = self.image_click
        self.image_label.mouseMoveEvent = self.image_move
        
        # 右侧控制面板
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # 创建一个滚动区域来容纳所有控制元素
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 加载图像按钮
        load_button = QPushButton("加载图像")
        load_button.clicked.connect(self.load_image)
        control_layout.addWidget(load_button)
        
        # 图像预处理组
        preprocess_group = QGroupBox("图像预处理")
        preprocess_layout = QVBoxLayout()
        
        self.equalize_brightness_btn = QPushButton("亮度均衡化")
        self.equalize_brightness_btn.clicked.connect(self.equalize_brightness)
        self.equalize_brightness_btn.setEnabled(False)
        preprocess_layout.addWidget(self.equalize_brightness_btn)
        
        self.rescale_intensity_btn = QPushButton("重新缩放亮度")
        self.rescale_intensity_btn.clicked.connect(self.rescale_intensity)
        self.rescale_intensity_btn.setEnabled(False)
        preprocess_layout.addWidget(self.rescale_intensity_btn)
        
        self.reset_image_btn = QPushButton("重置为原始图像")
        self.reset_image_btn.clicked.connect(self.reset_image)
        self.reset_image_btn.setEnabled(False)
        preprocess_layout.addWidget(self.reset_image_btn)
        
        preprocess_group.setLayout(preprocess_layout)
        control_layout.addWidget(preprocess_group)
        
        # 阈值控制组
        threshold_group = QGroupBox("阈值控制")
        threshold_layout = QVBoxLayout()
        
        # 最小阈值控制
        min_thresh_layout = QHBoxLayout()
        min_thresh_layout.addWidget(QLabel("最小阈值:"))
        self.min_thresh_slider = QSlider(Qt.Horizontal)
        self.min_thresh_slider.setRange(0, 255)
        self.min_thresh_slider.setValue(50)
        self.min_thresh_slider.valueChanged.connect(self.on_slider_changed)
        min_thresh_layout.addWidget(self.min_thresh_slider)
        
        # 添加最小阈值输入框
        self.min_thresh_input = QSpinBox()
        self.min_thresh_input.setRange(0, 255)
        self.min_thresh_input.setValue(50)
        self.min_thresh_input.valueChanged.connect(self.on_input_changed)
        min_thresh_layout.addWidget(self.min_thresh_input)
        
        threshold_layout.addLayout(min_thresh_layout)
        
        # 最大阈值控制
        max_thresh_layout = QHBoxLayout()
        max_thresh_layout.addWidget(QLabel("最大阈值:"))
        self.max_thresh_slider = QSlider(Qt.Horizontal)
        self.max_thresh_slider.setRange(0, 255)
        self.max_thresh_slider.setValue(200)
        self.max_thresh_slider.valueChanged.connect(self.on_slider_changed)
        max_thresh_layout.addWidget(self.max_thresh_slider)
        
        # 添加最大阈值输入框
        self.max_thresh_input = QSpinBox()
        self.max_thresh_input.setRange(0, 255)
        self.max_thresh_input.setValue(200)
        self.max_thresh_input.valueChanged.connect(self.on_input_changed)
        max_thresh_layout.addWidget(self.max_thresh_input)
        
        threshold_layout.addLayout(max_thresh_layout)
        
        # 添加刷新图像按钮
        refresh_button = QPushButton("刷新图像")
        refresh_button.clicked.connect(self.update_threshold)
        threshold_layout.addWidget(refresh_button)
        
        threshold_group.setLayout(threshold_layout)
        control_layout.addWidget(threshold_group)
        
        # 添加重置掩码按钮
        self.reset_mask_btn = QPushButton("重置掩码")
        self.reset_mask_btn.clicked.connect(self.reset_mask)
        self.reset_mask_btn.setEnabled(False)
        control_layout.addWidget(self.reset_mask_btn)
        
        # 原子尺寸范围控制
        atom_size_group = QGroupBox("原子尺寸范围")
        atom_size_layout = QVBoxLayout()
        
        # 最小原子尺寸控制
        min_size_layout = QHBoxLayout()
        min_size_layout.addWidget(QLabel("最小尺寸:"))
        self.min_size_input = QSpinBox()
        self.min_size_input.setRange(1, 200)
        self.min_size_input.setValue(3)  # 默认最小尺寸
        self.min_size_input.valueChanged.connect(self.update_atom_size_range)
        min_size_layout.addWidget(self.min_size_input)
        atom_size_layout.addLayout(min_size_layout)
        
        # 最大原子尺寸控制
        max_size_layout = QHBoxLayout()
        max_size_layout.addWidget(QLabel("最大尺寸:"))
        self.max_size_input = QSpinBox()
        self.max_size_input.setRange(1, 200)
        self.max_size_input.setValue(7)  # 默认最大尺寸
        self.max_size_input.valueChanged.connect(self.update_atom_size_range)
        max_size_layout.addWidget(self.max_size_input)
        atom_size_layout.addLayout(max_size_layout)
        
        # 添加应用尺寸范围按钮
        refresh_size_button = QPushButton("应用尺寸范围")
        refresh_size_button.clicked.connect(self.update_threshold)
        atom_size_layout.addWidget(refresh_size_button)
        
        atom_size_group.setLayout(atom_size_layout)
        control_layout.addWidget(atom_size_group)
        
        # 模式控制按钮组
        mode_group = QGroupBox("操作模式")
        mode_layout = QVBoxLayout()
        
        self.draw_button = QPushButton("添加蓝色点模式")
        self.draw_button.setCheckable(True)
        self.draw_button.clicked.connect(self.toggle_draw_mode)
        mode_layout.addWidget(self.draw_button)
        
        self.erase_button = QPushButton("删除蓝色区域模式")
        self.erase_button.setCheckable(True)
        self.erase_button.clicked.connect(self.toggle_erase_mode)
        mode_layout.addWidget(self.erase_button)
        
        self.delete_center_button = QPushButton("删除中心点模式")
        self.delete_center_button.setCheckable(True)
        self.delete_center_button.clicked.connect(self.toggle_delete_center_mode)
        mode_layout.addWidget(self.delete_center_button)
        
        mode_group.setLayout(mode_layout)
        control_layout.addWidget(mode_group)
        
        # 计算与保存按钮组
        calc_save_group = QGroupBox("计算与保存")
        calc_save_layout = QVBoxLayout()
        
        # 计算按钮
        calc_button = QPushButton("计算团簇中心点")
        calc_button.clicked.connect(self.calculate_centers)
        calc_save_layout.addWidget(calc_button)
        
        # 保存结果按钮
        save_button = QPushButton("保存结果到CSV")
        save_button.clicked.connect(self.save_results)
        calc_save_layout.addWidget(save_button)
        
        calc_save_group.setLayout(calc_save_layout)
        control_layout.addWidget(calc_save_group)
        
        # ============= 晶格分析模块 (放在右侧面板) ==============
        # 添加晶格分析功能的分组框
        lattice_group = QGroupBox("晶格分析")
        lattice_layout = QVBoxLayout()
        
        # 添加选择A坐标文件的按钮和显示路径的标签
        a_file_layout = QHBoxLayout()
        self.load_a_coords_btn = QPushButton("加载A坐标")
        self.load_a_coords_btn.clicked.connect(self.load_a_coordinates)
        self.a_coords_path_label = QLabel("未选择文件")
        self.a_coords_path_label.setWordWrap(True)  # 允许标签文本换行
        a_file_layout.addWidget(self.load_a_coords_btn)
        lattice_layout.addLayout(a_file_layout)
        lattice_layout.addWidget(self.a_coords_path_label)
        
        # 添加选择B坐标文件的按钮和显示路径的标签
        b_file_layout = QHBoxLayout()
        self.load_b_coords_btn = QPushButton("加载B坐标")
        self.load_b_coords_btn.clicked.connect(self.load_b_coordinates)
        self.b_coords_path_label = QLabel("未选择文件")
        self.b_coords_path_label.setWordWrap(True)  # 允许标签文本换行
        b_file_layout.addWidget(self.load_b_coords_btn)
        lattice_layout.addLayout(b_file_layout)
        lattice_layout.addWidget(self.b_coords_path_label)
        
        # 添加晶格中心计算按钮
        self.calculate_centers_btn = QPushButton("计算A晶格中心")
        self.calculate_centers_btn.clicked.connect(self.calculate_lattice_centers)
        lattice_layout.addWidget(self.calculate_centers_btn)
        
        # 添加极化矢量缩放系数控制
        vector_scale_layout = QHBoxLayout()
        vector_scale_layout.addWidget(QLabel("矢量放大系数:"))
        self.vector_scale_input = QDoubleSpinBox()
        self.vector_scale_input.setRange(0.1, 10.0)
        self.vector_scale_input.setValue(1.0)  # 默认系数为1.0
        self.vector_scale_input.setSingleStep(0.1)
        self.vector_scale_input.valueChanged.connect(self.update_vector_scale)
        vector_scale_layout.addWidget(self.vector_scale_input)
        lattice_layout.addLayout(vector_scale_layout)
        # 添加显示选项复选框
        self.show_points_checkbox = QCheckBox("显示坐标点和中心点")
        self.show_points_checkbox.setChecked(True)
        self.show_points_checkbox.stateChanged.connect(self.update_display_with_coordinates)
        lattice_layout.addWidget(self.show_points_checkbox)

        # 添加极化矢量计算按钮
        self.calculate_vectors_btn = QPushButton("计算极化矢量")
        self.calculate_vectors_btn.clicked.connect(self.calculate_polarization_vectors)
        lattice_layout.addWidget(self.calculate_vectors_btn)
        
        # 添加清除矢量按钮
        self.clear_vectors_btn = QPushButton("清除矢量")
        self.clear_vectors_btn.clicked.connect(self.clear_vectors)
        lattice_layout.addWidget(self.clear_vectors_btn)
        
        lattice_group.setLayout(lattice_layout)
        control_layout.addWidget(lattice_group)
        # =========================================
        
        # 设置控制面板在滚动区域内
        scroll_area.setWidget(control_panel)
        right_layout.addWidget(scroll_area)
        
        right_panel.setLayout(right_layout)
        right_panel.setFixedWidth(300)
        
        # 添加到主布局
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(right_panel)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")
        
    def on_slider_changed(self):
        # 滑块值改变时更新对应的输入框
        self.min_thresh_input.setValue(self.min_thresh_slider.value())
        self.max_thresh_input.setValue(self.max_thresh_slider.value())
    
    def on_input_changed(self):
        # 输入框值改变时更新对应的滑块
        self.min_thresh_slider.setValue(self.min_thresh_input.value())
        self.max_thresh_slider.setValue(self.max_thresh_input.value())
    
    def update_atom_size(self):
        self.atom_size = self.atom_size_input.value()
        self.statusBar.showMessage(f"原子尺寸已更新: {self.atom_size}")
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图像", "", 
                                                 "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)")
        if file_path:
            try:
                # 读取图像
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    self.statusBar.showMessage("无法读取图像")
                    return
                
                # 保存原始图像
                self.raw_image = img.copy()
                self.original_image = img.copy()
                
                # 初始化或重置掩码
                self.mask = None
                self.centers = []
                
                # 创建彩色显示图像
                self.displayed_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
                self.displayed_image_backup = None
                
                # 启用预处理按钮
                self.equalize_brightness_btn.setEnabled(True)
                self.rescale_intensity_btn.setEnabled(True)
                self.reset_image_btn.setEnabled(True)
                self.reset_mask_btn.setEnabled(True)
                
                # 更新界面
                self.update_image_display()
                self.statusBar.showMessage(f"已加载图像: {os.path.basename(file_path)}")
            except Exception as e:
                self.statusBar.showMessage(f"加载图像时出错: {str(e)}")

    def reset_image(self):
        if self.raw_image is None:
            return
        
        self.original_image = self.raw_image.copy()
        self.displayed_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        self.displayed_image_backup = None
        self.mask = None
        self.centers = []
        self.update_image_display()
        self.statusBar.showMessage("已重置为原始图像")

    def reset_mask(self):
        if self.original_image is None:
            return
        
        # 完全重置掩码
        self.mask = None
        # 重置中心点
        self.centers = []
        
        # 重新应用当前阈值生成新掩码
        self.update_threshold()
        
        self.statusBar.showMessage("掩码和中心点已重置")

    def update_atom_size_range(self):
        # 获取最小和最大尺寸值
        min_size = self.min_size_input.value()
        max_size = self.max_size_input.value()
        
        # 确保最小尺寸不大于最大尺寸
        if min_size > max_size:
            self.min_size_input.setValue(max_size)
            min_size = max_size
        
        self.min_atom_size = min_size
        self.max_atom_size = max_size
        self.statusBar.showMessage(f"原子尺寸范围已更新: {self.min_atom_size} - {self.max_atom_size}")
    
    def equalize_brightness(self):
        if self.original_image is None:
            return
        
        try:
            # 获取窗口大小 (原图像维度的1/10)
            window_size = max(3, int(min(self.original_image.shape) / 20))
            if window_size % 2 == 0:  # 确保是奇数
                window_size += 1
                
            self.statusBar.showMessage(f"正在执行亮度均衡化，窗口大小: {window_size}...")
            
            # 创建均匀化后的图像
            equalized = np.zeros_like(self.original_image, dtype=np.float32)
            
            # 计算整个图像的平均亮度
            global_mean = np.mean(self.original_image)
            
            # 使用均值滤波计算局部平均亮度
            local_mean = cv2.GaussianBlur(self.original_image.astype(np.float32), 
                                         (window_size, window_size), 0)
            
            # 对每个像素进行亮度校正: 像素值 * (全局平均亮度 / 局部平均亮度)
            # 避免除以零
            local_mean = np.maximum(local_mean, 1.0)
            equalized = self.original_image.astype(np.float32) * (global_mean / local_mean)
            
            # 裁剪到合理范围
            equalized = np.clip(equalized, 0, 255).astype(np.uint8)
            
            # 更新图像
            self.original_image = equalized
            self.displayed_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
            self.update_image_display()
            self.statusBar.showMessage("亮度均衡化完成")
            
        except Exception as e:
            self.statusBar.showMessage(f"亮度均衡化时出错: {str(e)}")
    # 添加亮度重新缩放方法

    def rescale_intensity(self):
        if self.original_image is None:
            return
        
        try:
            # 计算亮度值的直方图
            hist = cv2.calcHist([self.original_image], [0], None, [256], [0, 256])
            hist_cumsum = np.cumsum(hist) / np.sum(hist)
            
            # 定义截断百分比 (裁剪最低2%和最高2%的亮度值)
            low_percent, high_percent = 0.02, 0.98
            
            # 找到对应的亮度值
            low_value = np.searchsorted(hist_cumsum, low_percent)
            high_value = np.searchsorted(hist_cumsum, high_percent)
            
            # 确保有效的范围
            if low_value >= high_value:
                low_value = 0
                high_value = 255
                
            self.statusBar.showMessage(f"重新缩放亮度，范围: [{low_value}, {high_value}]...")
            
            # 线性缩放到0-255
            rescaled = np.clip(self.original_image, low_value, high_value)
            if high_value > low_value:
                rescaled = ((rescaled - low_value) / (high_value - low_value) * 255).astype(np.uint8)
            
            # 更新图像
            self.original_image = rescaled
            self.displayed_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
            self.update_image_display()
            self.statusBar.showMessage(f"亮度重新缩放完成，截断范围: [{low_value}, {high_value}]")
            
        except Exception as e:
            self.statusBar.showMessage(f"亮度重新缩放时出错: {str(e)}")
        
    def update_threshold(self):
        if self.original_image is None:
            return
        
        min_thresh = self.min_thresh_slider.value()
        max_thresh = self.max_thresh_slider.value()
        
        # 确保下限不大于上限
        if min_thresh > max_thresh:
            self.min_thresh_slider.setValue(max_thresh)
            self.min_thresh_input.setValue(max_thresh)
            min_thresh = max_thresh
        
        # 创建灰度图像的彩色版本
        display_img = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        
        # # 跟踪用户手动修改
        # user_modifications = None
        #
        # if self.mask is not None:
        #     # 保存所有用户手动修改（包括添加和删除）
        #     user_modifications = self.mask.copy()
        # 保存用户手动修改
        manual_additions = None
        manual_removals = None
    
        if self.mask is not None:
        # 找出用户手动添加的点(255)和手动删除的点(1)
            manual_additions = self.mask == 255
            manual_removals = self.mask == 1
        
        # 根据阈值完全重新生成掩码
        new_mask = cv2.inRange(self.original_image, min_thresh, max_thresh)

        # 在生成new_mask后，形态学操作前添加以下代码
        if new_mask is not None:
            # 获取图像尺寸
            h, w = new_mask.shape
            
            # 计算边缘5%的范围
            edge_h = int(h * 0.005)  # 高度边缘
            edge_w = int(w * 0.005)  # 宽度边缘
            
            # 清除上下左右边缘5%的区域
            new_mask[0:edge_h, :] = 0  # 上边缘
            new_mask[h-edge_h:h, :] = 0  # 下边缘
            new_mask[:, 0:edge_w] = 0  # 左边缘
            new_mask[:, w-edge_w:w] = 0  # 右边缘
        
        if manual_additions is not None:
            new_mask[manual_additions] = 0
        # 执行形态学操作，将零散的点连接成团簇
        avg_size = (self.min_atom_size + self.max_atom_size) // 2
        kernel_size = max(1, int(avg_size / 4))  # 根据平均原子尺寸确定核大小
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 先执行闭操作(膨胀后腐蚀)连接临近区域
        new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)
        
        # 去除小噪点
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, small_kernel)
            # 现在恢复手动添加和删除的部分
        if manual_additions is not None:
            new_mask[manual_additions] = 255  # 添加用户添加的点
        
        if manual_removals is not None:
            new_mask[manual_removals] = 0     # 删除用户删除的点
        
        # 对自动检测部分进行过滤，只保留符合原子尺寸范围的团簇
        filtered_mask = self.filter_clusters_by_size(new_mask, manual_additions, manual_removals)
    
        # 再次应用手动修改标记
        if manual_removals is not None:
            filtered_mask[manual_removals] = 1  # 保留删除标记
        
        self.mask = filtered_mask
        
        self.mask = filtered_mask
        
        # 创建彩色显示图像 - 直接用蓝色显示掩码区域
        display_img[self.mask == 255] = [255, 0, 0]  # BGR格式，蓝色
        
        # 如果有计算出的中心点，用红色显示
        if hasattr(self, 'centers') and self.centers:
            for center in self.centers:
                x, y = int(center[0]), int(center[1])
                cv2.circle(display_img, (x, y), 3, (0, 0, 255), -1)  # 红色圆点
        
        # 更新显示
        self.displayed_image = display_img
        self.update_image_display()
        
        self.statusBar.showMessage(f"阈值已更新: 下限={min_thresh}, 上限={max_thresh}")

    def filter_clusters_by_size(self, mask, manual_additions=None, manual_removals=None):
        """过滤团簇，只保留大小在指定尺寸范围内的团簇"""
        # 创建临时掩码用于处理
        temp_mask = mask.copy()
        
        # 如果有手动添加的区域，先将其从掩码中移除，以免受到尺寸过滤的影响
        if manual_additions is not None:
            temp_mask[manual_additions] = 0
        
        # 创建二值图像
        binary = (temp_mask == 255).astype(np.uint8)
        
        # 标记连通区域
        num_labels, labels = cv2.connectedComponents(binary)
        
        # 计算期望的团簇面积范围
        min_target_size = np.pi * (self.min_atom_size ** 2)  # 最小原子尺寸对应的圆形面积
        max_target_size = np.pi * (self.max_atom_size ** 2)  # 最大原子尺寸对应的圆形面积
        
        # 允许的偏差范围
        tolerance = 0.2  # 20%的偏差
        min_size = min_target_size * (1 - tolerance)  # 下限
        max_size = max_target_size * (1 + tolerance)  # 上限
        
        # 创建新掩码，初始全为0
        filtered_mask = np.zeros_like(mask)
        
        # 过滤团簇
        for label in range(1, num_labels):  # 从1开始，跳过背景(0)
            # 计算当前团簇的面积(像素数量)
            cluster_size = np.sum(labels == label)
            
            # 如果团簇大小在允许范围内，保留该团簇
            if min_size <= cluster_size <= max_size:
                filtered_mask[labels == label] = 255
        
        # 恢复手动添加的区域
        if manual_additions is not None:
            filtered_mask[manual_additions] = 255
        
        # 恢复手动删除的标记
        if manual_removals is not None:
            filtered_mask[manual_removals] = 1
        
        return filtered_mask

    def update_image_display(self,otherimg = None):
        if otherimg is None and self.displayed_image_backup is not None:
            self.displayed_image = self.displayed_image_backup

        if self.displayed_image is None and otherimg is None:
            return

        if otherimg is not None:
            self.displayed_image_backup = self.displayed_image.copy()
            self.displayed_image = otherimg
        
        h, w = self.displayed_image.shape[:2]
        bytes_per_line = 3 * w
        
        # 将OpenCV图像转换为Qt图像
        qt_image = QImage(self.displayed_image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # 设置Qt图像到标签
        pixmap = QPixmap.fromImage(qt_image)
        
        # 根据标签大小缩放图像
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
    
    def toggle_draw_mode(self):
        self.drawing_mode = self.draw_button.isChecked()
        if self.drawing_mode:
            if self.erase_button.isChecked():
                self.erase_button.setChecked(False)
                self.erasing_mode = False
            if self.delete_center_button.isChecked():
                self.delete_center_button.setChecked(False)
                self.delete_center_mode = False
        
        if self.drawing_mode:
            self.statusBar.showMessage("添加模式：点击图像添加蓝色点")
        else:
            self.statusBar.showMessage("普通模式")
    
    def toggle_erase_mode(self):
        self.erasing_mode = self.erase_button.isChecked()
        if self.erasing_mode:
            if self.draw_button.isChecked():
                self.draw_button.setChecked(False)
                self.drawing_mode = False
            if self.delete_center_button.isChecked():
                self.delete_center_button.setChecked(False)
                self.delete_center_mode = False
        
        if self.erasing_mode:
            self.statusBar.showMessage("删除模式：点击图像删除蓝色区域")
        else:
            self.statusBar.showMessage("普通模式")

    def toggle_delete_center_mode(self):
        self.delete_center_mode = self.delete_center_button.isChecked()
        
        # 如果启用了删除中心点模式，关闭其他模式
        if self.delete_center_mode:
            if self.draw_button.isChecked():
                self.draw_button.setChecked(False)
                self.drawing_mode = False
            if self.erase_button.isChecked():
                self.erase_button.setChecked(False)
                self.erasing_mode = False
            self.statusBar.showMessage("删除中心点模式：点击图像删除最近的中心点")
        else:
            self.statusBar.showMessage("普通模式")
    
    def get_image_coordinates(self, event):
        if self.displayed_image is None:
            return None
        
        # 获取图像在标签中的实际位置和大小
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return None
        
        # 计算图像位置和缩放比例
        w_scale = pixmap.width() / self.displayed_image.shape[1]
        h_scale = pixmap.height() / self.displayed_image.shape[0]
        scale = min(w_scale, h_scale)
        
        # 计算图像的偏移量（居中显示）
        x_offset = (self.image_label.width() - pixmap.width()) / 2
        y_offset = (self.image_label.height() - pixmap.height()) / 2
        
        # 计算鼠标在原始图像中的坐标
        x = int((event.pos().x() - x_offset) / scale)
        y = int((event.pos().y() - y_offset) / scale)
        
        # 确保坐标在图像范围内
        if (0 <= x < self.displayed_image.shape[1] and 
            0 <= y < self.displayed_image.shape[0]):
            return (x, y)
        
        return None
    
    def image_click(self, event):
        if self.mask is None:
            return
        
        coords = self.get_image_coordinates(event)
        if coords is None:
            return
        
        x, y = coords
        
        if self.drawing_mode:
            # 使用平均尺寸添加蓝色点
            avg_size = (self.min_atom_size + self.max_atom_size) // 2
            cv2.circle(self.mask, (x, y), avg_size, 255, -1)
            self.last_point = QPoint(x, y)
            self.update_threshold()
            self.statusBar.showMessage(f"已添加点 ({x}, {y})")
        
        elif self.erasing_mode:
            # 删除蓝色连通区域
            # 创建临时掩码用于寻找连通区域
            temp_mask = np.zeros_like(self.mask)
            temp_mask[self.mask == 255] = 255  # 只复制值为255的部分（蓝色区域）
            
            # 标记连通区域
            num_labels, labels = cv2.connectedComponents(temp_mask)
            
            # 找到点击位置的标签
            if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                label_at_point = labels[y, x]
                
                if label_at_point > 0:  # 0是背景
                    # 标记该连通区域为用户手动删除(1)
                    # 这样在update_threshold中会保留这个删除操作
                    self.mask[labels == label_at_point] = 1
                    
                    # 更新显示
                    self.update_threshold()
                    self.statusBar.showMessage(f"已删除区域，标签 {label_at_point}")
        elif self.delete_center_mode and hasattr(self, 'centers') and self.centers:
            # 找到距离点击位置最近的中心点
            click_point = np.array([x, y])
            min_dist = float('inf')
            min_idx = -1
            
            for i, center in enumerate(self.centers):
                center_point = np.array([center[0], center[1]])
                dist = np.linalg.norm(click_point - center_point)
                
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            
            # 设置一个最大距离阈值，防止误点击
            max_click_distance = 20  # 像素
            
            if min_idx >= 0 and min_dist <= max_click_distance:
                # 从列表中删除该中心点
                deleted_center = self.centers.pop(min_idx)
                
                # 更新显示
                self.update_threshold()
                
                self.statusBar.showMessage(
                    f"已删除中心点 ({int(deleted_center[0])}, {int(deleted_center[1])})，"
                    f"还剩 {len(self.centers)} 个中心点"
                )
            else:
                self.statusBar.showMessage(f"附近没有中心点可删除（最近距离: {min_dist:.1f}像素）")
    
    def image_move(self, event):
        if self.mask is None or not self.drawing_mode or self.last_point is None:
            return
        
        coords = self.get_image_coordinates(event)
        if coords is None:
            return
        
        x, y = coords
        
        # 使用平均尺寸绘制线条
        avg_size = (self.min_atom_size + self.max_atom_size) // 2
        cv2.line(self.mask, 
                 (self.last_point.x(), self.last_point.y()), 
                 (x, y), 
                 255, 
                 2 * avg_size)
        
        self.last_point = QPoint(x, y)
        self.update_threshold()
        
    
    def calculate_centers(self):
        if self.mask is None:
            self.statusBar.showMessage("没有可用的掩码")
            return
        
        try:
            # 创建二值图像
            binary = (self.mask == 255).astype(np.uint8)
            
            # 标记连通区域
            labeled, num_features = ndimage.label(binary)
            
            if num_features == 0:
                self.statusBar.showMessage("未找到任何团簇")
                return
            
            # 计算每个团簇的中心
            self.centers = []
            for i in range(1, num_features + 1):
                y, x = np.where(labeled == i)
                if len(x) > 0 and len(y) > 0:
                    center_x = np.mean(x)
                    center_y = np.mean(y)
                    self.centers.append((center_x, center_y))
            
            # 更新显示以添加中心点
            self.update_threshold()
            
            self.statusBar.showMessage(f"找到 {len(self.centers)} 个团簇中心")
        except Exception as e:
            self.statusBar.showMessage(f"计算中心点时出错: {str(e)}")
    
    def save_results(self):
        if not self.centers:
            self.statusBar.showMessage("没有可用的中心点")
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(self, "保存结果", "", "CSV文件 (*.csv)")
            
            if file_path:
                # 创建DataFrame
                data = {
                    'ID': range(1, len(self.centers) + 1),
                    'X': [center[0] for center in self.centers],
                    'Y': [center[1] for center in self.centers]
                }
                
                df = pd.DataFrame(data)
                
                # 保存到CSV
                df.to_csv(file_path, index=False)
                
                self.statusBar.showMessage(f"已将 {len(self.centers)} 个中心点保存到 {os.path.basename(file_path)}")
        except Exception as e:
            self.statusBar.showMessage(f"保存结果时出错: {str(e)}")
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.displayed_image is not None:
            self.update_image_display()

    def load_a_coordinates(self):
        """加载A坐标CSV文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择A坐标CSV文件", "", "CSV文件 (*.csv)")
        if not file_path:
            return
        
        try:
            # 读取CSV文件
            data = pd.read_csv(file_path)
            
            # 确保CSV文件至少有三列
            if data.shape[1] < 3:
                self.statusBar.showMessage("格式错误, CSV文件应至少包含三列，第二列和第三列为坐标。")
                return
            
            # 提取第二列和第三列作为坐标
            self.a_coordinates = data.iloc[:, 1:3].values
            
            # 更新标签显示文件路径
            self.a_coords_path_label.setText(os.path.basename(file_path))
            
            # 更新显示
            self.update_display_with_coordinates()
            
            self.statusBar.showMessage(f"已加载{len(self.a_coordinates)}个A坐标")
        except Exception as e:
            self.statusBar.showMessage(f"错误, 加载A坐标文件时出错: {str(e)}")

    def load_b_coordinates(self):
        """加载B坐标CSV文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择B坐标CSV文件", "", "CSV文件 (*.csv)")
        if not file_path:
            return
        
        try:
            # 读取CSV文件
            data = pd.read_csv(file_path)
            
            # 确保CSV文件至少有三列
            if data.shape[1] < 3:
                self.statusBar.showMessage("格式错误, CSV文件应至少包含三列，第二列和第三列为坐标。")
                return
            
            # 提取第二列和第三列作为坐标
            self.b_coordinates = data.iloc[:, 1:3].values
            
            # 更新标签显示文件路径
            self.b_coords_path_label.setText(os.path.basename(file_path))
            
            # 更新显示
            self.update_display_with_coordinates()
            
            self.statusBar.showMessage(f"已加载{len(self.b_coordinates)}个B坐标")
        except Exception as e:
            self.statusBar.showMessage(f"错误, 加载B坐标文件时出错: {str(e)}")

    def update_display_with_coordinates(self):
        """更新显示，包含A和B坐标点以及其他元素"""
        if self.displayed_image_backup is not None:
            self.displayed_image = self.displayed_image_backup

        if self.displayed_image is None:
            return
        
        # 创建显示图像的副本，以免修改原图
        display_img = self.displayed_image.copy()
        
        show_points = self.show_points_checkbox.isChecked() or self.polarization_vectors is None
        # 绘制A坐标点（蓝色）
        if self.a_coordinates is not None and show_points:
            for point in self.a_coordinates:
                x, y = int(point[0]), int(point[1])
                cv2.circle(display_img, (x, y), 3, (255, 0, 0), -1)  # 蓝色
        
        # 绘制B坐标点（红色）
        if self.b_coordinates is not None and show_points:
            for point in self.b_coordinates:
                x, y = int(point[0]), int(point[1])
                cv2.circle(display_img, (x, y), 3, (0, 0, 255), -1)  # 红色
        
        # 绘制晶格中心点（黄色）
        if self.lattice_centers is not None and show_points:
            for point in self.lattice_centers:
                x, y = int(point[0]), int(point[1])
                cv2.circle(display_img, (x, y), 4, (0, 255, 255), -1)  # 黄色
        
        # 绘制极化矢量（绿色箭头）
        if self.polarization_vectors is not None:
            for start, end in self.polarization_vectors:
                start_x, start_y = int(start[0]), int(start[1])
                
                # 计算原始矢量
                original_vector = (end[0] - start[0], end[1] - start[1])
                
                # 应用放大系数
                scaled_end_x = int(start[0] + original_vector[0] * self.vector_scale)
                scaled_end_y = int(start[1] + original_vector[1] * self.vector_scale)
                
                cv2.arrowedLine(display_img, (start_x, start_y), (scaled_end_x, scaled_end_y), 
                            (0, 255, 0), 2, tipLength=0.3)  # 绿色箭头
        # 更新显示
        self.update_image_display(display_img)

    def update_vector_scale(self):
        """更新矢量放大系数并刷新显示"""
        self.vector_scale = self.vector_scale_input.value()
        self.update_display_with_coordinates()

    def calculate_lattice_centers(self):
        """计算A晶格中心，基于角度和距离关系识别四个角点"""
        if self.a_coordinates is None or len(self.a_coordinates) < 15:
            self.statusBar.showMessage("数据不足: 需要至少15个A坐标点来计算晶格中心")
            return
        
        try:
            # 构建KD树用于快速近邻搜索
            tree = cKDTree(self.a_coordinates)
            
            # 用于存储计算出的晶格中心
            centers = []
            
            # 遍历每个A点，假设其为晶格的左上角
            for point in self.a_coordinates:
                # 查询距离当前点最近的15个点（包括自身）
                distances, indices = tree.query(point, k=10)  # 多查一个以防自身也包含在内
                
                # 去掉自身（如果存在），保留15个近邻点
                if distances[0] < 1e-10:  # 如果第一个是自身
                    neighbors_indices = indices[1:10]  # 跳过第一个
                    neighbors_distances = distances[1:10]
                else:
                    neighbors_indices = indices[:9]  # 取前15个
                    neighbors_distances = distances[:9]
                
                neighbors = [self.a_coordinates[i] for i in neighbors_indices]
                
                # 计算从当前点A到每个近邻点的矢量
                vectors = [neighbor - point for neighbor in neighbors]
                
                # 计算每个矢量的角度（相对于x轴正方向，范围为[-180, 180]度）
                angles = [np.degrees(np.arctan2(vec[1], vec[0])) for vec in vectors]
                
                # 计算每个矢量的长度
                lengths = [np.sqrt(vec[0]**2 + vec[1]**2) for vec in vectors]
                
                # 保存已使用的索引，确保一个点不被重复用作不同角点
                used_indices = set()
                
                # 1. 寻找右上角的点：矢量角度在-45~45度范围内，距离最近的点
                right_top_candidates = []
                for i, angle in enumerate(angles):
                    if -35 <= angle <= 35:
                        right_top_candidates.append((i, angle, lengths[i]))
                
                if not right_top_candidates:
                    continue  # 找不到符合条件的右上角点，跳过这个点
                    
                # 选择距离最近的作为右上角点
                right_top_candidates.sort(key=lambda x: x[2])
                right_top_idx, right_top_angle, right_top_length = right_top_candidates[0]
                right_top_point = neighbors[right_top_idx]
                
                # 将右上角点标记为已使用
                used_indices.add(right_top_idx)
                
                # 参考矢量为从A指向右上角点的矢量
                reference_vector = vectors[right_top_idx]
                reference_angle = right_top_angle
                reference_length = right_top_length
                
                # 2. 寻找左下角的点：参考矢量顺时针旋转90度的方向
                target_angle = reference_angle + 90
                if target_angle > 180:
                    target_angle -= 360  # 确保在[-180, 180]范围内
                    
                left_bottom_candidates = []
                min_angle_diff = float('inf')
                
                for i, angle in enumerate(angles):
                    # 跳过已经使用的点
                    if i in used_indices:
                        continue
                    
                    angle_diff = abs(angle - target_angle)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff  # 处理跨越±180边界的情况
                    
                    length_ratio = lengths[i] / reference_length

                    if angle_diff < 15 and 0.8 < length_ratio < 1.5:
                        score = angle_diff + abs(length_ratio - 1.414)   # 加权得分，长度误差权重更高
                        left_bottom_candidates.append((i, score))
                
                if not left_bottom_candidates:
                    continue  # 找不到符合条件的右下角点，跳过这个点
                    
                # 选择得分最低的作为右下角点
                left_bottom_candidates.sort(key=lambda x: x[1])
                left_bottom_idx = left_bottom_candidates[0][0]
                left_bottom_point = neighbors[left_bottom_idx]
                # 将左下角点标记为已使用
                used_indices.add(left_bottom_idx)
                
                # 3. 寻找右下角的点：参考矢量顺时针旋转45度（减去45度）的方向，长度约为参考矢量的1.7倍
                target_angle = reference_angle + 45
                if target_angle > 180:
                    target_angle -= 360  # 确保在[-180, 180]范围内
                    
                right_bottom_candidates = []
                for i, angle in enumerate(angles):
                    # 跳过已经使用的点
                    if i in used_indices:
                        continue
                    
                    angle_diff = abs(angle - target_angle)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff  # 处理跨越±180边界的情况
                    
                    length_ratio = lengths[i] / reference_length
                    
                    # 判断角度和长度是否在合理范围内
                    if angle_diff < 15 and 1.0 < length_ratio < 2.0:  # 允许一定范围的误差
                        score = angle_diff + abs(length_ratio - 1.414)   # 加权得分，长度误差权重更高
                        right_bottom_candidates.append((i, score))
                
                if not right_bottom_candidates:
                    continue  # 找不到符合条件的右下角点，跳过这个点
                    
                # 选择得分最低的作为右下角点
                right_bottom_candidates.sort(key=lambda x: x[1])
                right_bottom_idx = right_bottom_candidates[0][0]
                right_bottom_point = neighbors[right_bottom_idx]
                
                # 4. 计算晶格中心坐标（四个点的平均值）
                center_x = (point[0] + right_top_point[0] + left_bottom_point[0] + right_bottom_point[0]) / 4
                center_y = (point[1] + right_top_point[1] + left_bottom_point[1] + right_bottom_point[1]) / 4
                
                centers.append([center_x, center_y])
            
            # 去除重复的中心点
            if centers:
                centers_array = np.array(centers)
                # 使用DBSCAN聚类合并接近的中心点
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=5, min_samples=1).fit(centers_array)
                
                # 计算每个聚类的中心
                unique_labels = set(clustering.labels_)
                unique_centers = []
                
                for label in unique_labels:
                    mask = clustering.labels_ == label
                    cluster_points = centers_array[mask]
                    cluster_center = np.mean(cluster_points, axis=0)
                    unique_centers.append(cluster_center)
                
                self.lattice_centers = np.array(unique_centers)
                self.statusBar.showMessage(f"已计算{len(self.lattice_centers)}个晶格中心")
            else:
                self.lattice_centers = np.array([])
                self.statusBar.showMessage("没有找到符合条件的晶格中心")
            
            # 更新显示
            self.update_display_with_coordinates()
            
        except Exception as e:
            self.statusBar.showMessage(f"计算晶格中心时出错: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印详细的错误堆栈信息

    def calculate_polarization_vectors(self):
        """计算从晶格中心到最近B点的极化矢量"""
        if self.lattice_centers is None or len(self.lattice_centers) == 0:
            self.statusBar.showMessage("数据不足，请先计算A晶格中心")
            return
        
        if self.b_coordinates is None or len(self.b_coordinates) == 0:
            self.statusBar.showMessage("数据不足, 需要B坐标点来计算极化矢量。")
            return
        
        try:
            # 构建B坐标的KD树用于快速近邻搜索
            b_tree = cKDTree(self.b_coordinates)
            
            # 存储极化矢量（起点和终点）
            vectors = []
            vector_lengths = []
            
            # 遍历每个晶格中心
            for center in self.lattice_centers:
                # 查询距离当前中心最近的B点
                distance, index = b_tree.query(center, k=1)
                nearest_b = self.b_coordinates[index]
                
                # 添加从中心到B点的矢量
                vectors.append((center, nearest_b))
                
                # 计算矢量长度
                length = np.sqrt((nearest_b[0] - center[0])**2 + (nearest_b[1] - center[1])**2)
                vector_lengths.append(length)
            
            # 过滤异常长度的矢量
            if len(vector_lengths) > 0:
                # 计算矢量长度的平均值和标准差
                mean_length = np.mean(vector_lengths)
                std_length = np.std(vector_lengths)
                
                # 设定阈值：平均值±3个标准差
                threshold = mean_length + 3 * std_length
                
                # 过滤掉超过阈值的矢量
                filtered_vectors = []
                for i, (vec, length) in enumerate(zip(vectors, vector_lengths)):
                    if length <= threshold:
                        filtered_vectors.append(vec)
                    else:
                        print(f"过滤掉异常长度的矢量: {length} > {threshold}")
                
                self.polarization_vectors = filtered_vectors
                
                # 更新显示
                self.update_display_with_coordinates()
                
                self.statusBar.showMessage(f"已计算{len(self.polarization_vectors)}个极化矢量，过滤掉{len(vectors) - len(filtered_vectors)}个异常矢量")
            else:
                self.statusBar.showMessage("数据不足，无法计算极化矢量")
        except Exception as e:
            self.statusBar.showMessage("数据不足，计算极化矢量时出错")

    def clear_vectors(self):
        """清除显示的极化矢量"""
        self.polarization_vectors = None
        self.update_display_with_coordinates()
        self.statusBar.showMessage("已清除极化矢量")
    

def main():
    app = QApplication(sys.argv)
    window = AtomDetectorApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
