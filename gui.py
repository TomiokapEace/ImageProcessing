import cv2
import os
from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import numpy as np
import image_processing as im

# 定义全局变量
image_path = None
image = None
processed_image = None
temp_image = None
history = []
history_index = -1
image_on_canvas = None
zoom_ratio = 1  # 定义缩放比例


# 文件操作
## 显示图像
def show_image(img):
    global temp_image, image_on_canvas
    
    # 创建画布并显示图像
    if temp_image is None:
        temp_image = Canvas(root, width=window_width, height=window_height-50)
        temp_image.pack(side=LEFT, padx=10, pady=10)

        # 在此处将鼠标移动、滚轮滚动和拖拽事件绑定到画布上
        temp_image.bind("<Motion>", on_mouse_move)
        temp_image.bind("<MouseWheel>", on_scroll)     # for Windows and MacOS
        temp_image.bind("<Button-4>", on_scroll)       # for Linux (up scroll)
        temp_image.bind("<Button-5>", on_scroll)       # for Linux (down scroll)
        temp_image.bind("<B1-Motion>", on_drag)

    # 将 OpenCV 图像转换为 PIL 图像格式并显示
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    image_on_canvas = temp_image.create_image(0, 0, anchor=NW, image=img)
    temp_image.image = img
## 选择图像文件
def select_image():
    global image_path, image, processed_image, history, history_index
    # 打开文件对话框
    filetypes = (("JPEG Files", "*.jpg"), ("PNG Files", "*.png"))
    image_path = filedialog.askopenfilename(title="Select Image File", filetypes=filetypes)
    # 如果用户选择了文件，则读取图像并显示，并更新标题栏
    if image_path:
        root.title("图像处理系统 - 20H034160209 - 刘昕 - " + image_path)  # 更新标题栏
        image = cv2.imread(image_path)
        processed_image = image.copy()
        show_image(image)
        # 清空历史记录
        history = [processed_image]
        history_index = 0
## 撤销操作
def undo():
    global processed_image, history_index
    if history_index > 0:
        history_index -= 1
        processed_image = history[history_index]
        show_image(processed_image)
## 重做操作
def redo():
    global processed_image, history_index
    if history_index < len(history) - 1:
        history_index += 1
        processed_image = history[history_index]
        show_image(processed_image)
## 新建空白图像
def new_image_question(): # 新建图片前查看是否有未保存图片，询问保存
    if image_path:
        answer = messagebox.askyesno(title="图像处理系统", message="是否保存当前图片?")
        if answer:
            save_image()
    new_image()
def new_image():
    global image_path, image, processed_image, history, history_index
    root.title("图像处理系统 - 20H034160209 - 刘昕 - 新建图像")  # 更新标题栏
    # 读取同文件夹下的 white.jpg 图像并复制给 processed_image
    processed_image = cv2.imread('white.jpg')
    image_path = None
    # 显示空白图像
    show_image(processed_image)
    # 清空历史记录
    history = [processed_image]
    history_index = 0
## 保存图像
def save_image():
    global processed_image
    # 打开文件对话框
    filetypes = (("JPEG Files", "*.jpg"), ("PNG Files", "*.png"))
    save_path = filedialog.asksaveasfilename(filetypes=filetypes)
    # 如果用户选择了文件，则保存图像
    if save_path:
        cv2.imwrite(save_path, processed_image)

# 创建主窗口和菜单栏
root = Tk()
root.title("图像处理系统 - 20H034160209 - 刘昕")
menubar = Menu(root)
root.config(menu=menubar)

# 创建“文件”菜单
file_menu = Menu(menubar, tearoff=0)
file_menu.add_command(label="打开", command=select_image)
file_menu.add_command(label="新建", command=new_image_question)
file_menu.add_command(label="另存为", command=save_image)
file_menu.add_separator()
file_menu.add_command(label="退出", command=root.quit)
menubar.add_cascade(label="文件", menu=file_menu)

# 创建“编辑”菜单
edit_menu = Menu(menubar, tearoff=0)
edit_menu.add_command(label="撤销", command=undo, accelerator="Ctrl+Z")
edit_menu.add_command(label="重做", command=redo, accelerator="Ctrl+Y")
menubar.add_cascade(label="编辑", menu=edit_menu)

# 图像处理
# 创建“图像处理”菜单和二级菜单
image_menu = Menu(menubar, tearoff=0)
filter_menu = Menu(image_menu, tearoff=0)
detection_menu = Menu(image_menu, tearoff=0)
equalization_menu = Menu(image_menu, tearoff=0)
operation_menu = Menu(image_menu, tearoff=0)
# (1) 点运算
## 二值化
def threshold():
    global processed_image, history, history_index
    # 调用函数实现二值化
    processed_image = im.threshold_algorithm(processed_image)
    show_image(processed_image)
    # 添加到历史记录
    history.append(processed_image)
    history_index += 1
# (2) 几何运算
## 旋转图像
def rotate():
    global processed_image, history, history_index
    processed_image = im.rotate_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
## 裁剪图像 - 按下ESC键就退出，仅保留第一次裁切的画面。
def crop():
    global processed_image, history, history_index
    processed_image = im.crop_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
# (3) 滤波
## 均值滤波
def mean_filter():
    global processed_image, history, history_index
    processed_image = im.mean_filter_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
## 高斯滤波
def gaussian_filter():
    global processed_image, history, history_index
    processed_image = im.gaussian_filter_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
## 中值滤波
def median_filter():
    global processed_image, history, history_index
    processed_image = im.median_filter_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
# (4) 边缘检测
## Canny 算法
def canny():
    global processed_image, history, history_index
    processed_image = im.canny_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
## 外轮廓检测
def contour_detection():
    global processed_image, history, history_index
    processed_image = im.contour_detection_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
## 填充轮廓
def fill_contour():
    global processed_image, history, history_index
    processed_image = im.fill_contour_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
# (5) 直方图均衡化
## 全局直方图均衡化
def global_histogram_equalization():
    global processed_image, history, history_index
    processed_image = im.global_histogram_equalization_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
## 局部直方图均衡化
def local_histogram_equalization():
    global processed_image, history, history_index
    processed_image = im.local_histogram_equalization_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
## 限制对比度自适应直方图均衡化
def contrast_limited_adaptive_histogram_equalization():
    global processed_image, history, history_index
    processed_image = im.contrast_limited_adaptive_histogram_equalization_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
# (6) 形态学操作
## 开运算
def opening_operation():
    global processed_image, history, history_index
    processed_image = im.opening_operation_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
## 闭运算
def closing_operation():
    global processed_image, history, history_index
    processed_image = im.closing_operation_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
## 膨胀操作
def dilation():
    global processed_image, history, history_index
    processed_image = im.dilation_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
## 腐蚀操作
def erosion():
    global processed_image, history, history_index
    processed_image = im.erosion_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
# (7) 水印
## 添加文字水印
def watermark():
    global processed_image, history, history_index
    processed_image = im.watermark_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1

# 添加菜单项和二级菜单项
image_menu.add_command(label="二值化", command=threshold)
image_menu.add_command(label="旋转", command=rotate)
image_menu.add_command(label="添加水印", command=watermark)
image_menu.add_command(label="裁剪", command=crop)

filter_menu.add_command(label="均值滤波", command=mean_filter)
filter_menu.add_command(label="高斯滤波", command=gaussian_filter)
filter_menu.add_command(label="中值滤波", command=median_filter)

detection_menu.add_command(label="Canny 算法", command=canny)
detection_menu.add_command(label="外轮廓检测", command=contour_detection)
detection_menu.add_command(label="填充轮廓", command=fill_contour)

equalization_menu.add_command(label="全局直方图均衡化", command=global_histogram_equalization)
equalization_menu.add_command(label="局部直方图均衡化", command=local_histogram_equalization)
equalization_menu.add_command(label="限制对比度自适应直方图均衡化", command=contrast_limited_adaptive_histogram_equalization)

operation_menu.add_command(label="开运算", command=opening_operation)
operation_menu.add_command(label="闭运算", command=closing_operation)
operation_menu.add_command(label="膨胀操作", command=dilation)
operation_menu.add_command(label="腐蚀操作", command=erosion)

menubar.add_cascade(label="图像处理", menu=image_menu)
image_menu.add_cascade(label="滤波", menu=filter_menu)
image_menu.add_cascade(label="边缘检测", menu=detection_menu)
image_menu.add_cascade(label="直方图均衡化", menu=equalization_menu)
image_menu.add_cascade(label="形态学操作", menu=operation_menu)

# 画图功能
## 画笔功能
def pencil():
    global processed_image, history, history_index
    processed_image = im.pencil_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
# 绘制矩形
def rectangle():
    global processed_image, history, history_index
    processed_image = im.rectangle_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
# 绘制圆形
def circle():
    global processed_image, history, history_index
    processed_image = im.circle_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1
# 绘制多边形（直线拼接）
def polygon():
    global processed_image, history, history_index
    processed_image = im.line_algorithm(processed_image)
    show_image(processed_image)
    history.append(processed_image)
    history_index += 1

# 创建“画图功能”菜单
draw_menu = Menu(menubar, tearoff=0)
# 添加菜单项
draw_menu.add_command(label="画笔", command=pencil)
draw_menu.add_command(label="绘制矩形", command=rectangle)
draw_menu.add_command(label="绘制圆形", command=circle)
draw_menu.add_command(label="绘制多边形", command=polygon)
menubar.add_cascade(label="画图功能", menu=draw_menu)

# 交互功能
## 鼠标移动
def on_mouse_move(event):
    x, y = event.x, event.y
    if processed_image is not None:
        h, w, _ = processed_image.shape
        if x >= 0 and x < w and y >= 0 and y < h:
            r, g, b = processed_image[y, x]
            statusbar.config(text=f"鼠标位置：({x}, {y}) | 颜色：({r}, {g}, {b}) | 缩放比例：{zoom_ratio * 100:.1f}%")  # 修改此行以显示缩放比例
        else:
            statusbar.config(text=f"鼠标位置：({x}, {y}) | 缩放比例：{zoom_ratio * 100:.1f}%")
    else:
        statusbar.config(text=f"鼠标位置：({x}, {y}) | 缩放比例：{zoom_ratio * 100:.1f}%")
## 滚轮
def on_scroll(event):
    global processed_image, zoom_ratio  # 修改此行，添加 zoom_ratio
    delta = -1 if event.delta > 0 else 1
    scale_factor = 1 - delta * 0.1

    zoom_ratio *= scale_factor  # 更新缩放比例

    # 获取原始图像的尺寸
    h, w, _ = processed_image.shape
    # 计算缩放后的图像尺寸
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    # 缩放图像
    resized_img = cv2.resize(processed_image, (new_w, new_h))

    # 显示缩放后的图像
    show_image(resized_img)

    # 更新 processed_image 为缩放后的图像
    processed_image = resized_img.copy()
## 拖拽
def on_drag(event):
    if temp_image:
        # 移动画布显示区域
        temp_image.scan_dragto(event.x, event.y, gain=1)

# 将主窗口放到屏幕中央
window_width = 800
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width - window_width) / 2)
y = int((screen_height - window_height) / 2)
root.geometry("{}x{}+{}+{}".format(window_width, window_height, x, y))

# 创建状态栏
statusbar = Label(root, text="鼠标位置：(0, 0) | 缩放比例：100%", relief=SUNKEN, anchor=W)
statusbar.pack(side=BOTTOM, fill=X)

# 运行主循环
temp_image = None
root.mainloop()
