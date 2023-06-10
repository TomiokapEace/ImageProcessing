import cv2
import os
from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageFont, ImageDraw
import numpy as np

# 定义全局变量
drawing = False
ix, iy = -1, -1

# (1) 点运算
## 二值化操作
def threshold_algorithm(image):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 自适应阈值二值化
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # 返回二值化后的图像
    return thresh
# (2) 几何运算
## 旋转图像
def rotate_algorithm(image, angle=90):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))
# 缩放 - 交互功能
## 裁剪图像
# 裁剪图片的鼠标回调函数
def crop_callback(event, x, y, flags, param):
    global drawing, ix, iy, cropped

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_image = param.copy()
            cv2.rectangle(temp_image, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Crop - Press the ESC key to exit, retaining only the first cropped image.', temp_image)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cropped = param[iy:y, ix:x]
# 裁剪图片函数
def crop_algorithm(input_image):
    global cropped
    cropped = input_image.copy() # 初始化 cropped 变量
    # 创建一个显示图像的窗口
    window_name = 'Crop - Press the ESC key to exit, retaining only the first cropped image.'
    cv2.namedWindow(window_name)

    # 将鼠标回调函数绑定到窗口
    cv2.setMouseCallback(window_name, crop_callback, input_image)

    # 循环显示图像，直到用户按下关闭按钮或ESC键就退出，仅保留第一次裁切的画面。
    while True:
        cv2.imshow(window_name, input_image)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q') or key == ord('Q'):
            break

    # 关闭窗口并返回裁剪后的图像
    cv2.destroyWindow(window_name)
    return cropped
# (3) 滤波
## 均值滤波
def mean_filter_algorithm(image):
    # 使用3x3的核进行均值滤波
    result = cv2.blur(image, (3, 3))
    return result
## 高斯滤波
def gaussian_filter_algorithm(image):
    # 使用3x3的 Gaussian 核进行高斯滤波
    result = cv2.GaussianBlur(image, (3, 3), 0)
    return result
## 中值滤波
def median_filter_algorithm(image):
    # 使用3x3的大小进行中值滤波
    result = cv2.medianBlur(image, 3)
    return result
# (4) 边缘检测
## Canny 算法
def canny_algorithm(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用 Canny 算法进行边缘检测
    edges = cv2.Canny(gray, 100, 200)
    return edges
## 外轮廓检测
def contour_detection_algorithm(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用二值化处理
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 在原图像上绘制轮廓
    result = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return result
## 填充轮廓
def fill_contour_algorithm(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用二值化处理
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 对每个轮廓进行填充
    for contour in contours:
        cv2.drawContours(image, [contour], 0, (0, 255, 0), -1)
    return image
# (5) 直方图均衡化
## 全局直方图均衡化
def global_histogram_equalization_algorithm(image):
    # 功能：全局直方图均衡化
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用全局直方图均衡化
    equalized = cv2.equalizeHist(gray)
    # 返回均衡化后的图像
    return equalized
## 局部直方图均衡化
def local_histogram_equalization_algorithm(image):
    # 功能：局部直方图均衡化
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 创建CLAHE对象，定义局部直方图均衡化参数
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 应用局部直方图均衡化
    equalized = clahe.apply(gray)
    # 返回均衡化后的图像
    return equalized
## 限制对比度自适应直方图均衡化
def contrast_limited_adaptive_histogram_equalization_algorithm(image):
    # 功能：限制对比度自适应直方图均衡化
    # 将图像转换为Lab空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # 拆分通道
    L, a, b = cv2.split(lab)
    # 创建CLAHE对象，定义限制对比度自适应直方图均衡化参数
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 应用限制对比度自适应直方图均衡化
    L_equalized = clahe.apply(L)
    # 合并通道
    lab = cv2.merge((L_equalized, a, b))
    # 将图像转换回BGR空间
    equalized_image = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    # 返回均衡化后的图像
    return equalized_image
# (6) 形态学操作
## 开运算
def opening_operation_algorithm(image):
    # 功能：开运算
    # 定义结构元素
    kernel = np.ones((5, 5), np.uint8)
    # 应用开运算
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # 返回处理后的图像
    return opened
## 闭运算
def closing_operation_algorithm(image):
    # 功能：闭运算
    # 定义结构元素
    kernel = np.ones((5, 5), np.uint8)
    # 应用闭运算
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # 返回处理后的图像
    return closed
## 膨胀操作
def dilation_algorithm(image):
    # 定义一个 3x3 的结构元素
    kernel = np.ones((3, 3), np.uint8)
    # 使用 cv2.dilate 对图像进行膨胀操作
    dilated = cv2.dilate(image, kernel, iterations=1)
    # 返回膨胀后的图像
    return dilated
## 腐蚀操作
def erosion_algorithm(image):
    # 定义一个 3x3 的结构元素
    kernel = np.ones((3, 3), np.uint8)
    # 使用 cv2.erode 对图像进行腐蚀操作
    eroded = cv2.erode(image, kernel, iterations=1)
    # 返回腐蚀后的图像
    return eroded
# (7) 水印
## 添加文字水印
def watermark_algorithm(image):
    h, w = image.shape[:2]
    
    watermark_text = simpledialog.askstring("添加文字水印", "请输入文字内容：")
    
    # 生成水印图像
    img_wm = np.zeros_like(image, np.uint8)
    img_wm.fill(255)
    fontpath = "./ziti.TTF"  # 字体文件
    font = ImageFont.truetype(fontpath, 30)  # 字体大小
    img_pil = Image.fromarray(img_wm)
    draw = ImageDraw.Draw(img_pil)
    text_size = font.getsize(watermark_text)
    x, y = int(w / 2 - text_size[0] / 2), int(h / 2 - text_size[1] / 2)
    draw.text((x, y), watermark_text, font=font, fill=(255, 0, 0))
    img_wm = np.array(img_pil)

    # 调整水印图片尺寸并计算在图片中的位置
    wm_h, wm_w = img_wm.shape[:2]
    img_w, img_h = w, h
    x, y = int(img_w / 2 - wm_w /2), int(img_h / 2 - wm_h / 2)

    # 在图片上添加水印
    img_wm = cv2.resize(img_wm, (wm_w, wm_h))
    img_roi = image[y:y+wm_h, x:x+wm_w]
    img_add = cv2.addWeighted(img_roi, 1, img_wm, 0.2, 0)
    ret = img_add
    # image[y:y+wm_h, x:x+wm_w] = img_add (这样写是错的，添加水印无法撤回)
    
    # 返回加水印后的图片
    return ret

# 画图功能
# 定义画笔函数的回调函数
def pencil_mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(param, (ix, iy), (x, y), (0, 255, 0), 2)
            ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(param, (ix, iy), (x, y), (0, 255, 0), 2)   
# 画笔函数
def pencil_algorithm(input_image):
    # 创建一个显示图像的窗口
    window_name = 'Pencil - Press the ESC key to exit'
    cv2.namedWindow(window_name)

    # 将鼠标回调函数绑定到窗口
    cv2.setMouseCallback(window_name, pencil_mouse_callback, input_image)

    # 循环显示图像，直到用户按下关闭按钮或ESC键
    while True:
        cv2.imshow(window_name, input_image)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q') or key == ord('Q'):
            break

    # 关闭窗口并返回添加文本后的图像
    cv2.destroyWindow(window_name)
    return input_image

# 画矩形的鼠标回调函数
def rectangle_callback(event, x, y, flags, param):
    global drawing, ix, iy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_image = param.copy()
            cv2.rectangle(temp_image, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Rectangle - Press the ESC key to exit', temp_image)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(param, (ix, iy), (x, y), (0, 255, 0), 2)
# 画矩形函数
def rectangle_algorithm(input_image):
    # 创建一个显示图像的窗口
    window_name = 'Rectangle - Press the ESC key to exit'
    cv2.namedWindow(window_name)

    # 将鼠标回调函数绑定到窗口
    cv2.setMouseCallback(window_name, rectangle_callback, input_image)

    # 循环显示图像，直到用户按下关闭按钮或ESC键
    while True:
        cv2.imshow(window_name, input_image)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q') or key == ord('Q'):
            break

    # 关闭窗口并返回添加文本后的图像
    cv2.destroyWindow(window_name)
    return input_image

# 画圆的鼠标回调函数
def circle_callback(event, x, y, flags, param):
    global drawing, ix, iy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_image = param.copy()
            radius = int(((x - ix) ** 2 + (y - iy) ** 2) ** 0.5)
            cv2.circle(temp_image, (ix, iy), radius, (0, 255, 0), 2)
            cv2.imshow('Circle - Press the ESC key to exit', temp_image)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        radius = int(((x - ix) ** 2 + (y - iy) ** 2) ** 0.5)
        cv2.circle(param, (ix, iy), radius, (0, 255, 0), 2)
# 画圆函数
def circle_algorithm(input_image):
    # 创建一个显示图像的窗口
    window_name = 'Circle - Press the ESC key to exit'
    cv2.namedWindow(window_name)

    # 将鼠标回调函数绑定到窗口
    cv2.setMouseCallback(window_name, circle_callback, input_image)

    # 循环显示图像，直到用户按下关闭按钮或ESC键
    while True:
        cv2.imshow(window_name, input_image)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q') or key == ord('Q'):
            break

    # 关闭窗口并返回添加文本后的图像
    cv2.destroyWindow(window_name)
    return input_image

# 画任意直线的鼠标回调函数
def line_callback(event, x, y, flags, param):
    global drawing, ix, iy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_image = param.copy()
            cv2.line(temp_image, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Polygon - make any lines to form a polygon - Press the ESC key to exit', temp_image)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(param, (ix, iy), (x, y), (0, 255, 0), 2)
# 画任意直线函数
def line_algorithm(input_image):
    # 创建一个显示图像的窗口
    window_name = 'Polygon - make any lines to form a polygon - Press the ESC key to exit'
    cv2.namedWindow(window_name)

    # 将鼠标回调函数绑定到窗口
    cv2.setMouseCallback(window_name, line_callback, input_image)

    # 循环显示图像，直到用户按下关闭按钮或ESC键
    while True:
        cv2.imshow(window_name, input_image)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q') or key == ord('Q'):
            break

    # 关闭窗口并返回添加文本后的图像
    cv2.destroyWindow(window_name)
    return input_image
        
#另一种实现方式,使用询问框确定图形位置:
## 在图像上绘制矩形
def draw_rectangle(image, x1, y1, x2, y2):
    return cv2.rectangle(image.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
## 在图像上绘制圆形
def draw_circle(image, x, y, radius):
    return cv2.circle(image.copy(), (x, y), radius, (0, 255, 0), 2)
## 在图像上绘制多边形
def draw_polygon(image, points):
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    return cv2.polylines(image.copy(), [pts], True, (0, 255, 0), 2)
## 在图像上添加文字
def add_text(image, text, x, y, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 255, 0), thickness=2):
    return cv2.putText(image.copy(), text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
## 绘制矩形询问框
def draw_rectangle_dialog():
    x1 = simpledialog.askinteger("Draw Rectangle", "Enter X1:")
    y1 = simpledialog.askinteger("Draw Rectangle", "Enter Y1:")
    x2 = simpledialog.askinteger("Draw Rectangle", "Enter X2:")
    y2 = simpledialog.askinteger("Draw Rectangle", "Enter Y2:")
    
    global processed_image
    processed_image = draw_rectangle(processed_image, x1, y1, x2, y2)
    show_image(processed_image)
## 绘制圆形询问框
def draw_circle_dialog():
    x = simpledialog.askinteger("Draw Circle", "Enter X:")
    y = simpledialog.askinteger("Draw Circle", "Enter Y:")
    radius = simpledialog.askinteger("Draw Circle", "Enter radius:")
    
    global processed_image
    processed_image = draw_circle(processed_image, x, y, radius)
    show_image(processed_image)
## 绘制多边形询问框
def draw_polygon_dialog():
    points = simpledialog.askstring("Draw Polygon", "Enter points as tuples separated by semicolons (e.g. 10,20;30,40;50,60):")
    points = [tuple(map(int, p.split(','))) for p in points.split(';')]
    
    global processed_image
    processed_image = draw_polygon(processed_image, points)
    show_image(processed_image)
## 添加文字询问框
def add_text_dialog():
    text = simpledialog.askstring("Add Text", "Enter the text:")
    x = simpledialog.askinteger("Add Text", "Enter X:")
    y = simpledialog.askinteger("Add Text", "Enter Y:")

    global processed_image
    processed_image = add_text(processed_image, text, x, y)
    show_image(processed_image)