import torch
import cv2
import numpy as np


def get_centraloffset(xyxy, gn, normalize=False):
    """
    计算的是边界框中心与整个画面中心的L2距离（默认为像素距离，normalize后为归一化的结果），其中
    归一化的方式为L2像素距离除以一半对角线像素长度，越接近0表示离中心越近，约接近1表示离中心越远
    """
    ic = torch.tensor([gn[0] / 2, gn[1] / 2])  # image_centre
    i2c = (ic[0].pow(2) + ic[1].pow(2)).sqrt()  # half_diagonal
    bc = torch.tensor([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2])  # box_centre
    cft = (ic[0] - bc[0]).pow(2) + (ic[1] - bc[1]).pow(2)  # centraloffset
    if normalize:
        cft = cft.sqrt() / i2c
    else:
        cft = cft.sqrt()
    return cft

def get_centeroffset_2version(xywh, normalize=False):
    """
    直接用框中心与(0.5,0.5)的距离表示
    """
    ic = torch.tensor([0.5, 0.5])
    i2c = (ic[0].pow(2) + ic[1].pow(2)).sqrt()
    cft = (xywh[0] - ic[0]).pow(2) + (xywh[1] - ic[1]).pow(2)
    if normalize:
        cft = cft.sqrt() / i2c
    else:
        cft = cft.sqrt()
    return cft

def get_box_thres_rate(xywh, thres):
    """
    计算边界框大小与标准化阈值的比例
    """
    rate = max(xywh[2], xywh[3]) / thres
    return rate

def get_box_size(xywh):
    """
    计算边界框大小
    """
    size = max(xywh[2], xywh[3])
    return size

def check_grasp(box_rate, cls, grasping_flag, x=1):
    if grasping_flag[0]:  # 上一时刻是抓的状态
        if cls == grasping_flag[1]:  # 如果还是要抓的目标
            if box_rate > x:  # 还在接近
                grasping_flag[0] = True
            else:  # 已经离开了
                grasping_flag[0] = False
                grasping_flag[1] = "None"
        else:  # 检测到了其他东西
            grasping_flag[0] = True
    else:  # 上一时刻不是抓取状态
        if box_rate > x:  # 新的目标进入抓取状态
            grasping_flag[0] = True
            grasping_flag[1] = cls
        else:  # 还在瞄准
            grasping_flag[0] = False
            grasping_flag[1] = "None"
    return grasping_flag

def check_grasp_null(grasping_flag):
    if grasping_flag[0]:  # 上一时刻是抓的状态
        grasping_flag[0] = True
    else:  # 上一时刻不是抓取状态
        grasping_flag[0] = False
    return grasping_flag


def plot_target_box(x, im, color=(128, 128, 128), label=None, line_thickness=2):
    """一般会用在detect.py中在nms之后变量每一个预测框，再将每个预测框画在原图上
    使用opencv在原图im上画一个bounding box
    :params x: 预测得到的bounding box  [x1 y1 x2 y2]
    :params im: 原图 要将bounding box画在这个图上  array
    :params color: bounding box线的颜色
    :params labels: 标签上的框框信息  类别 + score
    :params line_thickness: bounding box的线宽，-1表示框框为实心
    """
    # check im内存是否连续
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    # 这里在画高亮部分
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle: 在im上画出框框   c1: start_point(x1, y1)  c2: end_point(x2, y2)
    # 注意: 这里的c1+c2可以是左上角+右下角  也可以是左下角+右上角都可以
    blk = np.zeros(im.shape, np.uint8)
    cv2.rectangle(blk, c1, c2, color, -1)  # 注意在 blk的基础上进行绘制；
    img = cv2.addWeighted(im, 1.0, blk, 0.5, 1)
    return img

def text_on_img(im, gn, zoom, color=[0,0,255], label=None, line_thickness=2):
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    scale = int(gn[1]) / 666 + 0.2
    tf = int(scale)  # label字体的线宽 font thickness
    d1 = (int(gn[0] * zoom[0]), int(gn[1] * zoom[1]))
    img = cv2.putText(im, label, (d1[0], d1[1]), 0, scale, color, thickness=tf + 1, lineType=cv2.LINE_AA)
    return img

def info_on_img(im, gn, zoom, color=[0,0,255], label=None, line_thickness=2):
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    scale = int(gn[1]) / 666 + 0.2
    tf = int(scale)  # label字体的线宽 font thickness
    d1 = (int(gn[0] * zoom[0]), int(gn[1] * zoom[1]))
    img = cv2.putText(im, label, (d1[0], d1[1]), 0, scale, color, thickness=tf + 1, lineType=cv2.LINE_AA)
    return img

def save_score(path, cls, class_score_log):
    filename = open(path, 'w')
    for value in class_score_log[cls, :]:
        value = value.item()
        filename.write(str(value) + '\n')
    filename.close()
    pass

def CLS(result_log):
    """
    种类定位抑制（Class Localization Suppression），在一定位置范围内的目标只能有一个种类和预测框输出
    发现种类抑制函数貌似并不是必须的？因为同一个位置的目标就算有两个输出，预测种类都是类似的，因此不影响对target的判断。
    这样输出累积的score就会有两个target非常相似，如果它们都是最高值（或相差很少），那么那个时候再做抑制就可以了；
    如果它们都不是最高值，那么说明在决策的时候可以直接忽略，都不是target。
    """
    for i, item in enumerate(result_log):
        # 先计算相邻元素坐标，小于某个阈值就标记为抑制——聚类？
        # 抑制之后重新输出一个新的result_log，留下来的种类是conf最大的那一个
        pass
    pass

def object_tracking():
    """
    对每一帧的各个目标按位置进行编号追踪
    """
    pass
