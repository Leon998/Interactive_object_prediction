import torch


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
