import argparse  # python的命令行解析的标准模块  可以让我们直接在命令行中就可以向程序中传入参数并让程序运行
import sys  # sys系统模块 包含了与Python解释器和它的环境有关的函数
import time  # 时间模块 更底层
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块
import numpy as np

import cv2  # opencv模块
import torch  # pytorch模块
import torch.backends.cudnn as cudnn  # cuda模块

FILE = Path(__file__).absolute()  # FILE = WindowsPath 'F:\yolo_v5\yolov5-U\detect.py'
# 将'F:/yolo_v5/yolov5-U'加入系统的环境变量  该脚本结束后失效
sys.path.append(FILE.parents[0].as_posix())  # add yolov5-U/ to path

# ----------------- 导入自定义的其他包 -------------------
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, xywh2xyxy, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, model_info, prune
from utils.myutils import *


@torch.no_grad()
def run(weights='weights/yolov5m.pt',  # 权重文件地址 默认 weights/best.pt
        source='data/images/',  # 测试数据文件(图片或视频)的保存路径 默认data/images
        imgsz=640,  # 输入图片的大小 默认640(pixels)
        conf_thres=0.25,  # object置信度阈值 默认0.25  用在nms中
        iou_thres=0.45,  # 做nms的iou阈值 默认0.45   用在nms中
        max_det=1000,  # 每张图片最多的目标数量  用在nms中
        device='',  # 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # 是否展示预测之后的图片或视频 默认False
        save_txt=False,  # 是否将预测的框坐标以txt文件格式保存 默认True 会在runs/detect/expn/labels下生成每张图片预测的txt文件
        save_conf=False,  # 是否保存预测每个目标的置信度到预测tx文件中 默认True
        save_crop=False,  # 是否需要将预测到的目标从原图中扣出来 剪切好 并保存 会在runs/detect/expn下生成crops文件，将剪切的图片保存在里面  默认False
        nosave=False,  # 是否不要保存预测后的图片  默认False 就是默认要保存预测后的图片
        classes=None,  # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留
        agnostic_nms=False,  # 进行nms是否也除去不同类别之间的框 默认False
        augment=False,  # 预测是否也要采用数据增强 TTA 默认False
        update=False,  # 是否将optimizer从ckpt中删除  更新模型  默认False
        project='runs/detect',  # 当前测试结果放在哪个主文件夹下 默认runs/detect
        name='exp',  # 当前测试结果放在run/detect下的文件名  默认是exp  =>  run/detect/exp
        exist_ok=False,  # 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
        line_thickness=3,  # bounding box thickness (pixels)   画框的框框的线宽  默认是 3
        hide_labels=False,  # 画出的框框是否需要隐藏label信息 默认False
        hide_conf=False,  # 画出的框框是否需要隐藏conf信息 默认False
        half=True,  # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
        prune_model=False,  # 是否使用模型剪枝 进行推理加速
        fuse=True,  # 是否使用conv + bn融合技术 进行推理加速
        ):
    # ===================================== 1、初始化一些配置 =====================================
    # 是否保存预测后的图片 默认nosave=False 所以只要传入的文件地址不是以.txt结尾 就都是要保存预测后的图片的
    save_img = not nosave and not source.endswith('.txt')  # save inference images   True
    # 是否是使用webcam 网页数据 一般是Fasle  因为我们一般是使用图片流LoadImages(可以处理图片/视频流文件)
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    # 检查当前Path(project) / name是否存在 如果存在就新建新的save_dir 默认exist_ok=False 需要重建
    # 将原先传入的名字扩展成新的save_dir 如runs/detect/exp存在 就扩展成 runs/detect/exp1
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # 如果需要save txt就新建save_dir / 'labels' 否则就新建save_dir
    # 默认save_txt=False 所以这里一般都是新建一个 save_dir(runs/detect/expn)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # Initialize 初始化日志信息
    set_logging()
    # 获取当前主机可用的设备
    device = select_device(device)
    # 如果设配是GPU 就使用half(float16)  包括模型半精度和输入图片半精度
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # ===================================== 2、载入模型和模型参数并调整模型 =====================================
    model = attempt_load(weights, map_location=device)
    if prune_model:
        model_info(model)  # 打印模型信息
        prune(model, 0.3)  # 对模型进行剪枝  加速推理
        model_info(model)  # 再打印模型信息  观察剪枝后模型变化
    if fuse:
        model = model.fuse()  # 将模型的conv+bn融合 可以加速推理
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size 保证img size必须是32的倍数
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to float16
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # ===================================== 3、加载推理数据 =====================================
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # ===================================== 4、推理前测试 =====================================
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # ===================================== 5、正式推理 =====================================
    t0 = time.time()
    # path: 图片/视频的路径
    # img: 进行resize + pad之后的图片
    # img0s: 原尺寸的图片
    # vid_cap: 当读取图片时为None, 读取视频时为视频源

    # stream_log：记录视频流每一帧累积信息的
    # class_score_lod: 80×n维的列表，表示80个类别的得分记录
    stream_log = []
    Box_thres = [0.6 for idx in range(80)]
    class_score_log = np.zeros((80, 1))
    new_frame = np.zeros(80)
    frame_idx = 0

    for path, img, im0s, vid_cap in dataset:
        # 分数记录
        if frame_idx >= 1:
            class_score_log = np.column_stack((class_score_log,new_frame))
        # 5.1、处理每一张图片的格式
        img = torch.from_numpy(img).to(device)  # numpy array to tensor and device
        img = img.half() if half else img.float()  # 半精度训练 uint8 to fp16/32
        img /= 255.0  # 归一化 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = time_synchronized()
        # pred shape=[1, num_boxes, xywh+obj_conf+classes] = [1, 18900, 25]
        pred = model(img, augment=augment)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        # ===================================至此，推理过程已经结束================================= #
        # 记录每张图片所有目标结果的列表
        frame_log = []
        # 记录图片里每个目标得分的列表
        score_list = []
        for i, det in enumerate(pred):  # detections per image
            if webcam:
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            # 当前图片路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
            p = Path(p)  # to Path
            # 图片/视频的保存路径save_path 如 runs\\detect\\exp8\\bus.jpg
            save_path = str(save_dir / p.name)  # img.jpg
            # txt文件(保存预测框坐标)保存路径 如 runs\\detect\\exp8\\labels\\bus
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # print string  输出信息  图片shape (w, h)
            s += '%gx%g ' % img.shape[2:]
            #  normalization gain gn = [w, h, w, h]  用于后面的归一化
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            # imc: for save_crop 在save_crop中使用
            im1 = im0
            # 以下if语句的意思是当前image里检测出了目标才执行
            if len(det):
                # 将预测信息（相对img_size 640）映射回原图 img0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # 输出信息s + 检测到的各个类别的目标个数
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # 保存预测信息: txt、img0上画框、crop_img
                for *xyxy, conf, cls in reversed(det):
                    coffset = get_centraloffset(xyxy, gn, normalize=True)  # 获得每个目标的中心偏移量coffset
                    # 将xyxy(左上角 + 右下角)格式转换为xywh(中心的 + 宽高)格式 并除以gn(whwh)做归一化 转为list再保存
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    thres = Box_thres[int(cls)]
                    box_rate = get_box_thres_rate(xywh, thres)  # 获取阈值比
                    box_size = get_box_size((xywh))  # 只获得框大小
                    score = box_rate / coffset  # 计分score
                    # 记录当前这个种类的特征
                    frame_log.append(
                        {"cls": names[int(cls)], "conf": conf, "loc": xyxy, "coffset": coffset, "box_rate": box_rate,
                         "box_size": box_size, "score": score})
                    score_list.append(score)
                    # 每次直接对应int(cls)的那个class_score_log进行append操作
                    if score >= class_score_log[int(cls), :][frame_idx]:
                        class_score_log[int(cls), :][frame_idx] = score
                    # 在原图上画框 + 将预测到的目标剪切出来 保存成图片 保存在save_dir/crops下
                    if save_img or save_crop or view_img:
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)

                # =====================================单个object检测结束================================= #
                target_idx = score_list.index(max(score_list))
                target = frame_log[target_idx]
                target_xyxy = target["loc"]
                im1 = info_on_img(im0, gn, zoom=[0.45, 0.9], label="Box_size: " + str(round(target["box_size"], 3)))
                im1 = info_on_img(im1, gn, zoom=[0.45, 0.95], label="Box_rate: " + str(round(target["box_rate"], 3)))
                im1 = info_on_img(im1, gn, zoom=[0.75, 0.95], label="Score: " + str(round(target["score"].item(), 3)))
                if target["box_rate"] > 1.5:
                    im1 = text_on_img(im1, gn, zoom=[0.05, 0.95], label="Grasping " + target["cls"])
                else:
                    im1 = plot_target_box(target_xyxy, im0, color=colors(0, True), line_thickness=2)
                    im1 = text_on_img(im1, gn, zoom=[0.05, 0.95], label="Targeting: " + target["cls"])
                stream_log.append(frame_log)

            else:
                im1 = text_on_img(im1, gn, zoom=[0.05, 0.95], label="No Target")
                stream_log.append(["None"])

            im1 = text_on_img(im1, gn, zoom=[0.05, 0.1], color=[0,0,0], label="Frame " + str(frame_idx))

            # 打印前向传播 + NMS 花费的时间
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            # 是否需要显示我们预测后的结果  img0(此时已将pred结果可视化到了img0中)
            if view_img:
                cv2.imshow(str(p), im1)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 是否需要保存图片或视频（检测后的图片/视频 里面已经被我们画好了框的） img0
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im1)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im1.shape[1], im1.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im1)
        frame_idx += 1
        # 至此结束当前帧

    # ===================================== 6、推理结束, 保存结果, 打印信息 =====================================
    # 保存预测的label信息 xywh等   save_txt
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        # strip_optimizer函数将optimizer从ckpt中删除  更新模型
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    # 打印预测的总时间


    print(frame_idx)
    print('39_bottle:', class_score_log[39, :])
    print('41_cup:', class_score_log[41, :])
    print('44_spoon:', class_score_log[44, :])
    print(f'Done. ({time.time() - t0:.3f}s)')

    filename = open('runs/bottle_score.txt', 'w')
    for value in class_score_log[39, :]:
        value = value.item()
        filename.write(str(value) + '\n')
    filename.close()

    filename = open('runs/cup_score.txt', 'w')
    for value in class_score_log[41, :]:
        value = value.item()
        filename.write(str(value) + '\n')
    filename.close()

    filename = open('runs/spoon_score.txt', 'w')
    for value in class_score_log[44, :]:
        value = value.item()
        filename.write(str(value) + '\n')
    filename.close()


def parse_opt():
    """
    opt参数解析
    weights: 模型的权重地址 默认 weights/best.pt
    source: 测试数据文件(图片或视频)的保存路径 默认data/images
    imgsz: 网络输入图片的大小 默认640
    conf-thres: object置信度阈值 默认0.25
    iou-thres: 做nms的iou阈值 默认0.45
    max-det: 每张图片最大的目标个数 默认1000
    device: 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
    view-img: 是否展示预测之后的图片或视频 默认False
    save-txt: 是否将预测的框坐标以txt文件格式保存 默认True 会在runs/detect/expn/labels下生成每张图片预测的txt文件
    save-conf: 是否保存预测每个目标的置信度到预测tx文件中 默认True
    save-crop: 是否需要将预测到的目标从原图中扣出来 剪切好 并保存 会在runs/detect/expn下生成crops文件，将剪切的图片保存在里面  默认False
    nosave: 是否不要保存预测后的图片  默认False 就是默认要保存预测后的图片
    classes: 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留
    agnostic-nms: 进行nms是否也除去不同类别之间的框 默认False
    augment: 预测是否也要采用数据增强 TTA
    update: 是否将optimizer从ckpt中删除  更新模型  默认False
    project: 当前测试结果放在哪个主文件夹下 默认runs/detect
    name: 当前测试结果放在run/detect下的文件名  默认是exp
    exist-ok: 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
    line-thickness: 画框的框框的线宽  默认是 3
    hide-labels: 画出的框框是否需要隐藏label信息 默认False
    hide-conf: 画出的框框是否需要隐藏conf信息 默认False
    half: 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images/3obj', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--prune-model', default=False, action='store_true', help='model prune')
    parser.add_argument('--fuse', default=False, action='store_true', help='fuse conv and bn')
    opt = parser.parse_args()
    return opt


def main(opt):
    # 调用colorstr函数彩色打印选择的opt参数
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    # 检查已经安装的包是否满足requirements对应txt文件的要求
    check_requirements(exclude=('tensorboard', 'thop'))
    # 执行run 开始推理
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
