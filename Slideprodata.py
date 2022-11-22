import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import openslide as ol
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image
import torch
import argparse
import torch.utils.data as data
import re

'''
第一步：seg_slide获取组织轮廓
第二步：tiling_slide根据提取的组织轮廓，获取patch对应的坐标点
第三步 ：vis_slide展示组织轮廓
第四步：展示patch拼接的slide
'''


'''-----------------------------------------第一步：获取组织轮廓-----------------------------------------'''
def seg_slide(slide, seg_level, patch_size):
    """
        Segment the tissue via HSV -> Median thresholding -> Binary threshold
       # 获取某一level的轮廓
    """

    def _filter_contours(contours, hierarchy, filter_params):  # 孔洞滤波
        """
            Filter contours by: area.
            hierarchy只有父轮廓和内嵌轮廓的索引
        """
        filtered = []

        # find foreground contours (parent == -1)
        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)#获取没有内嵌轮廓的索引(也就是前向沦落的索引) np.flatnonzero(hierarchy[:, 1] == -1)获取非零元素的索引构建新数组

        for cont_idx in hierarchy_1:
            cont = contours[cont_idx]
            a = cv2.contourArea(cont)
            if a > filter_params['a_t']:
                filtered.append(cont_idx)

        all_holes = []
        for parent in filtered:
            all_holes.append(np.flatnonzero(hierarchy[:, 1] == parent))

        foreground_contours = [contours[cont_idx] for cont_idx in filtered]

        hole_contours = []

        for hole_ids in all_holes:
            unfiltered_holes = [contours[idx] for idx in hole_ids]
            unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
            unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
            filtered_holes = []

            for hole in unfilered_holes:
                if cv2.contourArea(hole) > filter_params['a_h']:
                    filtered_holes.append(hole)

            hole_contours.append(filtered_holes)

        return foreground_contours, hole_contours

    level_dim = slide.level_dimensions
    mthresh = 7
    sthresh = 120
    sthresh_up = 255
    close = 4
    ref_patch_size = 512
    filter_params = {'a_t': 1, 'a_h': 1, 'max_n_holes': 1}

    img = np.array(slide.read_region((0, 0), seg_level, level_dim[seg_level]))#获取原始的图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to GRAY space，将原始图像变成灰度图
    img_med = cv2.medianBlur(img_gray, mthresh)  # Apply median blurring， 使用中值滤波去噪，突出组织结构
    # Thresholding
    _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY_INV)#提取组织结构,
    # plt.imshow(img_otsu)
    # plt.show()

    if close > 0:
        kernel = np.ones((close, close), np.uint8)
        img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_OPEN, kernel)
        #morphology形态学，cv2.morphologyEx(src, op, kernel)进行形态学操作，src是输入的图片，op是进行变化的方式
        #cv2.MORPH_OPEN进行开运算，指的是先进行腐蚀操作，然后进行膨胀操作。
        # cv2.MORPH_CLOSE进行闭运算，指的是先进行膨胀操作，在进行腐蚀操作,也是用于图像去噪操作


    scale = _assertLevelDownsamples(slide)[seg_level]#获取缩放因子
    scaled_patch_area = int(ref_patch_size ** 2 / (scale[0] * scale[1]))
    filter_params['a_t'] = filter_params['a_t'] * scaled_patch_area
    filter_params['a_h'] = filter_params['a_h'] * scaled_patch_area

    # Find and filter contours
    contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # Find contours 检测物体轮廓  hierarchy等级制度
    #cv2.findContours()函数用于物体轮廓检测，opencv2返回两个值contours(轮廓本身，是一个list，这个list中每个元素都是一个轮廓),
    # hierarchy(每个轮廓对应的属性，这是一个ndarray数组，其中的元素个数和轮廓个数相同，每个元素中有四个值，分别表示后一个轮廓，前一个轮廓，父轮廓，内嵌轮廓，没有的话就是负值)，opencv3返回三个值, img, contours, hierarchy
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]#获取父轮廓和内嵌轮廓的索引
    if filter_params:
        foreground_contours, hole_contours = _filter_contours(contours, hierarchy,filter_params)  # Necessary for filtering out artifacts，获取最重要的轮廓

    #输出的是level0下的轮廓坐标
    contours_tissue = scaleContourDim(foreground_contours, scale)#获取主要组织的轮廓。把提取的其他level的坐标点，转化为scale下
    holes_tissue = scaleHolesDim(hole_contours, scale)#获取组织中孔洞的轮廓
    return contours_tissue, holes_tissue

'''-----------------------------------------第三步：展示组织轮廓-----------------------------------------'''
def vis_slide(slide, contours_tissue, holes_tissue, vis_level):
    """
    可视化带有轮廓的病理图像
    :param slide:
    :param contours_tissue:
    :param holes_tissue:
    :param vis_level:
    :return:
    """
    level_dim = slide.level_dimensions
    line_thickness = 250
    color = (0, 255, 0)
    hole_color = (0, 0, 255)
    annot_color = (255, 0, 0)
    crop_window = None
    max_size = None

    img = np.array(slide.read_region((0, 0), vis_level, level_dim[vis_level]).convert("RGB"))
    downsample = _assertLevelDownsamples(slide)[vis_level]
    scale = [1 / downsample[0], 1 / downsample[1]]  # Scaling from 0 to desired level
    line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
    if contours_tissue is not None:
        cv2.drawContours(img, scaleContourDim(contours_tissue, scale),
                         -1, color, line_thickness, lineType=cv2.LINE_8)

        for holes in holes_tissue:
            cv2.drawContours(img, scaleContourDim(holes, scale),
                             -1, hole_color, line_thickness, lineType=cv2.LINE_8)

    img = Image.fromarray(img)

    if crop_window is not None:
        top, left, bot, right = crop_window
        left = int(left * scale[0])
        right = int(right * scale[0])
        top = int(top * scale[1])
        bot = int(bot * scale[1])
        crop_window = (top, left, bot, right)
        img = img.crop(crop_window)
    w, h = img.size
    if max_size is not None and (w > max_size or h > max_size):
        resizeFactor = max_size / w if w > h else max_size / h
        img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))
    return img


'''-----------------------------------------第二步：根据组织轮廓提取patch对应的坐标-----------------------------------------'''
def tiling_slide(slide, contours_tissue, holes_tissue, patch_level, patch_size, patch_step, name):
    """
    输出病理图片轮廓内组织的坐标点
    :param slide:
    :param contours_tissue:
    :param holes_tissue:
    :param patch_level:
    :param patch_size:
    :param patch_step:
    :param save_path:
    :param name:
    :return:
    """
    grid = []
    print("Creating patches for: ", name, "...", )
    num = 0
    for idx, cont in enumerate(contours_tissue):
        print("提取病理图片{}中第{}个contours的组织坐标！".format(name, idx))
        patch_gen = _getPatchGenerator(slide, cont, idx, patch_level, holes_tissue, patch_size, patch_step)
        for coord in patch_gen:
            x, y = coord["x"], coord["y"]
            grid.append((x, y))
            num += 1
    print("共采集{}个坐标点".format(num))
    return grid

def isInContourV1(cont, pt, patch_size=None):
    return 1 if cv2.pointPolygonTest(cont, pt, False) >= 0 else 0
"""
cv2.pointPolygonTest检测一个点是否在contours内  cv2.pointPolygonTest(contours, point, measureDist) 三个参数，分别是轮廓，坐标，是否计算坐标到轮廓的距离
点在轮廓内，返回正值，在轮廓外，返回负值，在轮廓上，返回0 如果measureDist=True，返回的值就是点到轮廓的距离
"""


def isInContourV2(cont, pt, patch_size=256):
    return 1 if cv2.pointPolygonTest(cont, (pt[0] + patch_size / 2, pt[1] + patch_size / 2), False) >= 0 else 0


def isInContourV3(cont, pt, patch_size=256):
    center = (pt[0] + patch_size // 2, pt[1] + patch_size // 2)
    all_points = [(center[0] - patch_size // 4, center[1] - patch_size // 4),
                  (center[0] + patch_size // 4, center[1] + patch_size // 4),
                  (center[0] + patch_size // 4, center[1] - patch_size // 4),
                  (center[0] - patch_size // 4, center[1] + patch_size // 4)
                  ]
    for points in all_points:
        if cv2.pointPolygonTest(cont, points, False) >= 0:
            return 1

    return 0


def isInContours(cont_check_fn, contour, pt, holes=None, patch_size=256):
    #0是不在轮廓内，1是在轮廓内
    if cont_check_fn(contour, pt, patch_size):
        if holes is not None:
            return not isInHoles(holes, pt, patch_size)
        else:
            return 1
    return 0


def isInHoles(holes, pt, patch_size):
    for hole in holes:
        if cv2.pointPolygonTest(hole, (pt[0] + patch_size / 2, pt[1] + patch_size / 2), False) > 0:
            return 1

    return 0


def isWhitePatch(patch, satThresh=5):
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return True if np.mean(patch_hsv[:, :, 1]) < satThresh else False


def isBlackPatch(patch, rgbThresh=40):
    return True if np.all(np.mean(patch, axis=(0, 1)) < rgbThresh) else False

def _getPatchGenerator(slide, cont, cont_idx, patch_level, holes_tissue, patch_size=256, step_size=256, use_padding = True):
    """
    get patch
    :param slide: openslide.OpenSlide读取的slide图片
    :param cont: contours组织轮廓
    :param cont_idx:组织轮廓的索引
    :param patch_level:在什么倍率下切patch
    :param save_path:保存的路径
    :param holes_tissue:
    :param patch_size:
    :param step_size:步长
    :param name:
    :return:
    """
    white_black = True
    white_thresh = 5
    black_thresh = 40
    contour_fn = 'four_pt'

    level_dim = slide.level_dimensions

    start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (
        0, 0, level_dim[patch_level][0], level_dim[patch_level][1])#cv2.boundingRect用最小矩形把其中一个组织包裹起来
    print("Bounding Box:", start_x, start_y, w, h)
    print("Contour Area:", cv2.contourArea(cont))#cv2.contourArea计算框内面积


    # the downsample corresponding to the patch_level
    patch_downsample = (int(_assertLevelDownsamples(slide)[patch_level][0]),
                        int(_assertLevelDownsamples(slide)[patch_level][1]))#获取对应level的降采样因子
    # size of patch at level 0 (reference size)
    ref_patch_size = (patch_size * patch_downsample[0], patch_size * patch_downsample[1])#获取在level0下的patch的尺寸

    # step sizes to take at levl 0
    step_size_x = step_size * patch_downsample[0]*1
    step_size_y = step_size * patch_downsample[1]*1   #获取在level0下的Patch的步长,*1,用来确定是否overlap的采集patch

    if contour_fn == 'four_pt':
        cont_check_fn = isInContourV3
    elif contour_fn == 'center':
        cont_check_fn = isInContourV2
    elif contour_fn == 'basic':
        cont_check_fn = isInContourV1
    else:
        raise NotImplementedError

    img_w, img_h = level_dim[0]#获取level0时的图片的尺寸

    if use_padding:
        stop_y = start_y + h
        stop_x = start_x + w
    else:
        stop_y = min(start_y + h, img_h - ref_patch_size[1])  # 这个地方没有使用
        stop_x = min(start_x + w, img_w - ref_patch_size[0])

    # stop_y = min(start_y + h, img_h - ref_patch_size[1])
    # stop_x = min(start_x + w, img_w - ref_patch_size[0])

    count = 0
    for y in range(start_y, stop_y, step_size_y):
        for x in range(start_x, stop_x, step_size_x):

            if not isInContours(cont_check_fn, cont, (x, y), holes_tissue[cont_idx],
                                ref_patch_size[0]):  # point not inside contour and its associated holes
                continue #如果这个点不在轮廓内，则继续执行（调出本次循环）

            count += 1
            patch_PIL = slide.read_region((x, y), patch_level, (ref_patch_size[0], ref_patch_size[1])).convert('RGB')

            if white_black:
                if isBlackPatch(np.array(patch_PIL), rgbThresh=black_thresh) or isWhitePatch(
                        np.array(patch_PIL), satThresh=white_thresh):
                    continue

            # x, y coordinates become the coordinates in the downsample, and no long correspond to level 0 of WSI
            patch_info = {'x': x // (patch_downsample[0]),
                          'y': y // (patch_downsample[1])}

            yield patch_info  # yield是生成器，return返回的是一个值

            # patch_info = {'x': x // (patch_downsample[0]),
            #               'y': y // (patch_downsample[1] ), 'cont_idx': cont_idx,
            #               'patch_level': patch_level,
            #               'downsample': slide.level_downsamples[patch_level],
            #               'downsampled_level_dim': tuple(np.array(slide.level_dim[patch_level]) ),
            #               'level_dim': slide.level_dim[patch_level],
            #               'patch_PIL': patch_PIL, 'name': slide.name, 'save_path': save_path}# 最后存的点，把点转化到了要提取的组织对应的level，比如level2，所以在region的时候要在放大到level0的倍率
            #
            # yield patch_info

            # #一些参数的含义
            # patch_level 表示当前提取的组织是level几下的组织
            # level_downsamples [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (8.0, 8.0), (16.0, 16.0), (32.0, 32.0), (64.0, 64.0), (128.0, 128.0),(256.0, 256.0), (512.0, 512.0)]
            # level_dim ((31744, 37888), (15872, 18944), (7936, 9472), (3968, 4736), (1984, 2368), (992, 1184), (496, 592),(248, 296), (124, 148), (62, 74))
            # name '70533_0-Tumor'

    print("patches extracted: {}".format(count))

def _assertLevelDownsamples(slide):
    """
    将slide的所有降采样因子返回成一个列表，如下所示
    <class 'list'>: [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (8.0, 8.0), (16.0, 16.0), (32.0, 32.0), (64.0, 64.0), (128.0, 128.0)]
    :param slide:
    :return:
    """
    level_downsamples = []
    dim_0 = slide.level_dimensions[0]
    #zip用于可迭代对象作为参数，将对象中的元素打包成元组构建成列表
    for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):#将下采样因子和对应的维度打包成元组
        estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
        level_downsamples.append(estimated_downsample) if estimated_downsample != (
            downsample, downsample) else level_downsamples.append((downsample, downsample))

    return level_downsamples

def scaleContourDim(contours, scale):
    return [np.array(cont * scale, dtype='int32') for cont in contours]

def scaleHolesDim(contours, scale):
    return [[np.array(hole * scale, dtype='int32') for hole in holes] for holes in contours]


'''--------------------------------第四步：根据提取的patch坐标点，拼接出来slide------------------------------'''
def StitchPatches(grid, slide,downscale=4, draw_grid=False, bg_color=(0, 0, 0), alpha=-1):
    coordls = []
    for gi in grid:
        gi_list = []
        gi_list.append(gi[0])
        gi_list.append(gi[1])
        coordls.append(gi_list)
    source_coords = np.array(coordls)
    # coords = list(grid ) #ndarray[[2424 3176],[3176 384],[3176 768],...]
    # 对应的level，slide的size((35712, 26368), (17856, 13184), (8928, 6592), (4464, 3296), (2232, 1648), (1116, 824), (558, 412), (279, 206))
    w, h =  slide.level_dimensions[args.level]

    print('original size: {} x {}'.format(w, h))
    w = w // downscale
    h = h // downscale
    resize_coords = (source_coords / downscale).astype(np.int32)  # 转化到对应尺寸的点
    print('downscaled size for stiching: {} x {}'.format(w, h))
    print('number of patches: {}'.format(len(grid)))
    downscaled_shape = (args.patch_size // downscale, args.patch_size // downscale)

    heatmap = Image.new(size=(w, h), mode="RGB", color=bg_color)
    heatmap = np.array(heatmap)
    heatmap = DrawMap(heatmap,source_coords, resize_coords, downscaled_shape, indices=None, draw_grid=draw_grid)  # dset是图片，coords是坐标点

    return heatmap


def DrawMap(canvas, source_coords, resize_coords, patch_size, indices=None, verbose=1, draw_grid=True):
    if indices is None:
        indices = np.arange(len(resize_coords))## 这个长度是50
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
        # print('start stitching {}'.format(patch_dset.attrs['wsi_name']))

    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(idx, total))

        patch_id = indices[idx]
        x_coord,y_coord = source_coords[patch_id]
        sour_coord = (x_coord* 2** args.level, y_coord *2** args.level)
        patch = np.array(slide.read_region(sour_coord, args.level, (256, 256)).convert("RGB"))

        # patch = patch_dset[patch_id]
        patch = cv2.resize(patch, patch_size)
        coord = resize_coords[patch_id]
        canvas_crop_shape = canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3] = patch[:canvas_crop_shape[0],
                                                                                           :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)

def DrawGrid(img, coord, shape, thickness=2, color=(0,0,0,255)):
    cv2.rectangle(img, tuple(np.maximum([0, 0], coord-thickness//2)), tuple(coord - thickness//2 + np.array(shape)), (0, 0, 0, 255), thickness=thickness)
    return img

# '''画出来网格的图像'''
def stitching(grid, downscale=64):
    start = time.time()
    heatmap = StitchPatches(grid, slide,downscale=downscale, bg_color=(0, 0, 0), alpha=-1,
                                draw_grid=False)  # 这个保存patch的图片的时候，里面有一个downscal把文件进行了缩小
    total_time = time.time() - start

    return heatmap, total_time


'''--------------------------------------------------------------'''

if __name__ == '__main__':
    parse = argparse.ArgumentParser("whole slide image preprocessing!")
    parse.add_argument("--slide_path", default=r"/home/omnisky/hdd_15T_sdc/RL_project/dataset/Mr_Zheng2/debug", type=str, help="slide path")
    parse.add_argument("--level", default=3, type=int, help="level of slide")
    parse.add_argument("--patch_size", default=256, type=int, help="patch size")
    parse.add_argument("--step_size", default=256, type=int, help="step")
    parse.add_argument("--save_pth", default=r"/home/omnisky/hdd_15T_sdc/RL_project/dataset/Mr_Zheng2/debug/output", type=str, help="Store the extracted coordinate points in the PTH file")
    # parse.add_argument("--save_patch", default=r"", type=str, help="The path where the patch is stored")
    args = parse.parse_args()
    slide_num, target = 0, 0
    #明确需要保存的信息
    file_name = []
    grids = []
    targets = []


    slides_mel = sorted(os.listdir(os.path.join(args.slide_path, "Tumor")))  # 获取所有肿瘤slide切片的名字
    slides_nev = sorted(os.listdir(os.path.join(args.slide_path, "Normal")))  # Normal
    slides = []  # 'Tumor/60890_0-Tumor-0.ndpi'，'Normal/80308_0-Normal-0.ndpi'
    for i, j in zip(slides_mel, slides_nev):  # 这样就控制了两个数据必须保持一致
        slides.append('Tumor/' + i)  # melanoma
        slides.append('Normal/' + j)  # Normal


    # os.path.isfile()函数判断slide是否存在，先执行if语句，值执行for循环
    slides = [slide for slide in slides if os.path.isfile(os.path.join(args.slide_path, slide))]

    for name in slides:
        disease_name,slide_name =  name.split("/")
    # for disease_name in os.listdir(args.slide_path):
    #     for slide_name in os.listdir(os.path.join(args.slide_path, disease_name)):

        print("-----------------开始处理第{}张{}病理图片！---------------------".format(slide_num, slide_name))
        slide_path = os.path.join(os.path.join(args.slide_path, disease_name), slide_name)
        slide = ol.OpenSlide(slide_path)
        contours_tissue, holes_tissue = seg_slide(slide=slide, seg_level=args.level, patch_size=args.patch_size)
        grid = tiling_slide(slide=slide, contours_tissue=contours_tissue, holes_tissue=holes_tissue,
                                patch_level=args.level,
                                patch_size=args.patch_size, patch_step=args.step_size, name=slide_name)
        if re.search("Tumor", slide_name) is not None:
            target = 1
        if re.search("Normal", slide_name) is not None:
            target = 0
        sn,_ = os.path.splitext(slide_name)
        torch.save({
                "slides": slide_path,
                "grid": grid,
                "targets": target,
                "level": args.level
            }, os.path.join(args.save_pth, "{}.pth".format(sn))) #存储提取点的文件.pth文件，xxx.svs,
        file_name.append(slide_path)
        grids.append(grid)
        targets.append(target)
        slide_num += 1

        #展示提取的组织轮廓
        vis_img = vis_slide(slide, contours_tissue, holes_tissue, vis_level=args.level) #image类型
        mask_path = os.path.join(args.save_pth, "VS"+sn + '.png')
        vis_img.save(mask_path)
        # plt.imshow(vis_img)
        # plt.show()

        #展示patch拼接后的组织图像
        heatmap, stitch_time_elapsed = stitching(grid, downscale=16)  # 第四步：根据提patch的点，把图片进行展示,downscale=64,开始是
        stitch_path = os.path.join(args.save_pth,
                                   "ST"+sn + '.png')
        heatmap.save(stitch_path)




    torch.save({
        "slides": file_name,
        "grid": grids,
        "targets": targets,
        "level": args.level,
        "mult": "1",
    }, os.path.join(args.save_pth, "level_{}.pth".format(args.level)))


   # #'''这个两个都是可视化一张slide提取的轮廓和一张slide对提取的patches'''
   #  vis_img = vis_slide(slide, contours_tissue, holes_tissue, vis_level=args.level)
   #  plt.imshow(vis_img)
   #  plt.show()


   #这段code可以获取patch图片
   #  num = 0
   #  for idx in range(len(grid)):
   #      coord = grid[idx]
   #      coord = (coord[0] ** args.level, coord[1]**args.level)
   #      img = np.array(slide.read_region(coord, args.level, (256,256)).convert("RGB"))
   #      img_bgr = img[:,:,::-1]
   #      cv2.imwrite(os.path.join(r"/home/omnisky/hdd_15T_sdc/RL_project/dataset/Mr_Zheng2/debug/output/1", "{}.jpg".format(num)), img_bgr)#opencv保存图片的格式是BGR，所以对于RGB数据一定要先转换成BGR，否则色调相反
   #      num += 1


   #  #存储裁剪的图片的
   #  heatmap, stitch_time_elapsed = stitching(grid, downscale=16)  # 第四步：根据提patch的点，把图片进行展示,downscale=64,开始是
   #  stitch_path = os.path.join('/home/omnisky/hdd_15T_sdc/RL_project/dataset/Mr_Zheng2/debug/output',
   #                                 str(slide_num) + '.png')
   #  heatmap.save(stitch_path)