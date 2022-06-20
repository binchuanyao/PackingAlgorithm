# -*- coding: utf-8 -*-
# @File   : miniCube2
# @Time   : 2022/05/18 15:47 
# @Author : BCY

import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from tqdm import tqdm
import random

def minCube(skuList):
    '''
    根据sku列表生成最小cube
    :param skuList: sku列表，sku为元组形式
    :return:
    '''

    # if len(skuList)>100:
    #     return stackHeight(skuList)

    O = (0, 0, 0)  # 初始放置点
    O_items = [O]  # 放置点列表
    used_point = []  # 已用放置点
    placed_sku = []  # 已放sku

    # 初始化最小cube为第一个sku
    cube = skuList[0]   # 最小cube
    # cubeVol = skuList[0][0] * skuList[0][1] * skuList[0][2]
    # 把堆叠后产生的新的点，加入放置点列表
    for new_O in newsite(O_items[0], skuList[0]):
        # 保证放入的可用点是不重复的
        if new_O not in O_items:
            O_items.append(new_O)
    used_point.append(O)
    placed_sku.append(skuList[0])
    O_items = list(filter(lambda x: x not in used_point, O_items))

    # 放置货物的朝向
    face = ['abc','acb','bac','bca','cab', 'cba']

    # 第2到最后一个SKU
    for i in tqdm(range(1, len(skuList))):
        # 初始化cube为三边叠加的最大立方体
        curr_sku = skuList[i]
        choose_point = O_items[0]
        choose_face = 'abc'
        # # 每增加一个货物，初始化当前cube为最大立方体
        curr_cube_vol = (cube[0] + curr_sku[0]) * (cube[1] + curr_sku[1]) * (cube[2] + curr_sku[2])

        # 新增加一个货物，初始化当前cube为最大cube
        curr_cube = (cube[0]+curr_sku[0], cube[1]+curr_sku[1], cube[2]+curr_sku[2])
        # curr_cube = tuple(sorted(curr_cube, reverse=True))   # 按三边长度重排序
        # curr_cube_vol = curr_cube[0] * curr_sku[1] * curr_sku[2]

        '''遍历放置点和当前sku朝向，找到指标最小的 点&朝向'''
        for point in O_items:
            for f in face:
                curr_sku = exchange(skuList[i], f)

                # 判断选择当前放置点会否与其他SKU重合
                isOverlapFlag = 0
                for k in range(len(used_point)):
                    if isOverlap(point, curr_sku, used_point[k], placed_sku[k]):
                        isOverlapFlag += 1

                if isOverlapFlag > 0:  # 如果当前放置点及朝向有重叠，则跳过
                    continue
                else:

                    # print('point: ', point, 'curr_sku: ', curr_sku)
                    c = gene_min_cube(cube, point, curr_sku)   # 返回当前放置方式的最小cube体积

                    '''
                    评价指标
                    1. 体积最小 
                    2. 表面积最小
                    3. 三边和最小
                    4. 最长边最小
                    '''
                    # if v < curr_cube_vol and max(c)<max(curr_cube):     # 体积最小
                    # if c[0] * c[1] + c[0] * c[2] + c[1] * c[2] < curr_cube[0] * curr_cube[1] + curr_cube[0] * curr_cube[2] + curr_cube[1] * curr_cube[2]:  #表面积最小
                    # if c[0] + c[1] + c[2] < curr_cube[0] + curr_cube[1] + curr_cube[2]:   #三边之和最小
                    if max(c) < max(curr_cube) and get_n_large_num(c, 2) < 76 and sum_edge(c) < 266:

                        choose_point = point  # 更新摆放点
                        choose_face = f       # 更新摆放方式
                        # curr_cube_vol = c[0] + c[1] + c[2]     # 更新当前体积
                        curr_cube = c

        curr_placed_sku = exchange(skuList[i], choose_face)
        # print('choose point: ', choose_point, 'curr_sku: ', placed_sku)
        # 添加新的放置点
        for new_O in newsite(choose_point, curr_placed_sku):
            if new_O not in O_items:
                O_items.append(new_O)
        used_point.append(choose_point)
        placed_sku.append(curr_placed_sku)
        cube = gene_min_cube(cube, choose_point, curr_placed_sku)   #当前SKU放入后的CUBE
        O_items = list(filter(lambda x: x not in used_point, O_items))
        O_items = sorted(O_items, key=lambda x: distance(x))  # 按与原点的距离升序排列，优先使用靠近原点的点

    # 最小cube按三边长度重排序
    cube = tuple(sorted(cube, reverse=True))

    # # 高叠加的cube
    # heightCube = stackHeight(skuList)
    #
    # if max(cube) > max(heightCube):
    #     cube = heightCube

    return cube


def minCubeWithWeight(skuList, orderWt, show=False):
    '''
    根据sku列表生成最小cube
    :param skuList: sku列表，sku为元组形式
    :param orderWt: 订单实物重量
    :return:
    '''
    # print('\n当前计算skulist：', skuList)
    O = (0, 0, 0)  # 初始放置点
    O_items = [O]  # 放置点列表
    used_point = []  # 已用放置点
    placed_sku = []  # 已放sku

    show_num = []  # 记录图显信息

    # 初始化最小cube为第一个sku
    cube = skuList[0]   # 最小cube
    # cubeVol = skuList[0][0] * skuList[0][1] * skuList[0][2]
    # 把堆叠后产生的新的点，加入放置点列表
    for new_O in newsite(O_items[0], skuList[0]):
        # 保证放入的可用点是不重复的
        if new_O not in O_items:
            O_items.append(new_O)
    used_point.append(O)
    placed_sku.append(skuList[0])
    show_num.append(make_color(O, skuList[0], 'blue'))
    if show:
        make_pic(show_num, max(cube))
    O_items = list(filter(lambda x: x not in used_point, O_items))

    # 放置货物的朝向
    face = ['abc','acb','bac','bca','cab', 'cba']

    # 第2到最后一个SKU
    for i in range(1, len(skuList)):
        # 初始化cube为三边叠加的最大立方体
        curr_sku = skuList[i]
        choose_point = O_items[0]
        choose_face = 'abc'
        # # 每增加一个货物，初始化当前cube为最大立方体
        curr_cube_vol = (cube[0] + curr_sku[0]) * (cube[1] + curr_sku[1]) * (cube[2] + curr_sku[2])

        # 新增加一个货物，初始化当前cube为最大cube
        curr_cube = (cube[0]+curr_sku[0], cube[1]+curr_sku[1], cube[2]+curr_sku[2])
        # curr_cube = tuple(sorted(curr_cube, reverse=True))   # 按三边长度重排序
        # curr_cube_vol = curr_cube[0] * curr_sku[1] * curr_sku[2]

        '''遍历放置点和当前sku朝向，找到指标最小的 点&朝向'''
        for point in O_items:
            for f in face:
                curr_sku = exchange(skuList[i], f)

                # 判断选择当前放置点会否与其他SKU重合
                isOverlapFlag = 0
                for k in range(len(used_point)):
                    if isOverlap(point, curr_sku, used_point[k], placed_sku[k]):
                        isOverlapFlag += 1

                if isOverlapFlag > 0:  # 如果当前放置点及朝向有重叠，则跳过
                    continue
                else:

                    # print('point: ', point, 'curr_sku: ', curr_sku)
                    c = gene_min_cube(cube, point, curr_sku)   # 返回当前放置方式的最小cube体积

                    '''
                    评价指标
                    1. 体积最小 
                    2. 表面积最小
                    3. 三边和最小
                    4. 最长边最小
                    5. 最长边最小 & L+2*(W+H)最小 & 次长边最小 
                    6. 最长边 & L+2*(W+H) & 次长边 临界值 
                        6.1  L ： 121, 243
                        6.2  W : 76
                        6.3  L+2*(W+H) 266, 330, 419
                    '''
                    # if v < curr_cube_vol and max(c)<max(curr_cube):     # 1. 体积最小
                    # if c[0] * c[1] + c[0] * c[2] + c[1] * c[2] < curr_cube[0] * curr_cube[1] + curr_cube[0] * curr_cube[2] + curr_cube[1] * curr_cube[2]:  #2. 表面积最小
                    # if c[0] + c[1] + c[2] < curr_cube[0] + curr_cube[1] + curr_cube[2]:   # 3. 三边之和最小
                    # if max(c) < max(curr_cube):  # 4 最长边最小

                    if max(c) > 121 or get_n_large_num(c, 2) > 76 or sum_edge(curr_cube) > 266 or c[0] * c[1] * c[2] / 5000 > orderWt:
                        continue
                    elif get_n_large_num(c, 2) < get_n_large_num(curr_cube, 2):  # 评价指标6
                        choose_point = point  # 更新摆放点
                        choose_face = f       # 更新摆放方式
                        # curr_cube_vol = c[0] + c[1] + c[2]     # 更新当前体积
                        curr_cube = c

        curr_placed_sku = exchange(skuList[i], choose_face)
        # print('choose point: ', choose_point, 'curr_sku: ', placed_sku)
        # 添加新的放置点
        for new_O in newsite(choose_point, curr_placed_sku):
            if new_O not in O_items:
                O_items.append(new_O)
        used_point.append(choose_point)
        placed_sku.append(curr_placed_sku)
        show_num.append(make_color(choose_point, curr_placed_sku, 'blue'))
        if show:
            make_pic(show_num, max(cube))
        # print('*'*30)
        cube = gene_min_cube(cube, choose_point, curr_placed_sku)   #当前SKU放入后的CUBE
        O_items = list(filter(lambda x: x not in used_point, O_items))
        O_items = sorted(O_items, key=lambda x: distance(x))  # 按与原点的距离升序排列，优先使用靠近原点的点

    # 增加图象信息中的最小cube
    show_num.append(make_color(O, cube, 'green'))

    # 最小cube按三边长度重排序
    cube = tuple(sorted(cube, reverse=True))

    # 高叠加的cube
    heightCube = stackHeight(skuList)

    # print('used point: ', used_point)
    # print('placed_sku: ', placed_sku)

    if max(cube) > max(heightCube):
        cube = heightCube
        if show:
            show_num = []
            show_num.append(make_color(O, skuList[0], 'blue'))
            point = (0,0,O[2]+skuList[0][2])
            for sku in skuList[1:]:
                show_num.append(make_color(point, sku, 'blue'))
                point = (0,0,point[2]+sku[2])
            show_num.append(make_color(O, cube,'green'))
            make_pic(show_num, max(cube))
    else:
        if show:
            make_pic(show_num, max(cube))

    return cube


def minCubeForSingleSKU(skuList, show=False):
    '''
    单品多件根据sku列表生成最小cube
    :param skuList:sku列表，sku为元组形式
    :return: 最小Cube
    '''

    N = len(skuList)  # 单品件数
    l, w, h = skuList[0][0],skuList[0][1],skuList[0][2]
    permutation = [1, 1, 0, N]

    # 如果高叠加的最长边小于SKU最长边，则高叠加最优
    if h*N <= l:
        return tuple(sorted((l,w,h*N), reverse=True))
    # 高叠加最长边超过SKU长
    else:
        # 初始化cube为高叠加的L，W，H
        cube = [l, w, h*N]

        for num in range(2,N):
            if np.mod(N, num) == 0:
                layer = int(N / num)   # 放的层数
            else:
                layer = int(N/num) + 1
            curr_cube = [0,0, h * layer]
            # 每一层放的sku数量num已定时，初始化平面的长和宽
            groups = crack(num)
            # print('每层个数： ', num, '\t摆放组合： ', groups)
            for group in groups:
                if group[2] == 0:   # 能摆成矩形, 其中group为3个元素的列表，group[0]<=group[1]， group[2]为剩余数量
                    curr_cube[0] = max( l*min(group[0:2]), w*max(group[0:2]))
                    curr_cube[1] = min( l*min(group[0:2]), w*max(group[0:2]))
                else:
                    curr_cube[0] = max(l * min(group[0:2]), w * (max(group[0:2])+1))
                    curr_cube[1] = min(l * min(group[0:2]), w * (max(group[0:2])+1))

                # else:  # 不能摆成矩形, 有剩余的rect
                #     # 先确定能摆成矩形部分的形状——大矩形
                #     tempL = max( l*min(group[0:2]), w*max(group[0:2]))  # 长边
                #     tempW = min( l*min(group[0:2]), w*max(group[0:2]))  # 短边
                #
                #     # 判断多出的rect是否能旋转放置, 如果旋转后表面积更小，则旋转放置
                #     if l*group[2] <= tempL and (tempL+w)*max(tempW,l*group[2]) <  tempL*(tempW + w):
                #         curr_cube[0] = tempL + w
                #         curr_cube[1] = tempW
                #     if l*group[2] <= tempL and tempL + (tempW+w) < (tempL+l) + tempW:
                #         curr_cube[0] = tempL
                #         curr_cube[1] = tempW + w
                #     # 不旋转剩余的rect
                #     elif (tempL+l)*tempW < tempL * (tempW+w):
                #         curr_cube[0] = tempL + l # 多余的rect加在长边
                #         curr_cube[1] = tempW
                #     else:
                #         curr_cube[0] = tempL
                #         curr_cube[1] = tempW + w # 多余的rect加在短边
                # print('每层个数： ', num, '摆放方式： ', group, 'cube: ', curr_cube)
                if max(curr_cube)<max(cube) and get_n_large_num(curr_cube, 2) < 76 and sum_edge(curr_cube)<266:
                # if max(curr_cube) < max(cube) :  # 最长边最小
                    cube = curr_cube.copy()
                    permutation[0:3] = group
                    permutation[3] = layer

    if show:
        print('cube : ', cube, 'permutation:', permutation)
        show_single_sku_plot(skuList[0], N, cube, permutation)

    # 返回最小cube和摆放方式
    # return tuple(sorted(cube, reverse=True)), permutation
    return tuple(sorted(cube, reverse=True))


def minCubeMultiToSingel(skuList, orderWt, show):
    '''
    将多品多件中单SKU件数超过xx件的sku，先按单品码垛，再对比多个垛型，按高叠加计算最终的Cube
    :param skuDict:
    :return:
    '''
    skuDict = dict(Counter(skuList))
    new_skuList = []

    for k, v in skuDict.items():
        if v >= 4:
            single_skuList = [k for i in range(v)]
            new_skuList.append(minCubeForSingleSKU(single_skuList))
        else:
            new_skuList += [k for i in range(v)]

    new_skuList = sorted(new_skuList, reverse=True)
    print('单SKU件数组合后的 new_skuList: ', new_skuList)
    cube = minCubeWithWeight(new_skuList, orderWt, show)
    return cube


def show_single_sku_plot(sku, N, cube, permutation):
    '''
    :param sku: sku尺寸，以tuple形式
    :param sku: sku总件数
    :param cube: 最小cube尺寸，以tuple形式
    :param permutation: sku排布方式 四个元素的列表，p[0]*p[1]+p[2] 为每层放置个数，p[3]为层数
    :return:
    '''
    l = sku[0]
    w = sku[1]
    h = sku[2]
    num_per_layer = permutation[0] * permutation[1] + permutation[2]
    layer = permutation[3]
    O = (0,0,0)

    show_num = []            # 图显信息列表
    first_layer_num = []     # 第一层的图显信息列表
    first_col_num = []       # 长度方向第一行的图显信息列表

    if np.mod(cube[0], l) == 0:
        # 长对长，宽对宽
        for j in range(permutation[0]):
            first_col_num.append(make_color(O, (l,w,h), 'blue'))
            O = (O[0] + l , 0, 0)
        if len(first_col_num) == num_per_layer:
            first_layer_num = first_col_num.copy()
        elif len(first_col_num) < num_per_layer:   # 一排摆不下，另一方向有多排
            for i in range(permutation[1]):
                for item in first_col_num:
                    new_item = item.copy()
                    new_item[1] = new_item[1] + w * i  # 每一层的Y轴坐标累加
                    first_layer_num.append(new_item)
                    if len(first_layer_num) >= num_per_layer:
                        break
    else:
        #货位旋转90度，长宽对调，
        for j in range(permutation[1]):
            first_col_num.append(make_color(O, (w, l, h), 'blue'))
            O = (O[0] + w, 0 , 0)
        if len(first_col_num) == num_per_layer:
            first_layer_num = first_col_num.copy()
        elif len(first_col_num) < num_per_layer:  # 一排摆不下，另一方向有多排
            for i in range(permutation[0]):
                for item in first_col_num:
                    new_item = item.copy()
                    new_item[1] = new_item[1] + l * i  # 每一层的Y轴坐标累加
                    first_layer_num.append(new_item)
                    if len(first_layer_num) == num_per_layer:
                        break

    print('first_layer_num : ', first_layer_num)

    for i in range(0, layer):
        # print('in first for i:  ', i)
        for item in first_layer_num:
            new_item = item.copy()
            new_item[2] = new_item[2] + h * i  # 每一层的高度坐标加上货物高度
            print('in second for new_item:  ', new_item)
            show_num.append(new_item)
            if len(show_num) == N:
                break

    show_num.append(make_color((0, 0, 0), cube, 'green'))
    print('minimal cube is : ', cube)
    make_pic(show_num, max(cube))


def packing3D_improve(skuList, ctnSize, show=False):
    '''
    按朝向和放置点，遍历搜索是否能放入
    :param show_num: 图显信息
    :param ctnSize: 纸箱尺寸
    :param skuList: sku列表
    :return: 满箱率
    '''
    # skuList按体积降序排列
    skuList = sorted(skuList, reverse=True)

    original = (0, 0, 0)  # 初始放置点为 原点（0,0,0)
    show_num = [make_color(original, ctnSize, 'red')]

    ctnVol = ctnSize[0] * ctnSize[1] * ctnSize[2]  # 箱型体积
    skuNum = len(skuList)  # 总件数

    ## 初始放置点
    O_items = [original]
    used_point = []  # 已放置点列表
    placed_sku = []  # 已放入SKU列表
    packedNum = 0  # 已放入sku数量
    packedVol = 0.0  # 已放入sku总体积

    # 初始化最小cube为第一个sku
    cube = skuList[0]

    # 放置货物的朝向
    # face = ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']
    # face = ['abc', 'bac', 'acb']
    color_dict = {'abc': 'blue', 'bac': 'orange', 'acb': 'yellow'}

    for i in range(len(skuList)):
        # 遍历SKU的可选朝向, 每换一个sku, face重新初始化
        face = ['abc', 'bac', 'acb']
        for choose_face in face:
            curr_sku = exchange(skuList[i], choose_face)

            for point in O_items:

                # 判断如果当前放置点及朝向有重叠，则跳过
                isOverlapFlag = 0
                if len(placed_sku) > 0:
                    for j in range(len(used_point)):
                        if isOverlap(point, curr_sku, used_point[j], placed_sku[j]):
                            isOverlapFlag += 1

                if isOverlapFlag > 0:
                    continue
                # 如果放置点放置货物后，三个方向都不会超过箱体限制,则认为可以堆放
                elif point[0] + curr_sku[0] <= ctnSize[0] and point[1] + curr_sku[1] <= ctnSize[1] and point[2] + curr_sku[2] <= ctnSize[2]:
                    # 使用放置点，添加一个图显信息
                    new_show = make_color(point, curr_sku, color_dict[choose_face])

                    if new_show not in show_num:
                        # print('1 new_show: ', new_show)
                        show_num.append(new_show)
                        # make_pic(show_num)
                        used_point.append(point)  # 在已用点列表中增加当前放置点
                        placed_sku.append(curr_sku)  # 在已放sku列表中增加当前sku
                        # 计数加1
                        # print('222 current face: ', choose_face,  '222 current Box: ', curr_sku)

                        packedNum += 1
                        packedVol += curr_sku[0] * curr_sku[1] * curr_sku[2]
                        cube = gene_min_cube(cube, point, curr_sku)

                        ## 把堆叠后产生的新的点，加入放置点列表
                        for new_O in newsite(point, curr_sku):
                            # 保证放入的可用点是不重复的
                            if new_O not in O_items:
                                O_items.append(new_O)

                        # 将已用点从放置点列表中删除
                        O_items = list(filter(lambda x: x not in used_point, O_items))
                        O_items = sorted(O_items, key=lambda x: distance(x))  # 按与原点的距离升序排列，优先使用靠近原点的点

                        # if len(O_items) > 1:
                        #     O_items = self.clear_newsite3(O_items)

                    # 如果当前sku能放入，则跳出选点的循环
                    break

            # 如果已装数量 大于当前列表index+1，表示当前sku已经装入，跳出旋转的循环
            if packedNum >= i + 1:
                break

    if packedNum == skuNum:
        if show:
            make_pic(show_num, max(ctnSize))
        return float(packedVol / ctnVol)
    else:
        if show:
            make_pic(show_num, max(ctnSize))
            print('目标箱型尺寸： ', ctnSize)
            print('订单sku列表： ', skuList)
            print('订单总件数： ', skuNum, '已装件数', packedNum)
            print('=' * 15, '当前订单无法匹配目标箱型!', '=' * 15)
        return 0

    # 只返回最小cube
    # return tuple(sorted(cube, reverse=True))

def isOverlap(aCoord, aSize, bCoord, bSize):
    """
    判断三面投影是否重叠
    按逻辑，如果两个立方体不重叠，在OXY,OXZ,OYZ上最多允许有一个投影面发生重叠（三维问题转二维）
    """
    overlap = 0  # 计算有几个面重叠
    if bCoord[0] >= aCoord[0] + aSize[0] or aCoord[0] >= bCoord[0] + bSize[0] or bCoord[1] >= aCoord[1] + aSize[1] or aCoord[1] >= bCoord[1] + bSize[1]:
        """
        按逻辑，如果两个方形不重叠
        在两个轴方向上，只要有一边超出另一个物体的长度范围即可（二维转一维）
        下同理
        """
        # print("XOY面不重叠")
        pass
    else:
        # print("XOY面重叠")
        overlap = overlap + 1

    if bCoord[0] >= aCoord[0] + aSize[0] or aCoord[0] >= bCoord[0] + bSize[0] or bCoord[2] >= aCoord[2] + aSize[2] or aCoord[2] >= bCoord[2] + bSize[2]:
        # print("XOZ面重叠")
        pass
    else:
        # print("XOZ面重叠")
        overlap = overlap + 1
    if bCoord[1] >= aCoord[1] + aSize[1] or aCoord[1] >= bCoord[1] + bSize[1] or bCoord[2] >= aCoord[2] + aSize[2] or aCoord[2] >= bCoord[2] + bSize[2]:
        # print("YOZ面不重叠")
        pass
    else:
        # print("YOZ面重叠")
        overlap = overlap + 1

    if overlap>1: # 如果这个值大于1则判断两个方体重叠
        return True
    else:
        return False

def crack(integer):
    '''
    将正整数拆解成2数之积+余数 eg 10=2*5+0, 10=3*3+1
    :param integer:
    :return:
    '''
    sqrt_num = int(np.sqrt(integer))
    groups = []
    # for n in range(1, sqrt_num+1):
    for n in range(1,integer):
        factor = int(integer / n)
        remainder = np.mod(integer,n)
        groups.append([n, factor, remainder])
    return groups


#把尺寸数据生成绘图数据
def make(O,C):
    data = [O[0],O[1],O[2],C[0],C[1],C[2]]
    return data

def make_color(O, C, color):
    data = [O[0], O[1], O[2], C[0], C[1], C[2], color]
    return data

# 显示图形的函数：Items = [[num[0],num[1],num[2],num[3],num[4],num[5],num[6]],]
def make_pic(Items, maxSize):
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    # ax.yaxis.set_major_locator(MultipleLocator(10))
    # ax.zaxis.set_major_locator(MultipleLocator(10))

    # ax.xaxis.set_units(10)
    # ax.yaxis.set_units(10)
    # ax.zaxis.set_units(10)

    Xmax = (int(maxSize/10) + 1)* 10
    print('Xmax: ', Xmax)

    ax.set_xlim3d(0, Xmax)
    ax.set_ylim3d(0, Xmax)
    ax.set_zlim3d(0, Xmax)

    # ax.set_xlim3d(0, 260)
    # ax.set_ylim3d(0, 50)
    # ax.set_zlim3d(0, 50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # print 画图的点
    print('---- show num: ')
    print(Items)

    for num in Items:
        box(ax, num[0], num[1], num[2], num[3], num[4], num[5], num[6])
    plt.show()

# make_pic内置函数
def box(ax, x, y, z, dx, dy, dz, color='red'):
    xx = [x, x, x + dx, x + dx, x]
    yy = [y, y + dy, y + dy, y, y]
    kwargs = {'alpha': 1, 'color': color}
    ax.plot3D(xx, yy, [z] * 5, **kwargs)  # 下底
    ax.plot3D(xx, yy, [z + dz] * 5, **kwargs)  # 上底
    ax.plot3D([x, x], [y, y], [z, z + dz], **kwargs)
    ax.plot3D([x, x], [y + dy, y + dy], [z, z + dz], **kwargs)
    ax.plot3D([x + dx, x + dx], [y + dy, y + dy], [z, z + dz], **kwargs)
    ax.plot3D([x + dx, x + dx], [y, y], [z, z + dz], **kwargs)
    return ax

#可用点的生成方法
def newsite(O,B_i):
    # 在X轴方向上生成
    O1 = (O[0]+B_i[0],O[1],O[2])
    # 在Y轴方向上生成
    O2 = (O[0],O[1]+B_i[1],O[2])
    # 在Z轴方向上生成
    O3 = (O[0],O[1],O[2]+B_i[2])
    # return [O1,O2,O3]  # 返回长宽高方向的放置点
    return [O3, O2, O1]  # 高宽长方向的放置点

def distance(x):  # 返回与原点的欧式距离
    return x[0]*x[0], x[1]*x[1], x[2]*x[2]

def isOversize(cube):
    if cube is None:
        return False
    elif cube == 'Null':
        return False
    else:
        return (max(cube)>121 or get_n_large_num(cube, 2)>76 or sum_edge(cube)>266)


# 生成当前最小cube
def gene_min_cube(cube, O, sku):
    current = [round(i + j, 2) for i, j in zip(O, sku)]
    new_cube = tuple([max(i, j) for i, j in zip(cube, current)])
    # new_cube = tuple(sorted(new_cube, reverse=True))
    # vol = new_cube[0] * new_cube[1] * new_cube[2]
    # print('='*30)
    # print('in gene_min_cube new_cube: ', cube, O, sku)
    # print('in gene_min_cube new_cube: ', new_cube)
    return new_cube

def clear_newsite(O_items):
    '''
    清除多余的放置点，即每个轴线及平面上只可能存在一个放置点，第一象限内可存在多个放置点
    :param O_items: 原始放置点
    :return: 清除覆盖点后的放置点
    '''
    # 3个平面上的点
    XOY = [i for i in O_items if i[0] != 0 and i[1] != 0 and i[2] == 0]
    XOZ = [i for i in O_items if i[0] != 0 and i[1] == 0 and i[2] != 0]
    YOZ = [i for i in O_items if i[0] == 0 and i[1] != 0 and i[2] != 0]

    # 3条轴线上的点
    Z_axis= [i for i in O_items if i[0] == 0 and i[1] == 0 and i[2] != 0]
    Y_axis = [i for i in O_items if i[0] == 0 and i[1] != 0 and i[2] == 0]
    X_axis = [i for i in O_items if i[0] != 0 and i[1] == 0 and i[2] == 0]

    # 其他象限中的点
    other = [i for i in O_items if i[0] != 0 and i[1] != 0 and i[2] != 0]

    # 轴线及平面上的点
    current = [XOY, YOZ, XOZ, X_axis, Y_axis, Z_axis]

    new_O_items = []
    for axis in current:
        if len(axis)>0:  # 列表不为空时才取最大值
            new_O_items.append(get_max_point(axis))

    for point in other:
        new_O_items.append(point)

    return new_O_items


def get_max_point(matrix):
    '''
    得到矩阵中每一列最大的值
    '''
    res_list=[]
    for j in range(3):
        one_list=[]
        for i in range(len(matrix)):
            one_list.append(matrix[i][j])
        res_list.append(max(one_list))
    return tuple(res_list)


def get_n_large_num(num_list, n):
    sorted_list = sorted(num_list, reverse=True)
    if n <= len(sorted_list):
        return sorted_list[n - 1]
    else:
        return 0

def sum_edge(cube):
    l = max(cube)
    return 2 * sum(cube) - l

def isConmbine(skuList):
    skuSet = set(skuList)
    if len(skuSet)==1:
        return False
    else:
        skuDict = dict(Counter(skuList))
        re = 0
        for k,v in skuDict.items():
            if v >= 4:
                re += 1
        if re > len(skuSet) / 2:  # 超过一半的SKU件数大于5，则组合sku，否则不组合
            return True
        else:
            return False


def exchange(sku, change): #change='abc','acb','bac','bca','cba','cab' 有6种对调可能，默认为abc
    if change == 'acb':
        new_sku = (sku[0], sku[2], sku[1])
    elif change == 'bac':
        new_sku = (sku[1],sku[0],sku[2])
    elif change == 'bca':
        new_sku = (sku[0], sku[2], sku[1])
    elif change == 'cba':
        new_sku = (sku[2], sku[1], sku[0])
    elif change == 'cab':
        new_sku = (sku[2], sku[0], sku[1])
    else:
        return sku
    return new_sku


def stackHeight(skuList, show=False):
    length = []
    width = []
    H = 0
    for sku in skuList:
        sku = sorted(sku, reverse=True)
        length.append(sku[0])
        width.append(sku[1])
        H += min(sku)
    L = max(length)
    W = max(width)
    H = round(H, 2)

    # 最小cube三边重排序
    cube = (L,W,H)
    # print('in stackHeight: skulist, cube ', skuList, cube)

    if show:
        show_num = []
        O = (0,0,0)
        show_num.append(make_color(O, skuList[0], 'blue'))
        point = (0, 0, O[2] + skuList[0][2])
        for sku in skuList[1:]:
            show_num.append(make_color(point, sku, 'blue'))
            point = (0, 0, point[2] + sku[2])
        show_num.append(make_color(O, cube, 'green'))
        make_pic(show_num, max(cube))

    return cube


#<<<---写一个函数专门用来调整方向和计算剩余货物
def surplus(num, Box_list, change): #change='ab','bc','ac',0有三组对调可能，共6种朝向
    # print('num, Box_list, change: ', num, Box_list, change)

    new_Box_list = Box_list[num-1:-1]
    # print('in surplus new_box_list: ', new_Box_list)
    if num == 0:
        new_Box_list = Box_list
    if change == 'ab':
        for i in range(0,len(new_Box_list)):
            new_Box_list[i]=(new_Box_list[i][1],new_Box_list[i][0],new_Box_list[i][2])
    elif change == 'bc':
        for i in range(0,len(new_Box_list)):
            new_Box_list[i]=(new_Box_list[i][0],new_Box_list[i][2],new_Box_list[i][1])
    elif change == 'ac':
        for i in range(0,len(new_Box_list)):
            new_Box_list[i]=(new_Box_list[i][2],new_Box_list[i][1],new_Box_list[i][0])
    elif change == 0:
        return new_Box_list
    else:
        return new_Box_list
    return new_Box_list

#残余点二次分配函数
# def twice(show_num, packed_result_list, O_pop, C, Box_list, used_point):
#     # print('in twice function： ', show_num,packedNum, O_pop,C,Box_list)
#     for a2 in O_pop:
#         if a2[0]==0 and a2[1]==0:
#             Plan = packing3D(packed_result_list, show_num,a2,C,Box_list, used_point)
#             Box_list = surplus(packed_result_list[0],Box_list,0)
#         elif a2[1]==0 and a2[2]==0:
#             Plan = packing3D(packed_result_list, show_num,a2,C,Box_list, used_point)
#             Box_list = surplus(packed_result_list[0],Box_list,0)
#         elif a2[0]==0 and a2[2]==0:
#             Plan = packing3D(packed_result_list, show_num,a2,C,Box_list, used_point)
#             # print('in twice Plan 3: ', Plan)
#             Box_list = surplus(packed_result_list[0],Box_list,0)
#     return packed_result_list

def gene_skuSize(skuList):
    '''
    将合并的skulist转化为packing能识别的参数 eg: [[(58.0, 17.0, 5.5)], [(16.5, 16.5, 6.0)]] → [(58.0, 17.0, 5.5), (16.5, 16.5, 6.0)]
    :param skuList:
    :return:
    '''
    n = len(skuList)
    if n==1:
        return skuList[0]
    else:
        new_skuList = []
        for item in skuList:
            for sku in item:
                new_skuList.append(sku)

        return new_skuList


def load_data(file_path, file_name, isMulti=False):
    if ".xlsx" in file_name:
        df = pd.read_excel('{}{}'.format(file_path, file_name))
    else:
        try:
            df = pd.read_csv('{}{}'.format(file_path, file_name), encoding='utf-8')
        except:
            df = pd.read_csv('{}{}'.format(file_path, file_name), encoding='gbk')

    shape1 = df.shape

    print('============', '数据导入完成！', '============ ')
    print('原始数据行数： ', shape1[0])
    print('原始订单数： ', df['订单号'].nunique())

    # 剔除sku尺寸异常的订单
    df = df.drop(df[(df['实际长cm'] <= 1) & (df['实际宽cm'] <= 1) & (df['实际高cm'] <= 1)].index)

    shape2 = df.shape

    print('剔除sku尺寸异常数据行数： ', shape1[0]-shape2[0])
    print('剔除异常数据后的订单数： ', df['订单号'].nunique())

    # 合并长宽高，按长度降序排列，格式为元祖
    df['l_w_h'] = df[['实际长cm', '实际宽cm', '实际高cm']].apply(lambda x: tuple(sorted(x, reverse=True)),axis=1)
    df['lineVol'] = df['实际长cm'] * df['实际宽cm'] * df['实际高cm'] * df['产品内件数']
    df['lineWt'] = df['实际重量kg'] * df['产品内件数']

    # 判断单个sku是否超尺寸
    df['skuOversize'] = df['l_w_h'].apply(lambda x: isOversize(x))


    print('原始数据订单字段数据类型')
    print(df.dtypes)

    df['skuSize_temp'] = (df['l_w_h'] * df['产品内件数']).apply(lambda x: [x[i:i + 3] for i in range(0, len(x), 3)])



    # 合并多品订单中SKU尺寸，合并为tuple的列表, 按SKU尺寸降序
    # 按订单号合并skuSize, skuSize_temp列形式为[[(58.0, 17.0, 5.5)], [(16.5, 16.5, 6.0)]]
    order_df= df.groupby('订单号')['skuSize_temp'].apply(lambda x: list(sorted(x, reverse=True))).reset_index()


    order_df['skuSize'] = order_df['skuSize_temp'].apply(gene_skuSize)
    order_df['sku'] = order_df['skuSize'].apply(lambda x: len(set(x)))
    order_df['qty'] = order_df['skuSize'].apply(len)

    # 删除辅助列 skuSize_temp
    order_df = order_df.drop(columns=['skuSize_temp'])


    # 订单中单sku超过尺寸的订单
    skuOversize_df = df.query('skuOversize == True')[['订单号', 'skuOversize']].drop_duplicates()
    order_df = pd.merge(order_df, skuOversize_df, on=['订单号'], how='left')
    order_df['skuOversize'] = order_df['skuOversize'].fillna(False)


    order_df2 = df.groupby('订单号')[['lineVol', 'lineWt']].sum().reset_index()
    order_df2.columns = ['订单号', 'orderVol', 'orderWt']

    order_df = pd.merge(order_df, order_df2, on=['订单号'], how='left')

    if isMulti:
        # 剔除件数单件订单
        order_df = order_df.drop(order_df[order_df['qty'] ==1].index)

    order_df['订单结构'] = '单品单件'
    order_df.loc[(order_df['sku'] == 1) & (order_df['qty'] > 1), ['订单结构']] = '单品多件'
    order_df.loc[(order_df['sku']>1) & (order_df['qty']>1), ['订单结构']] = '多品多件'

    df = pd.merge(order_df[['订单号', 'sku', 'skuSize', 'qty', 'orderVol']], df, on=['订单号'], how='left')
    # df = df.drop(df[df['qty'] ==1].index)
    print(df.columns)

    # 合并多品订单中SKU货型，合并为字符串
    # ‘产品货型’ 为系统按尺寸和重量计算的货型
    # order_size_df = df.groupby('订单号')['产品货型'].apply(lambda x: np.unique(x)).reset_index()
    # order_size_df = df.groupby(['订单号'])['产品货型'].unique().agg('-'.join).reset_index()

    # 'new_sku_size' 为按sku尺寸计算的货型
    # order_size_df2 = df.groupby('订单号')['new_sku_size'].apply(lambda x: np.unique(x)).reset_index()
    #
    # order_size_df = pd.merge(order_size_df, order_size_df2, on=['订单号'])
    if '打单类型' in df.columns:
        order_detail = df[['订单号', '客户代码', '服务渠道名称', '物流产品', '创建时间 仓库当地','打单类型', '是否一票一箱']].drop_duplicates(subset=['订单号'])
    elif 'channel' in df.columns:
        order_detail = df[['订单号', '客户代码', '服务渠道名称', '物流产品', 'channel', '创建时间_仓库当地', '物理仓编码']].drop_duplicates()
    elif '物流产品' in df.columns:
        order_detail = df[['订单号', '客户代码', '服务渠道名称', '物流产品', '创建时间 仓库当地', '物理仓编码', 'Area']].drop_duplicates()
    elif '箱子长' in df.columns:
        order_detail = df[['订单号', '订单类型', '装箱类型', '渠道', '箱子长', '箱子宽', '箱子高', '箱子重量']].drop_duplicates()
    else:
        order_detail = df[['订单号', '客户代码', '创建时间 仓库当地', '物理仓编码', 'Area']].drop_duplicates()


    # temp_df = pd.merge(order_detail, order_size_df, on=['订单号'], how='left')
    order_df = pd.merge(order_df, order_detail, on=['订单号'], how='left').reset_index()

    # 删除多品订单件数为1的行
    # 由于单品单件和单品多件都需要 试装箱，可以不剔除

    print('============', '数据处理完成！', '============ ')
    print('单品多件&多品多件订单数： ', order_df.shape[0])
    print('数据字段：', order_df.columns)

    return order_df


def load_carton(file_path, file_name):
    if ".xlsx" in file_name:
        df = pd.read_excel('{}{}'.format(file_path, file_name))
    else:
        try:
            df = pd.read_csv('{}{}'.format(file_path, file_name), encoding='utf-8')
        except:
            df = pd.read_csv('{}{}'.format(file_path, file_name), encoding='gbk')

    # 将箱子的长，宽，高合并为元组，不改变三边的朝向
    df['箱型尺寸'] = df[['Length', 'Width', 'Height']].apply(lambda x: tuple(x),axis=1)

    ctn_dict = dict(zip(df['Ratio'],df['CartonName']))

    # 返回所有箱型的列表
    return list(df['箱型尺寸']), ctn_dict, df[['CartonName', '箱型尺寸']]



def run_min_cube(order_df, save_path):
    '''
    根据订单计算匹配的最小Cube
    :param order_df: 订单数据的dataframe
    :return:
    '''

    # '''
    # GC现在用的高叠加算法,先算高叠加，高叠加超尺寸的再用其他方式计算
    # '''
    print('=' * 15, '高叠加算法的cube')
    order_df['stackCube'] = order_df['skuSize'].apply(lambda x: tuple(sorted(stackHeight(x), reverse=True)))
    order_df['stackVol'] = order_df['stackCube'].apply(lambda x: x[0] * x[1] * x[2])


    # 高叠加的长宽高
    order_df['stackCube_L'] = order_df['stackCube'].apply(lambda x: x[0])
    order_df['stackCube_W'] = order_df['stackCube'].apply(lambda x: x[1])
    order_df['stackCube_H'] = order_df['stackCube'].apply(lambda x: x[2])

    print('=' * 15, '高叠加是否oversize')
    order_df['stackOversize'] = order_df['stackCube'].apply(lambda x: isOversize(x))

    oversize_df1 = order_df.query('stackOversize == True')
    print('高叠加超尺寸数量： ', oversize_df1.shape[0])

    '''
    匹配订单的最小cube
    '''
    # print('=' * 15, '计算订单的最小cube')
    # order_df['minCube'] = order_df['skuSize'].apply(lambda x: minCube(x[:]))



    ''' 分多品多件 和 单品多件 '''

    print('=' * 15, '计算单品多件的最小CUBE')

    singleSKU_df = order_df.query('订单结构 == "{}"'.format('单品多件'))
    print('单品多件订单数： ', singleSKU_df.shape[0])
    singleSKU_df['minCube'] = singleSKU_df['skuSize'].apply(lambda x: minCubeForSingleSKU(x[:]))

    print('=' * 15, '计算多品多件的最小CUBE')
    multiSKU_df = order_df.query('订单结构 == "{}"'.format('多品多件'))
    print('多品多件订单数： ', multiSKU_df.shape[0])
    multiSKU_df['minCube'] = multiSKU_df['skuSize'].apply(lambda x: minCube(x[:]))


    order_df = pd.concat([singleSKU_df, multiSKU_df])

    print('=' * 15, '三维装箱是否oversize')
    order_df['minCubeOversize'] = order_df['minCube'].apply(lambda x: isOversize(x))

    oversize_df2 = order_df.query('stackOversize == True')
    print('高叠加超尺寸数量： ', oversize_df2.shape[0])


    ''' 不分订单类型 '''
    # print('=' * 15, '考虑渠道计费规则及材积重 计算订单的最小CUBE')
    # order_df['minCube'] = order_df.apply(lambda x: minCubeWithWeight(x['skuSize'], x['orderWt']), axis=1)


    ''' 分订单类型
    ①单品多件先按 单SKU的摆法计算最小cube，
    ②cube长宽高超过限制的，再按多品多件计算，覆盖单品单件摆法的最小cube'''

    # order_df['minCube'] = order_df['stackCube']  # 初始化minCube列
    #
    # print('=' * 15, '计算单品多件的最小CUBE\n')
    # order_df.loc[((order_df['订单结构'] == '单品多件') & (order_df['stackOversize'] == True)), ['minCube']] = order_df['skuSize'].apply(lambda x: minCubeForSingleSKU(x[:]))
    #
    # print('=' * 15, 'testing~~~~~~~~~~~~~')
    # print(order_df.head(30))
    #
    # order_df['isCombine'] = order_df['skuSize'].apply(lambda x: isConmbine(x))
    #
    # print('=' * 15, '同SKU多件组合成一个cube计算最小CUBE\n')
    # order_df.loc[((order_df['stackOversize'] == True) & (order_df['订单结构'] == '多品多件') & (order_df['isCombine'] == True)),
    #              ['minCube']] = order_df.apply(lambda x: minCubeMultiToSingel(x['skuSize'].copy(), x['orderWt']), axis=1 )
    #
    # print('=' * 15, '判断是否超尺寸\n')
    # order_df['minOversize'] = order_df['minCube'].apply(lambda x: isOversize(x))
    #
    #
    # print('=' * 15, '计算多品多件的最小CUBE 及 更新单品多件超尺寸cube\n')
    # order_df.loc[(order_df['minOversize'] == True) , ['minCube']] = order_df.apply(lambda x: minCubeWithWeight(x['skuSize'].copy(), x['orderWt']), axis=1)
    #
    # order_df['minCubeVol'] = order_df['minCube'].apply(lambda x: x[0] * x[1] * x[2])
    #
    # print('=' * 15, '2种算法都超尺寸的，更新回高叠加\n')
    # order_df.loc[((order_df['minOversize'] == True) & (order_df['stackOversize'] == True) & (order_df['minCubeVol'] > order_df['stackVol'])),
    #              ['minCube']] = order_df['stackCube']

    order_df['minCubeVol'] = order_df['minCube'].apply(lambda x: x[0] * x[1] * x[2])  # 更新minCubeVol值
    # 最小cube的长宽高
    order_df['cube_L'] = order_df['minCube'].apply(lambda x: x[0])
    order_df['cube_W'] = order_df['minCube'].apply(lambda x: x[1])
    order_df['cube_H'] = order_df['minCube'].apply(lambda x: x[2])



    # '''比较2种算法选择的体积'''
    # order_df['差异率'] = (order_df['minCubeVol'] - order_df['stackVol']) / order_df['stackVol']
    # order_df['优势算法'] = '高叠加'
    # order_df.loc[(order_df['minCubeVol'] > order_df['stackVol']), ['优势算法']] = 'minCube'

    time = datetime.now()
    str_time = time.strftime('%Y_%m_%d_%H_%M')
    # order_df.to_csv('{}{}'.format(save_path, '多件单最小Cube明细_{}.csv'.format(str_time)), index=False, na_rep='Null')
    order_df.to_csv('{}{}'.format(save_path, 'test_data_results_{}.csv'.format(str_time)), index=False, na_rep='Null')


def run_stack_height(order_df, save_path):
    '''
    根据订单计算匹配的最小Cube
    :param order_df: 订单数据的dataframe
    :return:
    '''

    '''
    GC现在用的高叠加算法,先算高叠加，高叠加超尺寸的再用其他方式计算
    '''
    print('=' * 15, '高叠加算法的cube')
    order_df['stackCube_cm'] = order_df['skuSize'].apply(lambda x: tuple(sorted(stackHeight(x), reverse=True)))
    order_df['stackVol_m3'] = order_df['stackCube_cm'].apply(lambda x: x[0] * x[1] * x[2]/ pow(10,6))

    # 高叠加的cube转换为英寸单位，重量转换为英镑单位
    order_df['stackCube_inch'] = order_df['stackCube_cm'].apply(lambda x: (round(x[0]/2.54,2), round(x[1]/2.54,2), round(x[2]/2.54,2)))
    order_df['orderWt_lb'] = order_df['orderWt'].apply(lambda x : round(x*2.205, 2))

    order_df['isOversize'] = order_df.apply(lambda x: check_oversize(x['channel'], x['stackCube_inch']), axis=1)

    time = datetime.now()
    str_time = time.strftime('%Y_%m_%d_%H_%M')
    order_df.to_csv('{}{}'.format(save_path, '高叠加是否超尺寸判断_{}.csv'.format(str_time)), index=False, na_rep='Null')


def check_oversize(channel, cube):
    if channel in ['UPS', 'Fedex']:
        if 48 < max(cube) <= 96 or get_n_large_num(cube, 2) > 30 or 105 < sum_edge(cube) <= 130:
            return 'AHS'
        elif 96 < max(cube) <= 108 or 130 < sum_edge(cube) <= 165:
            return 'OS'
        elif max(cube) > 108 or sum_edge(cube) > 165:
            return 'Limit'
        else:
            return 'Normal'

    elif channel in ['UPS_SUREPOST', 'JS_SUREPOST']:
        if 48 < max(cube) <= 96 or get_n_large_num(cube, 2) > 30:
            return 'AHS'
        elif 34 < cube[0] <= 48 or 34 < cube[1] <= 48 or 34 < cube[2] <= 48 or (17 < cube[0] <= 30 and 17 < cube[1] <= 30):
            return 'Others'
        else:
            return 'Normal'

    elif channel in ['DHL']:
        if max(cube) > 27 or 50 < sum_edge(cube) <= 84 or cube[0]*cube[1]*cube[2] > 2*1728 :
            return 'Others'
        elif sum_edge(cube) > 84:
            return 'Limit'
        else:
            return 'Normal'
    elif channel in ['DHL_INT_EXPRESS']:
        if max(cube) > 48:
            return 'OS'
        else:
            return 'Normal'
    elif channel in ['IPA', 'E_Packet']:
        if max(cube) > 24 or cube[1] < 6 or cube[2] < 4 or sum(cube) > 36:
            return 'Others'
        else:
            return 'Normal'

    elif channel in ['PRIORITY_PLUS']:
        if max(cube) > 31 or sum(cube)>54:
            return 'Others'
        else:
            return 'Normal'
    elif channel in ['OnTrac_Ground']:
        if 48 < max(cube) <= 96 or get_n_large_num(cube, 2) > 30:
            return 'AHS'
        elif 96 < max(cube) <= 96 or 130 < sum_edge(cube) <= 165:
            return 'OS'
        elif max(cube) > 108 or sum_edge(cube) > 165:
            return 'Limit'
        else:
            return 'Normal'
    elif channel in ['PASSPORT_DDU']:
        if max(cube) > 59 or sum(cube)>118:
            return 'Limit'
        elif max(cube) > 48:
            return 'Others'
        else:
            return 'Normal'
    elif channel in ['PASSPORT_DDP']:
        if max(cube) > 24 or sum(cube)>36:
            return 'Limit'
        elif max(cube) > 48:
            return 'Others'
        else:
            return 'Normal'
    else:
        return 'Normal'


def test_single_efficiency(n, sku=None):
    if sku is None:
        sku = (20.5, 19.0, 12.5)
    sku_list = [sku for i in range(n)]

    startTime1 = datetime.now()
    cube = minCubeForSingleSKU(sku_list)
    endTime1 = datetime.now()
    print('单品多件单：件数 {}, 运行时间 {} ms'.format(n, (endTime1 - startTime1).seconds * 1000 + (endTime1 - startTime1).microseconds / 1000))


def test_multi_efficiency(n):
    sku_list = []
    while len(sku_list)<n:
        sku = [random.randint(1,30) for i in range(3)]
        sku = sorted(sku, reverse=True)
        # print(sku)
        t = tuple(sku)
        if t not in sku_list:
            sku_list.append(tuple(t))

    sku_list = sorted(sku_list, reverse=True)
    # print('n : ', n, 'sku_list: ', sku_list)

    startTime1 = datetime.now()
    cube = minCube(sku_list)
    endTime1 = datetime.now()
    print('多品多件单：sku数 {}, 运行时间 {} ms'.format(n, (endTime1 - startTime1).seconds * 1000 + (endTime1 - startTime1).microseconds / 1000))




if __name__ == '__main__':


    print('\n')
    startTime = datetime.now()
    print('-' * 20 + '程序开始' + '-' * 20 + '\n')
    #
    # # 文件路径
    # file_path = 'D:/Documents/Desktop/箱型推荐/'
    #
    # # file_name = 'multiQty_3.csv'
    # # file_name = '5月Week1筛选渠道_剔除FBA_已签出订单_UPS_GROUND.csv'
    # file_name = 'test_data.csv'
    #
    # order_df = load_data(file_path, file_name)
    #
    # # 计算订单匹配的最小cube
    # run_min_cube(order_df=order_df, save_path=file_path)
    #
    # # # 计算高叠加是否超尺寸
    # # run_stack_height(order_df=order_df, save_path=file_path)
    #
    # print('\n')
    # print('=' * 15, '订单数据预览', '=' * 15)
    # print(order_df.head(10))
    #



    # 单品多件效率测算
    # N = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]
    #
    # for i in N:
    #     test_single_efficiency(i)

    # 多品多件效率测算
    # N = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]
    # for i in N:
    #     test_multi_efficiency(i)



    print('-' * 20 + '程序运行完成！' + '-' * 20 + '\n')
    endTime = datetime.now()
    print('-' * 50)
    print('程序运行总时间：', (endTime - startTime).microseconds, ' ms')












