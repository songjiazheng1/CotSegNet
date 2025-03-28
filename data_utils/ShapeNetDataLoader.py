# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):#点云归一化处理
    centroid = np.mean(pc, axis=0) # 计算点云的中心点
    pc = pc - centroid#  # 将点云中的每个点减去中心点，使得新的点云以原点为中心
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1))) # 计算点云中距离原点最远的点的欧氏距离
    pc = pc / m# 将点云中的每个点除以最大距离，使得新的点云的最大距离为1
    return pc# 返回归一化后的点云

class PartNormalDataset(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        # 初始化数据集类，设置数据集的根目录、点数、分割方式、类别选择和法线通道选项
        self.npoints = npoints
        self.root = root# 设置数据集的根目录
        self.catfile = os.path.join(self.root, 'class.txt')# 设置类别文件的路径
        self.cat = {}# 创建一个空字典来存储类别信息
        self.normal_channel = normal_channel


        with open(self.catfile, 'r') as f:# 遍历文件的每一行
            for line in f:# 去除行首尾的空白字符，然后按空格分割成列表
                ls = line.strip().split() # 将列表的第一个元素作为键，第二个元素作为值，添加到字典self.cat中
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}# 重新构建字典，确保没有重复的键（这一步可能是多余的，因为字典本身不允许重复键）
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))# 创建一个新字典，将原始类别映射到从0开始的整数索引

        if not class_choice is  None:# 如果提供了特定的类别选择，则更新self.cat以仅包含这些类别
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        #with open(os.path.join(self.root, 'train_test_split', 'train.txt'), 'r') as f:
            #train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'train.txt'), 'r') as fin:# 读取训练集的文件列表，并将其转换为集合形式
            hyper_list = [line[:-5] for line in fin] # 从文件中读取每一行，并去掉每行的换行符和最后5个字符


            train_ids = set([str(hyper_list[i]) for i in range(len(hyper_list))])# 将处理后的列表转换为集合，以去除重复元素
            #print(train_ids)
        with open(os.path.join(self.root, 'train_test_split', 'val.txt'), 'r') as fin:
            hyper_list = [line[:-5] for line in fin]
            val_ids = set([str(hyper_list[i]) for i in range(len(hyper_list))])
        with open(os.path.join(self.root, 'train_test_split', 'test.txt'), 'r') as fin:
            hyper_list = [line[:-5] for line in fin]
            test_ids = set([str(hyper_list[i]) for i in range(len(hyper_list))])



        #with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            #val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        #with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            #test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:# 遍历self.cat字典中的每个元素（类别）
            # print('category', item)
            self.meta[item] = []# 初始化一个空列表，用于存储当前类别下的文件名
            dir_point = os.path.join(self.root, self.cat[item])# 拼接类别对应的文件夹路径
            fns = sorted(os.listdir(dir_point))# 获取并排序该文件夹下的所有文件名
            #print(fns[0][0:-4])
            if split == 'trainval':# 判断是否需要筛选训练集和验证集
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]# 从文件名列表中筛选出属于训练集或验证集的文件名



            elif split == 'train':# 判断是否需要筛选训练集
                fns = [fn for fn in fns if fn[0:-4] in train_ids]# 从文件名列表中筛选出属于训练集的文件名

            elif split == 'val':# 判断是否需要筛选验证集
                fns = [fn for fn in fns if fn[0:-4] in val_ids]# 从文件名列表中筛选出属于验证集的文件名
            elif split == 'test':# 判断是否需要筛选测试集

                fns = [fn for fn in fns if fn[0:-4] in test_ids]# 此处未给出具体操作，可能需要补充代码以处理测试集的情况
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)


            #print(os.path.basename(fns))
            for fn in fns:# 遍历文件列表
                token = (os.path.splitext(os.path.basename(fn))[0]) # 获取文件名的基本部分（不包括扩展名）
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))# 将基本文件名加上'.txt'后缀，并与目录路径拼接，添加到meta字典中对应的item列表中

        self.datapath = []# 初始化一个空列表用于存储数据路径
        for item in self.cat:# 遍历类别字典
            for fn in self.meta[item]:# 遍历每个类别下的文件路径列表
                self.datapath.append((item, fn))# 将类别和文件路径作为元组添加到datapath列表中

        self.classes = {}# 初始化一个空字典用于存储类别映射
        for i in self.cat.keys():# 遍历类别字典的键
            self.classes[i] = self.classes_original[i]# 将原始类别映射到新的类别字典中

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = { 'cotton': [0, 1,2]# 初始化一个字典，用于存储分割类别映射，例如：'cotton' -> [0, 1, 2]
                            }# 初始化一个字典，用于存储分割类别映射，例如：'cotton' -> [0, 1, 2]

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple# 初始化一个空字典用于缓存数据
        self.cache_size = 20000# 设置缓存大小为20000



    def __getitem__(self, index):

        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)



            data = np.loadtxt(fn[1]).astype(np.float32)# 加载文件中的数据并转换为浮点类型
            point_set = data[:, 0:3]# 提取点集和分割信息
            if not self.normal_channel:# 检查是否存在正常通道
                point_set = data[:, 0:3]# 如果不存在正常通道，则从数据中提取前3列作为点集
            else:
                point_set = data[:, 0:6]# 如果存在正常通道，则从数据中提取前6列作为点集
            seg = data[:, -1].astype(np.int32)# 将数据的最后一列转换为整数类型，并赋值给seg变量
            if len(self.cache) < self.cache_size:# 如果缓存未满，则将数据添加到缓存中
                self.cache[index] = (point_set, cls, seg)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)

        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        print(seg)


        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)



