import numpy as np
import warnings
import os
from torch.utils.data import Dataset#从torch.utils.data模块中导入Dataset类，用于创建自定义数据集
warnings.filterwarnings('ignore')



def pc_normalize(pc):#pc_normalize函数的作用是对输入的点云数据进行归一化处理。具体来说，它会计算点云的中心坐标，然后将每个点的坐标减去中心坐标，使得新的点云数据的中心位于原点。
    centroid = np.mean(pc, axis=0)# 计算点云的中心点
    pc = pc - centroid# 将点云减去中心点，得到新的点云
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))# 计算点云中每个点到原点的最大距离
    pc = pc / m# 将点云除以最大距离，得到归一化后的点云
    return pc

def farthest_point_sample(point, npoint):#最远距离下采样
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape# 获取点云的形状
    xyz = point[:,:3]# 提取前3列数据作为xyz坐标
    centroids = np.zeros((npoint,))# 初始化一个形状为(npoint,)的全零数组作为采样后的点云索引
    distance = np.ones((N,)) * 1e10# 初始化一个形状为(N,)的全1数组，乘以1e10作为初始距离
    farthest = np.random.randint(0, N)# 随机生成一个范围在0到N之间的整数作为最远点的索引
    for i in range(npoint):
        centroids[i] = farthest#将当前最远点的索引farthest存入centroids数组的第i个位置。
        centroid = xyz[farthest, :]#获取当前最远点的坐标centroid
        dist = np.sum((xyz - centroid) ** 2, -1)#计算点云中每个点到当前最远点的距离dist
        mask = dist < distance#创建一个布尔类型的掩码数组mask，表示距离小于distance的元素
        distance[mask] = dist[mask]#更新距离数组distance，将小于当前距离的值替换为当前距离
        farthest = np.argmax(distance, -1)#找到距离数组中最大值对应的索引，作为新的最远点索引farthest
    point = point[centroids.astype(np.int32)]#根据centroids数组中的索引，从原始点云数据中提取对应的点，作为采样后的点云数据point
    return point

class ModelNetDataLoader(Dataset): #指定好到哪里读数据
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root## 数据根目录
        self.npoints = npoint## 每个样本点的数量
        self.uniform = uniform# 是否使用均匀采样
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')# 类别文件路径

        self.cat = [line.rstrip() for line in open(self.catfile)]# 读取类别文件，获取类别列表
        self.classes = dict(zip(self.cat, range(len(self.cat))))# 将类别与对应的索引组成字典
        self.normal_channel = normal_channel## # 是否使用法向量通道
       #指定训练和测试数据的路径
        shape_ids = {}# 存储训练和测试数据的路径
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]#读取训练数据路径
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]#读取测试数据路径

        assert (split == 'train' or split == 'test')#确保split变量的值为'train'或'test'，否则程序会抛出异常
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]#获取shape_names列表，将shape_ids[split]中的每个元素按照'_'分割，取前n-1个部分重新组合成新的字符串
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]#构建self.datapath列表，包含(shape_name, shape_txt_file_path)元组，其中shape_name为shape_names中的元素，shape_txt_file_path为对应的文本文件路径
        print('The size of %s data is %d'%(split,len(self.datapath)))#打印训练集或测试集的大小

        self.cache_size = cache_size  # how many data points to cache in memory设置缓存大小为cache_size
        self.cache = {}  # from index to (point_set, cls) tuple初始化一个空字典self.cache，用于存储索引到(point_set, cls)元组的映射

    def __len__(self):
        return len(self.datapath)#定义一个名为__len__的方法，返回datapath的长度

    def _get_item(self, index): #取数据，index表示要处理的数据，循环batchsize次，这里是8
        if index in self.cache:  #设置缓存
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32) #得到标签
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32) #得到点的具体信息
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints) #最远点采样
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])#相当于3列值，做标准化

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls) #把点的信息放到缓存当中

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)




if __name__ == '__main__':#判断当前脚本是否作为主程序运行，如果是，则执行以下代码
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/',split='train', uniform=False, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    #创建一个ModelNetDataLoader对象，用于加载ModelNet40数据集的训练数据。其中，/data/modelnet40_normal_resampled/是数据集的路径，split='train'表示加载训练数据，uniform=False表示不使用均匀采样，normal_channel=True表示使用法向量通道。
    for point,label in DataLoader:#遍历数据加载器中的每个批次
        print(point.shape)#打印当前批次中点集的形状
        print(label.shape)#打印当前批次中标签的形状