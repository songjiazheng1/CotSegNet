import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from block.SEAttention import SEAttention
from block.metaformer import MetaFormerBlock,Pooling,LayerNormGeneral,partial,MetaFormerCGLUBlock,SepConv
from models.pointnet import PointNetEncoder, feature_transform_reguliarzer
from block.SENET_ATTENTION import SegNext_Attention

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):  #用最远点采样方法得到比较均匀的点
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]  #返回哪个点是中心点
    """
    device = xyz.device
    B, N, C = xyz.shape  #8,1024,3
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  #初始化8*512的矩阵，一共有8个batch，每个batch里有512个点，最终要返回的
    distance = torch.ones(B, N).to(device) * 1e10 #定义一个8*1024的距离矩阵，里面存储的是除了中心点的每个点距离当前所有已采样点的最小距离
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)#batch里每个样本随机初始化一个最远点的索引（第一个点是随机选择的）
    batch_indices = torch.arange(B, dtype=torch.long).to(device) #batch的索引，0 1 2 3 4 5 6 7
    for i in range(npoint):
        centroids[:, i] = farthest #第一个采样点选随机初始化的索引
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)#得到当前采样点的坐标 B*3，后续要算距离
        dist = torch.sum((xyz - centroid) ** 2, -1)#计算当前采样点centroid与其他点的距离
        mask = dist < distance#选择距离最近的来更新距离（更新维护这个表），小于当前距离就是TRUE，否则是FALSE
        distance[mask] = dist[mask]#更新距离矩阵
        farthest = torch.max(distance, -1)[1]#重新计算得到最远点索引（在更新后的距离矩阵中选择距离最大的那个点）
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region #每个组里的点的个数
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample] S：中心点的个数，
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1]) #让每一个圆圈里的点的个数一致
    sqrdists = square_distance(new_xyz, xyz)#得到B*N*M（中心点）的矩阵 （就是N个点中每一个和M中每一个的欧氏距离）  N=1024 M=512
    group_idx[sqrdists > radius ** 2] = N #找到距离大于给定半径的设置成一个N值（1024）索引，值为1024表示这个点不在半径当中
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]#做升序排序，只取规定的圆圈里的个数就行了。后面的都是大的值（1024）  可能有很多点到中心点的距离都小于半径，而我们只需要16个，所以排序一下，取前16个离中心点最近的点
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])#如果半径内的点没那么多，则复制离中心点最近的那个点即第一个点来代替值为1024的点
    mask = group_idx == N #判断是否有值=1024，返回TRUE或FALSE（若点在圆圈里则值为FALSE，否则值为TRUE）
    group_idx[mask] = group_first[mask]  #如果有值为1024，则把第一个值赋值给距离=1024的点
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        #print(xyz.shape)  #第三次：[8,128,3]
        #print(points)
        if points is not None:
            points = points.permute(0, 2, 1)
        #print(points.shape) #第三次：[8,128,640],640个特征
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]，D是3个位置特征
        #print(new_points.shape) #[8,1,128,643]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        #print(new_points.shape) #[8,643,128,1]
        for i, conv in enumerate(self.mlp_convs): #提取特征
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        #print(new_points.shape) #[8,1024,128,1]
        new_points = torch.max(new_points, 2)[0]
        #print(new_points.shape) #[8,1024,1]
        new_xyz = new_xyz.permute(0, 2, 1)
        #print(new_xyz.shape) #[8,3,1]
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]，N=1024
            points: input points data, [B, D, N]，原始的特征信息，3个法向量，D=3,N=1024
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]，不同的半径提取不同的特征，最后将所有特征连接起来，个数就不是3个了，D'是特征个数
        """
        xyz = xyz.permute(0, 2, 1)  #就是坐标点位置特征
        #print(xyz.shape) #第一次：[8,1024，3]，第二次：[8,512，3]
        if points is not None:
            points = points.permute(0, 2, 1)  #就是额外提取的特征，第一次的时候就是那个法向量特征
        #print(points.shape) #第二次：[8,512，320]
        B, N, C = xyz.shape
        S = self.npoint  #S为我们选择的中心点的个数，为了使得选择的点均匀分布，用最远点采样的方法
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))  #最远点采样方法得到的是点的索引值，我们想得到点的实际值，通过index_points()就能得到采样后的点的实际信息
        #print(new_xyz.shape) #第一次：[8,512,3])，第二次：[8,128，3]
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i] #圆圈里圈出来的点的个数
            group_idx = query_ball_point(radius, K, xyz, new_xyz) #返回的是索引  new_xyz是中心点，xyz是原始点  最后得到512个组
            grouped_xyz = index_points(xyz, group_idx) #通过索引得到各个组中实际点
            grouped_xyz -= new_xyz.view(B, S, 1, C) #去均值操作  每个组中的点减去中心点的值 （new_xyz相当于簇的中心点）
            if points is not None:
                grouped_points = index_points(points, group_idx) #法向量特征
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1) #把位置特征和法向量特征拼在一起
                #print(grouped_points.shape)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # 维度转换操作，将[B,S,K,D]转换成[B, D, K, S]
            #print(grouped_points.shape)
            for j in range(len(self.conv_blocks[i])): #卷积核大小1*1，步长=1，通道数6->32->64,进行3次卷积之后，每个点特征为64个
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            #print(grouped_points.shape)
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S] 就是pointnet里的maxpool操作，即在每一个特征维度上，从一个组中选一个值最大的出来，作为这个维度上的特征值。
            #print(new_points.shape) #[8,64,512]（第一次卷积）
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1) #r=0.1,channel=64;r=0.2,channel=128;r=0.4,channel=128,把所有的channel都连接起来。
        #print(new_points_concat.shape) #第一次：[8,320,512]，第二次：[8,640,128]
        return new_xyz, new_points_concat

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1):  # c1: 输入通道数；c2: 输出通道数；kernel: 卷积核大小，默认为1
        super(Conv, self).__init__()# 调用父类的构造函数
        self.conv = nn.Conv2d(c1, c2, k) # 定义一个二维卷积层，输入通道数为c1，输出通道数为c2，卷积核大小为k
        self.bn = nn.BatchNorm2d(c2)# 定义一个二维批量归一化层，对卷积层的输出进行归一化处理，输出通道数为c2
        self.act = nn.ReLU()# 定义一个ReLU激活函数

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))# 前向传播过程：先进行卷积操作，然后进行批量归一化，最后通过ReLU激活函数

    def fuseforward(self, x):
        return self.act(self.conv(x))# 融合前向传播过程：只进行卷积操作和ReLU激活函数，不进行批量归一化



class PointNetSetAbstractionAttention(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstractionAttention, self).__init__()# 继承PyTorch的nn.Module类
        self.npoint = npoint# 设置采样点的数量
        self.radius = radius# 设置半径，用于确定邻域范围
        self.nsample = nsample# 设置每个点的邻居样本数量
        #self.mlp_convs = nn.ModuleList()# 创建一个空的模块列表，用于存储多层感知机（MLP）的卷积层
        self.mlp_conv1 = Conv(in_channel,mlp[0],1)# 创建第一个卷积层，输入通道数为in_channel，输出通道数为mlp[0]，卷积核大小为1
        self.mlp_attention = MetaFormerCGLUBlock(dim=mlp[0], token_mixer=Pooling,norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3),eps=1e-7, bias=False))#——————————————————单个：0.8547
        self.mlp_conv2 = Conv(mlp[0],mlp[1],1)# 创建一个卷积层对象，输入通道数为mlp[0]，输出通道数为mlp[1]，卷积核大小为1x1
        self.mlp_conv3 = Conv(mlp[1],mlp[2],1)# 创建另一个卷积层对象，输入通道数为mlp[1]，输出通道数为mlp[2]，卷积核大小为1x1
        self.group_all = group_all# 设置一个布尔值group_all，用于决定是否将所有输入分组

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)# 将输入的xyz张量进行维度置换，使其形状变为[B, C, N]
        if points is not None:# 如果points不为空，则对其进行维度置换，使其形状变为[B, C, N]
            points = points.permute(0, 2, 1)

        if self.group_all:# 根据group_all属性的值，选择不同的采样和分组方法
            new_xyz, new_points = sample_and_group_all(xyz, points)# 对所有点进行采样和分组
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)# 根据给定的参数进行采样和分组
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)# 对new_points进行维度置换，使其形状变为[B, C+D, nsample, npoint]
        # 通过多层感知机（MLP）卷积层进行处理
        new_points=self.mlp_conv1(new_points)
        new_points = self.mlp_attention(new_points)
        new_points = self.mlp_conv2(new_points)
        #new_points = self.mlp_attention1(new_points)
        new_points = self.mlp_conv3(new_points)
        """  
        self.mlp_convs = nn.ModuleList(): 创建一个空的模块列表，用于存储多个卷积层。
        self.mlp_bns = nn.ModuleList(): 创建一个空的模块列表，用于存储多个批量归一化层。
        for i, conv in enumerate(self.mlp_convs):: 遍历self.mlp_convs中的每个卷积层。
        bn = self.mlp_bns[i]: 获取与当前卷积层对应的批量归一化层。
        new_points = F.relu(bn(conv(new_points))): 将输入数据new_points通过卷积层conv进行卷积操作，然后通过批量归一化层bn进行归一化处理，最后应用ReLU激活函数
        """
        new_points = torch.max(new_points, 2)[0]# 在最后一个维度上取最大值，得到最终的new_points
        new_xyz = new_xyz.permute(0, 2, 1)# 将new_xyz进行维度置换，使其形状变为[B, N, C]
        return new_xyz, new_points# 返回处理后的new_xyz和new_points



##PointNetSetAbstractionKAN1将KAN添加到两个卷积层中间
class PointNetSetAbstractionKAN1(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstractionKAN1, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlp_conv1 = Conv(in_channel, mlp[0], 1)
        self.mlp_attention = KAN([mlp[0],mlp[0]])  # Adjusted input size for KAN
        self.mlp_conv2 = Conv(mlp[0], mlp[1], 1)
        self.mlp_conv3 = Conv(mlp[1], mlp[2], 1)
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]

        new_points = self.mlp_conv1(new_points)

        batch_size = new_points.size(0)
        num_features = new_points.size(1)
        new_points = new_points.view(batch_size, num_features, -1).permute(0, 2, 1)  # Flatten the tensor for KAN

        new_points = self.mlp_attention(new_points)


        new_points = new_points.permute(0, 2, 1).view(batch_size, num_features, self.nsample,
                                                      self.npoint)  # Reshape back to 4D tensor for Conv2
        new_points = self.mlp_conv2(new_points)
        new_points = self.mlp_conv3(new_points)

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

##PointNetSetAbstractionKAN2将KAN完全替换MLP层
class PointNetSetAbstractionKAN2(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstractionKAN2, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlp_attention = KAN([in_channel, mlp[0], mlp[1], mlp[2]])

        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]

        batch_size = new_points.size(0)
        num_features = new_points.size(1)
        new_points = new_points.reshape(batch_size, num_features, -1).permute(0, 2, 1)  # Flatten the tensor for KAN

        new_points = self.mlp_attention(new_points)

        new_points = new_points.permute(0, 2, 1).reshape(batch_size, -1, self.nsample, self.npoint)  # Reshape back to 4D tensor
        #new_points = new_points.permute(0, 3, 2, 1)
        new_points = torch.max(new_points, 2)[0]  # [B, D', S]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points



#2024-10-17 10:42:56,831 - Model - INFO - Best inctance avg mIOU is: 0.82111
class PointNetSetAbstractionMsgAttention(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgAttention, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list# 设置半径列表，用于不同层级的局部区域搜索
        self.nsample_list = nsample_list# 设置每个层级采样点的数量
        self.mlp_conv00 = Conv(in_channel+3,mlp_list[0][0],1)# 定义第一层卷积层，输入通道数为in_channel+3（包括点的坐标和特征），输出通道数为mlp_list[0][0]

        self.mlp_attention0 = SegNext_Attention(mlp_list[0][0])#——————————————————————————————0.880.0.872


        self.mlp_conv01 = Conv(mlp_list[0][0],mlp_list[0][1],1)# 定义第二层卷积层，输入通道数为mlp_list[0][0]，输出通道数为mlp_list[0][1]
        self.mlp_conv02 = Conv(mlp_list[0][1],mlp_list[0][2],1)# Dual定义第三层卷积层，输入通道数为mlp_list[0][1]，输出通道数为mlp_list[0][2]
        self.mlp_conv10 = Conv(in_channel+3,mlp_list[1][0],1)# 定义第四层卷积层，输入通道数为in_channel+3，输出通道数为mlp_list[1][0]
        self.mlp_conv11 = Conv(mlp_list[1][0],mlp_list[1][1],1)# 定义第五层卷积层，输入通道数为mlp_list[1][0]，输出通道数为mlp_list[1][1]
        self.mlp_conv12 = Conv(mlp_list[1][1],mlp_list[1][2],1)#Dual 定义第六层卷积层，输入通道数为mlp_list[1][1]，输出通道数为mlp_list[1][2]

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)# 将输入的xyz张量进行维度置换，使其形状变为[B, C, N]
        if points is not None:# 如果points不为空，则对其进行维度置换
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape# 获取xyz的形状信息
        S = self.npoint# 设置采样点的数量
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))# 对xyz进行最远点采样，得到新的采样点集合new_xyz
        new_points_list = []# 初始化一个新的点列表
        for i, radius in enumerate(self.radius_list):# 遍历半径列表
            K = self.nsample_list[i]# 获取当前半径下的采样点数量
            group_idx = query_ball_point(radius, K, xyz, new_xyz)# 根据半径和采样点数量查询球内的点索引
            grouped_xyz = index_points(xyz, group_idx) # 根据索引获取对应的点集合
            grouped_xyz -= new_xyz.view(B, S, 1, C)# 减去中心点的坐标，得到相对坐标
            if points is not None:# 如果points不为空，则根据索引获取对应的点集合，并与相对坐标拼接
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:# 如果points为空，则直接使用相对坐标作为点集合
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)# 将点集合的维度置换为[B, D, K, S]
            if i==0:# 如果是第一个半径，则使用mlp_conv00、mlp_attention0、mlp_conv01和mlp_conv02进行处理
               grouped_points =self.mlp_conv00(grouped_points)
               grouped_points =self.mlp_attention0(grouped_points)  #注意力机制
               grouped_points = self.mlp_conv01(grouped_points)
               grouped_points = self.mlp_conv02(grouped_points)
            else:# 如果不是第一个半径，则使用mlp_conv10、mlp_attention1、mlp_conv11和mlp_conv12进行处理
                grouped_points = self.mlp_conv10(grouped_points)
                #grouped_points = self.mlp_attention1(grouped_points)  #注意力机制
                grouped_points = self.mlp_conv11(grouped_points)
                grouped_points = self.mlp_conv12(grouped_points)
            # for j in range(len(self.conv_blocks[i])):
            #     conv = self.conv_blocks[i][j]
            #     bn = self.bn_blocks[i][j]
            #     grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]# 对处理后的点集合进行最大池化操作，得到新的点集合new_points
            new_points_list.append(new_points)# 将新的点集合添加到列表中

        new_xyz = new_xyz.permute(0, 2, 1)# 将新的采样点集合的维度置换回[B, N, C]
        new_points_concat = torch.cat(new_points_list, dim=1)# 将所有新的点集合拼接在一起
        return new_xyz, new_points_concat# 将所有新的点集合拼接在一起



class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        #print(xyz1.shape)  #[4,128,3]  [4,512,3]  [4,2048,3]
        #print(xyz2.shape)  #[4,1,3]    [4,128,3]  [4,512,3]

        points2 = points2.permute(0, 2, 1)
        #print(points2.shape) #[4,1,1024]  [4,128,256]   [4,512,128]
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape   #S是采样点个数

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)  #复制128次那一个点
            #print(interpolated_points.shape) #[4,128,1024]  128个点，每个点的特征是1024
        #128个点变成512个，根据距离去插值
        else:
            dists = square_distance(xyz1, xyz2) #计算xyz1与xyz2的欧氏距离，得到距离矩阵
            #print(dists.shape) #[4,512,128] 512*128的距离矩阵，每一行是前一层中的所有点，值为这个点到每个中心点的距离值  [4,2048,512]
            dists, idx = dists.sort(dim=-1) #对距离进行从小到大排序，距离越近影响越大
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]，取前三个距离值和点的索引值
            #计算权重
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            #print(weight.shape) #[4,512,3]  [4,2048,3]
            #print(index_points(points2, idx).shape) #[4,512,3,256]  [4,2048,3,128]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)  #权重*对应点的特征作为插入的点的特征值
            #print(interpolated_points.shape) #[4,512,256] [4,2048,128]

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1) #把采样之后的128个点的1024个特征和采样之前的128个点的512个特征连接起来
        else:
            new_points = interpolated_points
        #print(new_points.shape) #[4,128,1536]  [4,512,576]（320+256）  [4,2048,150]
        new_points = new_points.permute(0, 2, 1)
        #print(new_points.shape) #[4,1536,128]  [4,576,512]  [4,150,2048]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        #print(new_points.shape) #[4,256,128]  [4,128,512]  [4,128,2048]
        return new_points

