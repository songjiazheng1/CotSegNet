import numpy as np

def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape# 获取批量数据的维度信息，B为批次大小，N为点的数量，C为点的坐标维度
    normal_data = np.zeros((B, N, C))# 初始化一个全零数组，用于存储归一化后的点云数据
    for b in range(B):# 遍历每个批次的数据
        pc = batch_data[b]# 获取当前批次的点云数据
        centroid = np.mean(pc, axis=0)# 计算点云的中心点（质心）
        pc = pc - centroid# 将点云数据减去中心点，使得新的点云数据围绕原点
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))# 计算点云数据的最大距离（即从原点到最远点的距离)
        pc = pc / m# 将点云数据除以最大距离，实现归一化
        normal_data[b] = pc# 将归一化后的点云数据存储到结果数组中
    return normal_data


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))# 创建一个与labels长度相同的索引数组
    np.random.shuffle(idx)# 使用numpy的random模块中的shuffle函数打乱索引数组的顺序
    return data[idx, ...], labels[idx], idx

def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])# 创建一个与batch_data第二维度大小相同的数组，元素为0到batch_data.shape[1]-1的整数
    np.random.shuffle(idx)# 使用numpy的random.shuffle函数对idx数组进行随机打乱
    return batch_data[:,idx,:]

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)# 创建一个与输入数据形状相同的全零数组，用于存储旋转后的点云数据
    for k in range(batch_data.shape[0]):# 遍历输入数据的每个点云
        rotation_angle = np.random.uniform() * 2 * np.pi# 随机生成一个旋转角度（范围为0到2π）
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)# 计算旋转角度的余弦值和正弦值
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])# 构建旋转矩阵
        shape_pc = batch_data[k, ...]# 提取当前点云数据
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)# 将点云数据转换为列向量形式，并与旋转矩阵相乘，得到旋转后的点云数据
    return rotated_data# 返回旋转后的点云数据

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)   # 初始化一个与输入数据形状相同的全零数组，用于存储旋转后的点云数据
    for k in range(batch_data.shape[0]):  # 遍历输入数据的每个批次
        rotation_angle = np.random.uniform() * 2 * np.pi# 随机生成一个旋转角度（范围为0到2π）
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)# 计算旋转角度的余弦值和正弦值
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])# 构建绕Z轴旋转的旋转矩阵
        shape_pc = batch_data[k, ...] # 获取当前批次的点云数据
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)# 将点云数据转换为齐次坐标形式，并与旋转矩阵相乘，得到旋转后的点云数据
    return rotated_data

def rotate_point_cloud_with_normal(batch_xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
    '''
    for k in range(batch_xyz_normal.shape[0]):# 遍历输入的点云批次
        rotation_angle = np.random.uniform() * 2 * np.pi# 随机生成一个旋转角度，范围为0到2π
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)# 计算旋转角度的余弦值和正弦值
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])# 构建旋转矩阵
        shape_pc = batch_xyz_normal[k,:,0:3]# 提取当前批次的点云坐标和法向量
        shape_normal = batch_xyz_normal[k,:,3:6]
        batch_xyz_normal[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)# 使用旋转矩阵对点云坐标进行旋转
        batch_xyz_normal[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)# 使用旋转矩阵对法向量进行旋转
    return batch_xyz_normal# 返回旋转后的点云批次

def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx6 array, original batch of point clouds and point normals
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)# 初始化一个与输入数据形状相同的全零数组，用于存储旋转后的数据
    for k in range(batch_data.shape[0]):# 遍历批次中的每个数据点
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)# 生成随机角度，并将其限制在给定的范围内
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]]) # 计算绕x轴、y轴和z轴的旋转矩阵
        R = np.dot(Rz, np.dot(Ry,Rx))# 将三个旋转矩阵相乘得到总的旋转矩阵
        shape_pc = batch_data[k,:,0:3]
        shape_normal = batch_data[k,:,3:6]# 提取当前数据点的点云和法线信息
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), R)# 对点云和法线应用旋转矩阵
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)# 初始化一个与输入数据形状相同的全零数组，用于存储旋转后的点云数据
    for k in range(batch_data.shape[0]):# 遍历输入数据的每个批次
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)# 计算旋转角度的余弦值和正弦值
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])# 构建旋转矩阵
        shape_pc = batch_data[k,:,0:3]# 提取当前批次的点云数据（只取前三个维度，即x, y, z坐标
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)# 将点云数据与旋转矩阵相乘，得到旋转后的点云数据
    return rotated_data

def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx6 array, original batch of point clouds with normal
          scalar, angle of rotation
        Return:
          BxNx6 array, rotated batch of point clouds iwth normal
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)# 初始化一个与输入数据形状相同的全零数组，用于存储旋转后的数据
    for k in range(batch_data.shape[0]):# 遍历批量数据中的每个点云
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)# 计算旋转角度的余弦值和正弦值
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])# 构建旋转矩阵
        shape_pc = batch_data[k,:,0:3]
        shape_normal = batch_data[k,:,3:6]# 提取点云的坐标和法向量
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)# 使用旋转矩阵对点云坐标进行旋转
        rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1,3)), rotation_matrix)# 使用旋转矩阵对法向量进行旋转
    return rotated_data



def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)# 初始化一个与输入数据形状相同的全零数组，用于存储旋转后的数据
    for k in range(batch_data.shape[0]):# 遍历输入数据的每个样本
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)# 生成三个随机角度，范围在-angle_clip到angle_clip之间
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]]) # 计算绕x轴、y轴和z轴的旋转矩阵
        R = np.dot(Rz, np.dot(Ry,Rx))# 将三个旋转矩阵相乘得到总的旋转矩阵
        shape_pc = batch_data[k, ...] # 获取当前样本的点云数据
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)# 将点云数据与旋转矩阵相乘，得到旋转后的点云数据
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape # 获取输入数据的维度信息，B表示批次大小，N表示点的数量，C表示点的坐标维度
    assert(clip > 0)# 确保clip参数大于0，否则抛出异常
    # 生成一个与输入数据形状相同的随机噪声矩阵，并乘以sigma参数进行缩放
    # np.random.randn(B, N, C)生成一个标准正态分布的随机数矩阵
    # np.clip将矩阵中的值限制在[-clip, clip]范围内
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data # 将噪声矩阵加到原始数据上，得到添加了噪声的数据
    return jittered_data # 返回添加了噪声的数据

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
          随机的平移（加或者减）
    """
    B, N, C = batch_data.shape# 获取输入数据的维度信息，B表示批次大小，N表示点的数量，C表示点的坐标维度
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))# 生成一个随机位移矩阵，形状为(B, 3)，每个元素在[-shift_range, shift_range]范围内均匀分布
    for batch_index in range(B):# 遍历每个批次的数据
        batch_data[batch_index,:,:] += shifts[batch_index,:]# 将当前批次的位移应用到所有点上，即将位移向量加到每个点的坐标上
    return batch_data# 返回经过位移处理后的点云数据


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
            各个点的数值大小随机改变[0.8到1.125之间随机扩大或缩小]           
    """
    B, N, C = batch_data.shape# 获取输入数据的维度信息，其中B表示批次大小，N表示点的数量，C表示点的维度
    scales = np.random.uniform(scale_low, scale_high, B)# 在指定的范围内生成随机缩放因子，数量与批次大小相同
    for batch_index in range(B):# 遍历每个批次的数据
        batch_data[batch_index,:,:] *= scales[batch_index]# 将当前批次的点云数据乘以对应的缩放因子
    return batch_data# 返回经过缩放处理后的点云数据

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 对batch中每一个数据选取一部分点来去掉（用第一个点来替代）'''
    """
    DP有什么用，是怎么实现的 ?
    DP指的是在训练时随机丢弃一些输入点(DP means random input dropout during training)，这样的训练方式对于预测低密度点云较为有效(相对于输入点云), 
    即在高密度点云中训练的模型，在低密度点云中进行预测，可以达到和训练集中旗鼓相当的效果。具体来说，人工设置超参数p(论文中p=0.95), 从[0, p]中随机出一个
    值dr(drouout ratio), 对于点云中的每一个点，随机产生一个0 - 1的值, 如果该值小于等于dr则表示该点被丢弃。这里有一个细节，某些点被丢弃之后，每个batch
    中的点的数量就不相同了，为了解决这个问题，所有被丢掉的点使用第一个点代替，这样就维持了每个batch中点的数量相同。
    """
    for b in range(batch_pc.shape[0]):# 遍历batch中的每个数据
        dropout_ratio =  np.random.random()*max_dropout_ratio # 生成一个随机的丢弃比例，范围为0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]# 找到需要丢弃的点的索引
        if len(drop_idx)>0:# 如果有需要丢弃的点
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # 将需要丢弃的点替换为第一个点
    return batch_pc



