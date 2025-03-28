# import argparse
# import os
# from data_utils.ShapeNetDataLoader import PartNormalDataset
# import torch
# import datetime
# import logging
# from pathlib import Path
# import sys
# import importlib
# import shutil
# from tqdm import tqdm
# import provider
# import numpy as np

import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from block.lion_pytorch.lion_pytorch import Lion
from data_utils.ShapeNetDataLoader import PartNormalDataset
"""
训练所需设置参数：
--model pointnet2_part_seg_msg 
--normal 
--log_dir pointnet2_part_seg_msg
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

#各个物体部件的编号
#seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_classes = {'cotton': [0,1,2]}# 定义一个字典，键为类别名称，值为该类别对应的标签列表
seg_label_to_cat = {0:'cotton',1:'cotton',2:'cotton'} # {0:Airplane, 1:Airplane, ...49:Table}# 定义一个字典，键为标签，值为对应的类别名称
for cat in seg_classes.keys():# 遍历 seg_classes 字典的所有键（类别名称
    for label in seg_classes[cat]:# 遍历当前类别对应的标签列表
        seg_label_to_cat[label] = cat# 将标签与类别名称关联起来

def to_categorical(y, num_classes):
    # 使用torch.eye创建一个单位矩阵，大小为num_classes x num_classes
    # 其中索引y对应的行被设置为1，其余行为0
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):# 检查输入张量y是否在GPU上
        return new_y.cuda()# 如果y在GPU上，将new_y转移到GPU上并返回
    return new_y# 如果y不在GPU上，直接返回new_y

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg', help='model name [default: pointnet2_part_seg_msg]')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=200, type=int, help='Epo  ch to run [default: 251]')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_part_seg_msg', help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--step_size', type=int,  default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.5, help='Decay rate for lr decay [default: 0.5]')

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        #print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('part_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TRAIN_DATASET = PartNormalDataset(root = root, npoints=args.npoint, split='train', normal_channel=args.normal)# 创建训练数据集对象，传入参数包括数据集根目录、点数、分割方式和法线通道
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,shuffle=True, num_workers=0)# 创建训练数据加载器，传入参数包括训练数据集、批量大小、是否打乱顺序和工作进程数
    TEST_DATASET = PartNormalDataset(root = root, npoints=args.npoint, split='val', normal_channel=args.normal)# 创建测试数据集对象，传入参数包括数据集根目录、点数、分割方式和法线通道
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=0)# 创建测试数据加载器，传入参数包括测试数据集、批量大小、是否打乱顺序和工作进程数
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))# 输出训练数据的个数
    log_string("The number of test data is: %d" %  len(TEST_DATASET))# 输出测试数据的个数
    num_classes = 16
    num_part = 50
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)# 导入指定的模型模块
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))# 将模型文件复制到实验目录
    shutil.copy('models/pointnet_util.py', str(experiment_dir))# 将pointnet_util.py文件复制到实验目录

    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()# 初始化分类器，并将其转移到GPU上
    criterion = MODEL.get_loss().cuda()# 初始化损失函数，并将其转移到GPU上

    def weights_init(m):
        """
        修改的权重初始化函数，添加的某些模块没有 bias。
        使用此方式可以避免报错。
        """
        classname = m.__class__.__name__

        # 检查是否有 weight 属性，并且类名包含 'Conv' 或 'Linear'
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

            if hasattr(m, 'bias') and m.bias is not None:
                #torch.nn.init.constant_(m.bias.data, 0.0)# 将偏置项初始化为常数0
                torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)

        # 针对 BatchNorm2d# 检查类名是否包含'BatchNorm2d'，如果是，则执行以下操作
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0)# 使用正态分布初始化权重，均值为1
            torch.nn.init.constant_(m.bias.data, 0.0)# 将偏置项初始化为常数0
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')# 尝试从实验目录中加载最佳模型的检查点
        start_epoch = checkpoint['epoch']# 获取检查点中的训练轮数
        classifier.load_state_dict(checkpoint['model_state_dict'])# 加载检查点中的模型状态字典
        log_string('Use pretrain model')# 记录日志，表示使用预训练模型
    except:
        log_string('No existing model, starting training from scratch...')# 如果加载失败，记录日志，表示从头开始训练
        start_epoch = 0# 设置起始训练轮数为0
        classifier = classifier.apply(weights_init)# 对分类器应用权重初始化函数

    if args.optimizer == 'Adam':# 根据参数选择优化器类型
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        ) # 创建Adam优化器
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)# 创建SGD优化器

    def bn_momentum_adjust(m, momentum):# 定义一个函数，用于调整批量归一化层的动量
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum# 如果模块是批量归一化层（2D或1D），则调整其动量

    LEARNING_RATE_CLIP = 1e-5# 设置学习率裁剪阈值
    MOMENTUM_ORIGINAL = 0.1# 设置原始动量值
    MOMENTUM_DECCAY = 0.5# 设置动量衰减系数
    MOMENTUM_DECCAY_STEP = args.step_size# 设置动量衰减步长，从args中获取

    best_acc = 0# 初始化最佳准确率为0
    global_epoch = 0# 初始化全局训练轮数为0
    best_class_avg_iou = 0# 初始化最佳类别平均IoU为0
    best_inctance_avg_iou = 0# 初始化最佳实例平均IoU为0

    for epoch in range(start_epoch,args.epoch):# 遍历所有训练轮次
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))# 打印当前轮次信息
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)# 计算当前学习率，并确保不低于设定的最小值
        log_string('Learning rate:%f' % lr)# 打印当前学习率
        for param_group in optimizer.param_groups:# 更新优化器的学习率
            param_group['lr'] = lr
        mean_correct = []# 初始化正确分类样本列表
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))# 计算当前动量值
        if momentum < 0.01:# 确保动量值不小于0.01
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)# 打印更新后的BN动量值
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))# 更新模型中的BN层动量值

        '''learning one epoch'''
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):# 遍历训练数据集，使用tqdm显示进度条
            points, label, target = data# 获取数据点、标签和目标
            points = points.data.numpy()# 将数据点转换为numpy数组

            points[:, :, 0:3] = provider.rotate_perturbation_point_cloud(points[:, :, 0:3])# 对点云数据进行平移
            points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3])# 对点云数据进行抖动
            points[:, :, 0:3] = provider.rotate_point_cloud(points[:, :, 0:3])# 再次对点云数据进行旋转



            points = torch.Tensor(points)# 将处理后的数据点转换回Tensor
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()# 将数据点、标签和目标转换为浮点数并移动到GPU上
            points = points.transpose(2, 1)# 调整数据点的维度顺序
            optimizer.zero_grad()# 清空优化器的梯度
            classifier = classifier.train()# 将分类器设置为训练模式
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))# 使用分类器对数据点进行预测，得到分割预测和转换特征
            seg_pred = seg_pred.contiguous().view(-1, num_part)# 调整分割预测的维度
            target = target.view(-1, 1)[:, 0]# 调整目标的维度
            pred_choice = seg_pred.data.max(1)[1]# 获取预测结果中概率最大的类别索引
            correct = pred_choice.eq(target.data).cpu().sum()# 计算预测正确的数量
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))# 将正确率添加到mean_correct列表中
            loss = criterion(seg_pred, target, trans_feat)# 计算损失函数
            loss.backward()# 反向传播计算梯度
            optimizer.step()# 更新优化器的参数
        train_instance_acc = np.mean(mean_correct)# 计算训练集的平均准确率
        log_string('Train accuracy is: %.5f' % train_instance_acc)# 输出训练准确率

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]

            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            # 创建一个字典，将分割标签映射到对应的类别名，这里假设cotton有三个子类，分别为Airplane, Airplane, Table
            seg_label_to_cat = {0:'cotton',1:'cotton',2:'cotton'}  # {0:Airplane, 1:Airplane, ...49:Table}
            # 遍历所有类别
            for cat in seg_classes.keys():
                # 遍历当前类别下的所有分割标签
                for label in seg_classes[cat]:
                    # 将分割标签映射到对应的类别名
                    seg_label_to_cat[label] = cat

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):# 遍历测试数据集的批次
                cur_batch_size, NUM_POINT, _ = points.size()# 获取当前批次的大小、点的数量和其他维度信息
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()# 将数据转换为浮点数并转移到GPU上
                points = points.transpose(2, 1)# 调整点的维度顺序
                classifier = classifier.eval()# 使用分类器进行预测
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                cur_pred_val = seg_pred.cpu().data.numpy()# 将预测结果从GPU转移到CPU并转换为numpy数组
                cur_pred_val_logits = cur_pred_val# 初始化当前批次的预测值
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()# 将目标值从GPU转移到CPU并转换为numpy数组
                #print(target)
                for i in range(cur_batch_size):# 遍历当前批次的所有样本
                    #print(target[i,0])
                    if target[i, 0]!=1 and target[i,0]!=0:# 如果目标值不为0或1，则将其设置为1
                        target[i, 0]=1
                    cat = seg_label_to_cat[target[i, 0]]# 获取类别标签对应的类别名称
                    logits = cur_pred_val_logits[i, :, :]# 获取当前样本的预测logits
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]# 根据类别名称计算预测值
                correct = np.sum(cur_pred_val == target)# 计算预测正确的点数
                total_correct += correct# 累加正确点数和总点数
                total_seen += (cur_batch_size * NUM_POINT)


                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            print(total_correct)
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)


        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
                 epoch+1, test_metrics['accuracy'],test_metrics['class_avg_iou'],test_metrics['inctance_avg_iou']))# 打印测试准确率、类别平均mIOU和实例平均mIOU
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):# 如果当前实例平均mIOU大于等于之前的最佳实例平均mIOU，则保存模型
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s'% savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            } # 创建一个字典，包含训练状态信息
            torch.save(state, savepath)
            log_string('Saving model....')# 将训练状态信息保存到指定路径的文件中

        if test_metrics['accuracy'] > best_acc:# 检查测试指标中的准确率是否高于当前最佳准确率
            best_acc = test_metrics['accuracy']# 如果是，则更新最佳准确率
        if test_metrics['class_avg_iou'] > best_class_avg_iou:# 检查测试指标中的平均类别IoU是否高于当前最佳平均类别IoU
            best_class_avg_iou = test_metrics['class_avg_iou']# 如果是，则更新最佳平均类别IoU
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:# 检查测试指标中的平均实例IoU是否高于当前最佳平均实例IoU
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']# 如果是，则更新最佳平均实例IoU
        log_string('Best accuracy is: %.5f'%best_acc)# 记录最佳准确率到日志
        log_string('Best class avg mIOU is: %.5f'%best_class_avg_iou)# 记录最佳平均类别IoU到日志
        log_string('Best inctance avg mIOU is: %.5f'%best_inctance_avg_iou)# 记录最佳平均实例IoU到日志
        logger.info(f"Train Loss: {loss.item()}")
        global_epoch+=1

if __name__ == '__main__':
    args = parse_args()
    main(args)

