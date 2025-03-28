"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))#获取当前脚本所在的绝对路径，并将其赋值给变量BASE_DIR
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))#将ROOT_DIR下的models文件夹添加到系统路径中，以便在其他地方导入该文件夹下的模块


classes = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']#定义一个包含13个类别名称的列表classes
class2label = {cls: i for i,cls in enumerate(classes)}#使用字典推导式，将classes中的每个类别名称与其对应的索引值（从0开始）进行映射，生成一个新的字典class2label
seg_classes = class2label#定义一个空字典seg_label_to_cat，用于存储类别标签到类别名称的映射关系
seg_label_to_cat = {}#遍历seg_classes字典的键（即类别名称），并获取其对应的索引值
for i,cat in enumerate(seg_classes.keys()):#遍历seg_classes字典的键（即类别名称），并获取其对应的索引值
    seg_label_to_cat[i] = cat#将索引值作为键，类别名称作为值，添加到seg_label_to_cat字典中。这样，seg_label_to_cat就存储了类别标签到类别名称的映射关系


def parse_args():# 定义解析命令行参数的函数
    parser = argparse.ArgumentParser('Model') # 创建一个ArgumentParser对象
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]') # 添加'model'参数
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')# 添加'batch_size'参数
    parser.add_argument('--epoch',  default=128, type=int, help='Epoch to run [default: 128]')# 添加'epoch'参数
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int,  default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')

    return parser.parse_args() # 返回解析后的参数

def main(args):# 定义主函数，接受解析后的命令行参数作为输入
    def log_string(str):# 定义日志记录函数
        logger.info(str)# 记录到日志
        print(str)# 打印到控制台

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu# 设置使用的GPU

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) # 获取当前时间字符串
    experiment_dir = Path('./log/')# 创建日志目录路径
    experiment_dir.mkdir(exist_ok=True)# 创建日志目录（如果不存在）
    experiment_dir = experiment_dir.joinpath('sem_seg')# 添加子目录'sem_seg'
    experiment_dir.mkdir(exist_ok=True)# 创建子目录（如果不存在）
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)# 如果没有指定日志目录，使用当前时间字符串
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)# 否则使用指定的日志目录
    experiment_dir.mkdir(exist_ok=True) # 创建最终的日志目录（如果不存在）
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')# 创建检查点目录路径
    checkpoints_dir.mkdir(exist_ok=True)# 创建检查点目录（如果不存在）
    log_dir = experiment_dir.joinpath('logs/') # 创建日志文件目录路径
    log_dir.mkdir(exist_ok=True)# 创建日志文件目录（如果不存在）

    '''LOG'''
    args = parse_args() # 解析命令行参数
    logger = logging.getLogger("Model")# 创建日志记录器
    logger.setLevel(logging.INFO)# 设置日志级别为INFO
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')# 设置日志格式
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))# 创建日志文件处理器
    file_handler.setLevel(logging.INFO)# 设置日志文件处理器级别为INFO
    file_handler.setFormatter(formatter)# 为日志文件处理器设置格式
    logger.addHandler(file_handler)# 将处理器添加到日志记录器
    log_string('PARAMETER ...')# 记录参数信息
    log_string(args)# 打印和记录解析后的参数

    root = 'data/stanford_indoor3d/' # 数据集根目录
    NUM_CLASSES = 13 # 类别数
    NUM_POINT = args.npoint# 每个点云的点数
    BATCH_SIZE = args.batch_size # 批次大小

    print("start loading training data ...")# 打印加载训练数据的消息
    TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None) # 创建训练数据集
    print("start loading test data ...")# 打印加载测试数据的消息
    TEST_DATASET = S3DISDataset(split='test', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)# 创建测试数据集
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn = lambda x: np.random.seed(x+int(time.time()))) # 创建训练数据加载器
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)# 创建测试数据加载器
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()# 获取并转换标签权重

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))# 记录训练数据的数量
    log_string("The number of test data is: %d" % len(TEST_DATASET))# 记录测试数据的数量

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)# 动态导入模型模块
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))# 复制模型文件到实验目录
    shutil.copy('models/pointnet_util.py', str(experiment_dir))# 复制utils文件到实验目录

    classifier = MODEL.get_model(NUM_CLASSES).cuda() # 获取模型并加载到GPU
    criterion = MODEL.get_loss().cuda()# 获取损失函数并加载到GPU

    def weights_init(m): # 初始化权重
        classname = m.__class__.__name__ # 获取类名
        if classname.find('Conv2d') != -1:# 如果类名包含'Conv2d'
            torch.nn.init.xavier_normal_(m.weight.data)# 用Xavier初始化卷积层的权重
            torch.nn.init.constant_(m.bias.data, 0.0)# 将偏置初始化为0
        elif classname.find('Linear') != -1:# 如果类名包含'Linear'
            torch.nn.init.xavier_normal_(m.weight.data)# 用Xavier初始化全连接层的权重
            torch.nn.init.constant_(m.bias.data, 0.0)# 将偏置初始化为0

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')# 尝试加载已有的检查点
        start_epoch = checkpoint['epoch']# 获取起始训练轮数
        classifier.load_state_dict(checkpoint['model_state_dict'])# 加载模型状态
        log_string('Use pretrain model')# 记录使用预训练模型
    except:
        log_string('No existing model, starting training from scratch...')# 记录没有预训练模型，重新开始训练
        start_epoch = 0# 设置起始训练轮数为0
        classifier = classifier.apply(weights_init) # 初始化模型权重

    if args.optimizer == 'Adam': # 如果选择Adam优化器
        optimizer = torch.optim.Adam(# 创建Adam优化器
            classifier.parameters(), # 传入模型参数
            lr=args.learning_rate, # 学习率
            betas=(0.9, 0.999),# Beta参数
            eps=1e-08,# Epsilon参数
            weight_decay=args.decay_rate# 权重衰减
        )
    else:# 否则使用SGD优化器
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)# 创建SGD优化器

    def bn_momentum_adjust(m, momentum): # 调整BN层的动量
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):# 如果是BatchNorm层
            m.momentum = momentum# 设置动量

    LEARNING_RATE_CLIP = 1e-5# 学习率下限
    MOMENTUM_ORIGINAL = 0.1 # 初始动量
    MOMENTUM_DECCAY = 0.5 # 动量衰减
    MOMENTUM_DECCAY_STEP = args.step_size# 动量衰减步长

    global_epoch = 0# 全局轮数
    best_iou = 0# 最佳IoU

    for epoch in range(start_epoch,args.epoch): # 循环训练
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))# 记录当前轮数
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)# 计算学习率
        log_string('Learning rate:%f' % lr)# 记录学习率
        for param_group in optimizer.param_groups:# 更新优化器中的学习率
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))# 计算动量
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))# 应用动量调整
        num_batches = len(trainDataLoader)# 获取训练数据的批次数
        total_correct = 0# 初始化正确数
        total_seen = 0# 初始化总数
        loss_sum = 0# 初始化损失和
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):# 迭代训练数据
            optimizer.zero_grad()  # 梯度清零
            points, target = data
            points = points.data.numpy()# 转换点云数据为numpy数组
            points[:,:, :3] = provider.rotate_point_cloud_z(points[:,:, :3]) # 对点云数据进行Z轴旋转
            points = torch.Tensor(points)# 转换点云数据为Tensor
            points, target = points.float().cuda(),target.long().cuda()# 转换数据类型并加载到GPU
            points = points.transpose(2, 1)# 转置点云数据
            optimizer.zero_grad()
            classifier = classifier.train()
            seg_pred, trans_feat = classifier(points)# 前向传播
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES) # 重新调整预测结果的形状
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()# 转换目标为numpy数组
            target = target.view(-1, 1)[:, 0]# 重新调整目标的形状
            loss = criterion(seg_pred, target, trans_feat, weights) # 计算损失
            loss.backward()# 反向传播
            optimizer.step() # 更新参数
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()# 获取预测结果的最大值索引
            correct = np.sum(pred_choice == batch_label) # 计算正确预测数
            total_correct += correct # 累加正确数
            total_seen += (BATCH_SIZE * NUM_POINT) # 累加总数
            loss_sum += loss# 累加损失
        log_string('Training mean loss: %f' % (loss_sum / num_batches))  # 记录训练的平均损失
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))# 记录训练准确率

        if epoch % 5 == 0:# 每5轮保存一次模型
            logger.info('Save model...')# 记录保存模型
            savepath = str(checkpoints_dir) + '/model.pth' # 设置模型保存路径
            log_string('Saving at %s' % savepath)# 记录模型保存路径
            state = {# 创建保存的状态字典
                'epoch': epoch,# 当前轮数
                'model_state_dict': classifier.state_dict(), # 模型状态字典
                'optimizer_state_dict': optimizer.state_dict(),# 优化器状态字典
            }
            torch.save(state, savepath) # 保存模型
            log_string('Saving model....') # 记录保存模型

        '''Evaluate on chopped scenes'''
        with torch.no_grad():# 不计算梯度
            num_batches = len(testDataLoader)# 获取测试数据的批次数
            total_correct = 0# 初始化
            total_seen = 0  # 初始化总数
            loss_sum = 0# 初始化损失和
            labelweights = np.zeros(NUM_CLASSES)# 初始化标签权重
            total_seen_class = [0 for _ in range(NUM_CLASSES)] # 初始化每类总数
            total_correct_class = [0 for _ in range(NUM_CLASSES)]# 初始化每类正确数
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)] # 初始化每类IoU分母
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))# 记录当前评估轮数
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):  # 迭代测试数据
                points, target = data
                points = points.data.numpy()# 转换点云数据为numpy数组
                points = torch.Tensor(points)# 转换点云数据为Tensor
                points, target = points.float().cuda(), target.long().cuda()# 转换数据类型并加载到GPU
                points = points.transpose(2, 1)# 转置点云数据
                classifier = classifier.eval()
                seg_pred, trans_feat = classifier(points)# 前向传播
                pred_val = seg_pred.contiguous().cpu().data.numpy()# 获取预测结果并转换为numpy数组
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)# 重新调整预测结果的形状
                batch_label = target.cpu().data.numpy()# 转换目标为numpy数组
                target = target.view(-1, 1)[:, 0]# 重新调整目标的形状
                loss = criterion(seg_pred, target, trans_feat, weights)# 计算损失
                loss_sum += loss# 累加损失
                pred_val = np.argmax(pred_val, 2)# 获取预测结果的最大值索引
                correct = np.sum((pred_val == batch_label))# 计算正确预测数
                total_correct += correct# 累加正确数
                total_seen += (BATCH_SIZE * NUM_POINT) # 累加总数
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))# 计算每类样本数
                labelweights += tmp# 累加标签权重
                for l in range(NUM_CLASSES):# 计算每类的统计数据
                    total_seen_class[l] += np.sum((batch_label == l) )# 累加每类总数
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) )# 累加每类正确数
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)) )# 累加每类IoU分母
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))# 计算标签权重
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))# 计算平均IoU
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches))) # 记录评估的平均损失
            log_string('eval point avg class IoU: %f' % (mIoU))# 记录评估的平均类IoU
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))# 记录评估的点云准确率
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))# 记录评估的平均类准确率
            iou_per_class_str = '------- IoU --------\n' # 初始化IoU字符串
            for l in range(NUM_CLASSES): # 遍历每类
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (# 记录每类的IoU
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)# 打印每类的IoU
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))# 记录评估的平均损失
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))# 记录评估的准确率
            if mIoU >= best_iou:# 如果当前mIoU大于最佳mIoU
                best_iou = mIoU # 更新最佳mIoU
                logger.info('Save model...')# 记录保存模型
                savepath = str(checkpoints_dir) + '/best_model.pth'# 设置模型保存路径
                log_string('Saving at %s' % savepath)# 记录模型保存路径
                state = {# 创建保存的状态字典
                    'epoch': epoch,# 当前轮数
                    'class_avg_iou': mIoU,# 平均IoU
                    'model_state_dict': classifier.state_dict(), # 模型状态字典
                    'optimizer_state_dict': optimizer.state_dict(), # 优化器状态字典
                }
                torch.save(state, savepath)# 保存模型
                log_string('Saving model....') # 记录保存模型
            log_string('Best mIoU: %f' % best_iou) # 记录最佳mIoU
        global_epoch += 1# 全局轮数加1


if __name__ == '__main__':
    args = parse_args()
    main(args)

