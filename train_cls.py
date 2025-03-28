from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
"""
需要配置的参数：
--model pointnet2_cls_msg 
--normal 
--log_dir pointnet2_cls_msg
"""

def parse_args():
    '''PARAMETERS'''
    #parser模型参数
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training [default: 24]')#batchsize 16  32
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet_cls]')#model文件名对应
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')#训练多少轮
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')#学习率
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')#是否使用gpu
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')#
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')#优化器adam sgd
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')#你训练的模型及结果存储的路径
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')#学习率下降的方式
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')#
    return parser.parse_args()

def test(model, loader, num_class=40):#定义一个名为test的函数，接收三个参数：model（模型），loader（数据加载器）和num_class（类别数量，默认为40）。
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    #创建文件夹
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))#记录时间
    experiment_dir = Path('./log/')#存储路径
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
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

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = 'data/modelnet40_normal_resampled/'

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train',
                                       normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                      normal_channel=args.normal)  # 数据集组建重点ModelNetDataLoader自己写的
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=4)  # pytorch自带的
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_class = 40  # 自己改
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    classifier = MODEL.get_model(num_class, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')#尝试加载预训练模型的权重文件。
        start_epoch = checkpoint['epoch']#从权重文件中获取开始训练的轮次
        classifier.load_state_dict(checkpoint['model_state_dict'])#预训练模型的权重加载到分类器中
        log_string('Use pretrain model')#输出日志信息，表示使用预训练模型
    except:#如果加载预训练模型失败，则执行下面的代码块。
        log_string('No existing model, starting training from scratch...')#输出日志信息，表示没有找到预训练模型，将从零开始训练。
        start_epoch = 0#设置开始训练的轮次为0。


    if args.optimizer == 'Adam':#判断是否使用Adam优化器。
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )#如果是，则使用Adam优化器。
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)#使用SGD优化器

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)#定义学习率调度器，每20个epoch衰减一次学习率。
    global_epoch = 0#初始化全局轮次为0。
    global_step = 0#初始化全局步数为0。
    best_instance_acc = 0.0#初始化最佳实例精度为0
    best_class_acc = 0.0#初始化最佳类别精度为0。
    mean_correct = []#初始化一个空列表，用于存储每个batch的正确预测比例。

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        # optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面,
        # 但也不是绝对的，可以根据具体的需求来做。
        # 只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整。
        #训练模型关键
        scheduler.step()#调用学习率调度器，更新学习率
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):#使用tqdm库对训练数据加载器进行迭代，显示进度条
            points, target = data#从数据中获取点云数据和目标标签。
            points = points.data.numpy()#将点云数据转换为numpy数组
            points = provider.random_point_dropout(points) #对点云数据进行随机点丢失的数据增强。
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3]) #在数值上调大或调小，设置一个范围，对点云数据进行随机缩放的数据增强
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3]) #增加随机抖动，使测试结果更好，对点云数据进行随机平移的数据增强
            points = torch.Tensor(points)#将处理后的点云数据转换为PyTorch张量
            target = target[:, 0]#获取目标标签的第一个元素。

            points = points.transpose(2, 1)#交换点云数据的第二个和第三个维度
            points, target = points.cuda(), target.cuda()#将点云数据和目标标签移动到GPU上
            optimizer.zero_grad()#清空优化器的梯度
            #classifier是你的模型

            classifier = classifier.train()#将模型设置为训练模式
            pred, trans_feat = classifier(points)#将点云数据输入模型，得到预测结果和转换特征
            loss = criterion(pred, target.long(), trans_feat) #计算损失
            pred_choice = pred.data.max(1)[1]#获取预测结果中概率最大的类别
            correct = pred_choice.eq(target.long().data).cpu().sum()#计算预测正确的样本数量
            mean_correct.append(correct.item() / float(points.size()[0]))#将预测准确率添加到列表中
            loss.backward() #反向传播计算梯度
            optimizer.step() #最好的测试结果,更新模型参数
            global_step += 1#全局步数加一

        train_instance_acc = np.mean(mean_correct)#计算训练集实例精度的平均值
        log_string('Train Instance Accuracy: %f' % train_instance_acc)#输出训练集实例精度

        with torch.no_grad():#这是一个PyTorch的上下文管理器，用于在该代码块中禁用梯度计算。这可以减少内存消耗并加速计算。
            instance_acc, class_acc = test(classifier.eval(), testDataLoader)
            #这行代码调用了一个名为test的函数，并将分类器和测试数据加载器作为参数传递给它。该函数的作用是评估模型在测试集上的性能，并返回实例精度（instance accuracy）和类别精度（class accuracy）。
            if (instance_acc >= best_instance_acc):#判断当前实例精度是否大于等于历史最佳实例精度。如果是，则执行下面的代码块。
                best_instance_acc = instance_acc#更新历史最佳实例精度为当前实例精度。
                best_epoch = epoch + 1#更新历史最佳实例精度对应的轮次为当前轮次加一

            if (class_acc >= best_class_acc):#判断当前类别精度是否大于等于历史最佳类别精度。如果是，则执行下面的代码块。
                best_class_acc = class_acc#更新历史最佳类别精度为当前类别精度
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))#输出当前测试实例精度和类别精度。
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))#输出历史最佳实例精度和类别精度。

            if (instance_acc >= best_instance_acc):#再次判断当前实例精度是否大于等于历史最佳实例精度。如果是，则执行下面的代码块。
                logger.info('Save model...')#记录日志信息，表示正在保存模型
                savepath = str(checkpoints_dir) + '/best_model.pth'#设置模型保存路径
                log_string('Saving at %s'% savepath)#输出模型保存路
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }#创建一个字典，包含当前轮次、实例精度、类别精度、模型状态字典和优化器状态字典等信息
                torch.save(state, savepath)#将字典保存到指定的文件路径。
            global_epoch += 1#全局轮次加一

    logger.info('End of training...')#记录日志信息，表示训练结束。

if __name__ == '__main__':#这是一个Python程序的常见结构，用于判断当前脚本是否作为主程序运行。如果是作为主程序运行，那么会执行后面的代码块；如果不是作为主程序运行（例如被其他脚本导入），则不会执行后面的代码块。
    args = parse_args()
    main(args)
