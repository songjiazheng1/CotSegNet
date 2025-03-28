import torch.nn as nn
import torch
import torch.nn.functional as F
# from models.pointnet_util import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation
from models.pointnet_util import PointNetSetAbstractionMsg,  PointNetFeaturePropagation, \
    PointNetSetAbstractionMsgAttention,PointNetSetAbstractionAttention,PointNetSetAbstraction

class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsgAttention(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionAttention(npoint=2048, radius=0.1, nsample=100, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        #得到了全局特征之后，进行上采样
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss




# ##Varifocal Loss损失函数
# class get_loss(nn.Module):
#     def __init__(self, gamma=2.0, alpha=0.1):
#         super(get_loss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#
#     def forward(self, pred_logits, target, trans_feat):
#         # 将 target 转换为 one-hot 编码格式
#         target_one_hot = F.one_hot(target, num_classes=pred_logits.shape[-1]).float()
#
#         # 计算预测概率
#         pred_probs = torch.sigmoid(pred_logits)
#
#         # 计算 Focal Loss 中的 modulating factor
#         focal_weight = (1 - pred_probs) ** self.gamma
#
#         # 计算 BCE Loss，这里 target 需要是 one-hot 编码的形式
#         bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_one_hot, reduction='none')
#
#         # 对正样本额外增加一个误差平方项
#         alpha_factor = target_one_hot * self.alpha + (1 - target_one_hot) * (1 - self.alpha)
#         varifocal_term = torch.where(target_one_hot == 1, (1 - pred_probs).pow(2),
#                                      torch.tensor(0., device=pred_logits.device))
#         loss = alpha_factor * (focal_weight * bce_loss + varifocal_term)
#         total_loss = loss.mean()
#
#         # 返回损失的均值
#         return total_loss
# #Varifocal Loss损失函数




# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()
#         self.gamma=2
#     def forward(self, pred, target, trans_feat):#pred: 模型预测的输出   target: 真实的标签或数据，用于计算损失
#         #total_loss = F.nll_loss(pred, target, weight=weight)
#         logp = F.nll_loss(pred, target)
#         p = torch.exp(-logp)
#         loss = (1 - p) ** self.gamma * logp   #focalloss实现
#         total_loss= loss.mean()
#         #return total_loss
#         return total_loss
##focalloss损失函数