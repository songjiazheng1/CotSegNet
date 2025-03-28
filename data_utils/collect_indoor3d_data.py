import os# 导入os模块，用于处理文件和目录
import sys# 导入sys模块，用于处理Python运行时环境
from indoor3d_util import DATA_PATH, collect_point_label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))# 获取当前脚本的绝对路径
ROOT_DIR = os.path.dirname(BASE_DIR)# 获取当前脚本所在目录的上一级目录
sys.path.append(BASE_DIR)# 将当前脚本所在的目录添加到系统路径中

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]# 读取'meta/anno_paths.txt'文件中的每一行，并去除行尾的空白字符
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]# 将每个路径与DATA_PATH拼接，得到完整的路径

output_folder = os.path.join(ROOT_DIR, 'data/stanford_indoor3d')# 设置输出文件夹的路径
if not os.path.exists(output_folder):# 如果输出文件夹不存在
    os.mkdir(output_folder)# 创建输出文件夹

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:# 遍历所有的注释路径
    print(anno_path)# 打印注释路径
    try:
        elements = anno_path.split('/')# 将注释路径按照'/'分割成元素列表
        out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy# 生成输出文件名，格式为Area_1_hallway_1.npy
        collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')# 调用collect_point_label函数，处理注释文件，并将结果保存到输出文件夹中
    except:
        print(anno_path, 'ERROR!!')# 如果处理过程中出现异常，打印错误信息
