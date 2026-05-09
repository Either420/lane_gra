dataset = 'Custom'
data_root = ''        # 填写你的图片文件夹路径，例如 'KITTI/test'
test_model = ''       # 填写模型路径，例如 'best_model/tusimple_res18.pth'
test_work_dir = 'result/custom'

# Tusimple 模型结构参数（不要修改）
backbone = '18'
num_row = 56
num_col = 41
num_cell_row = 100
num_cell_col = 100
num_lanes = 4
train_width = 800
train_height = 320
crop_ratio = 0.8
fc_norm = False

# 以下为训练参数，推理时不使用，保留防止报错
epoch = 100
batch_size = 1
optimizer = 'SGD'
learning_rate = 0.05
weight_decay = 0.0001
momentum = 0.9
scheduler = 'multi'
steps = [50, 75]
gamma = 0.1
warmup = 'linear'
warmup_iters = 100
use_aux = False
griding_num = 100
sim_loss_w = 0.0
shp_loss_w = 0.0
mean_loss_w = 0.05
var_loss_power = 2.0
note = ''
log_path = ''
finetune = None
resume = None
auto_backup = False
