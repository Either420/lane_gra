import os, argparse
from data.dali_data import TrainCollect
from utils.dist_utils import get_rank, get_world_size, is_main_process, dist_print, DistSummaryWriter
from utils.config import Config
import torch
import time

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help = 'path to config file')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--dataset', default = None, type = str)
    parser.add_argument('--data_root', default = None, type = str)
    parser.add_argument('--epoch', default = None, type = int)
    parser.add_argument('--batch_size', default = None, type = int)
    parser.add_argument('--optimizer', default = None, type = str)
    parser.add_argument('--learning_rate', default = None, type = float)
    parser.add_argument('--weight_decay', default = None, type = float)
    parser.add_argument('--momentum', default = None, type = float)
    parser.add_argument('--scheduler', default = None, type = str)
    parser.add_argument('--steps', default = None, type = int, nargs='+')
    parser.add_argument('--gamma', default = None, type = float)
    parser.add_argument('--warmup', default = None, type = str)
    parser.add_argument('--warmup_iters', default = None, type = int)
    parser.add_argument('--backbone', default = None, type = str)
    parser.add_argument('--griding_num', default = None, type = int)
    parser.add_argument('--use_aux', default = None, type = str2bool)
    parser.add_argument('--sim_loss_w', default = None, type = float)
    parser.add_argument('--shp_loss_w', default = None, type = float)
    parser.add_argument('--note', default = None, type = str)
    parser.add_argument('--log_path', default = None, type = str)
    parser.add_argument('--finetune', default = None, type = str)
    parser.add_argument('--resume', default = None, type = str)
    parser.add_argument('--test_model', default = None, type = str)
    parser.add_argument('--test_work_dir', default = None, type = str)
    parser.add_argument('--num_lanes', default = None, type = int)
    parser.add_argument('--auto_backup', action='store_false', help='automatically backup current code in the log path')
    parser.add_argument('--var_loss_power', default = None, type = float)
    parser.add_argument('--num_row', default = None, type = int)
    parser.add_argument('--num_col', default = None, type = int)
    parser.add_argument('--train_width', default = None, type = int)
    parser.add_argument('--train_height', default = None, type = int)
    parser.add_argument('--num_cell_row', default = None, type = int)
    parser.add_argument('--num_cell_col', default = None, type = int)
    parser.add_argument('--mean_loss_w', default = None, type = float)
    parser.add_argument('--fc_norm', default = None, type = str2bool)
    parser.add_argument('--soft_loss', default = None, type = str2bool)
    parser.add_argument('--cls_loss_col_w', default = None, type = float)
    parser.add_argument('--cls_ext_col_w', default = None, type = float)
    parser.add_argument('--mean_loss_col_w', default = None, type = float)
    parser.add_argument('--eval_mode', default = None, type = str)
    parser.add_argument('--eval_during_training', default = None, type = str2bool)
    parser.add_argument('--split_channel', default = None, type = str2bool)
    parser.add_argument('--match_method', default = None, type = str, choices = ['fixed', 'hungarian'])
    parser.add_argument('--selected_lane', default = None, type = int, nargs='+')
    parser.add_argument('--cumsum', default = None, type = str2bool)
    parser.add_argument('--masked', default = None, type = str2bool)
    
    
    return parser

import numpy as np
def merge_config():
    args = get_args().parse_args()
    cfg = Config.fromfile(args.config)

    items = ['dataset','data_root','epoch','batch_size','optimizer','learning_rate',
    'weight_decay','momentum','scheduler','steps','gamma','warmup','warmup_iters',
    'use_aux','griding_num','backbone','sim_loss_w','shp_loss_w','note','log_path',
    'finetune','resume', 'test_model','test_work_dir', 'num_lanes', 'var_loss_power', 'num_row', 'num_col', 'train_width', 'train_height',
    'num_cell_row', 'num_cell_col', 'mean_loss_w','fc_norm','soft_loss','cls_loss_col_w', 'cls_ext_col_w', 'mean_loss_col_w', 'eval_mode', 'eval_during_training', 'split_channel', 'match_method', 'selected_lane', 'cumsum', 'masked']
    for item in items:
        if getattr(args, item) is not None:
            dist_print('merge ', item, ' config')
            setattr(cfg, item, getattr(args, item))

    if cfg.dataset == 'CULane':
        cfg.row_anchor = np.linspace(0.42,1, cfg.num_row)
        cfg.col_anchor = np.linspace(0,1, cfg.num_col)
    elif cfg.dataset == 'Tusimple':
        cfg.row_anchor = np.linspace(160,710, cfg.num_row)/720
        cfg.col_anchor = np.linspace(0,1, cfg.num_col)
    elif cfg.dataset == 'CurveLanes':
        cfg.row_anchor = np.linspace(0.4, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    else:
        cfg.row_anchor = np.linspace(0.42, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)

    return args, cfg


def save_model(net, optimizer, epoch,save_path, distributed):
    if is_main_process():
        model_state_dict = net.state_dict()
        state = {'model': model_state_dict, 'optimizer': optimizer.state_dict()}
        # state = {'model': model_state_dict}
        assert os.path.exists(save_path)
        model_path = os.path.join(save_path, 'model_best.pth')
        torch.save(state, model_path)

import fnmatch

def _match_gitignore(patterns, path):
    for pattern in patterns:
        pattern = pattern.strip()
        if not pattern or pattern.startswith('#'):
            continue
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
            return True
    return False

def cp_projects(auto_backup, to_path):
    if is_main_process() and auto_backup:
        with open('./.gitignore','r') as fp:
            ign = fp.read()
        ign += '\n.git'
        patterns = ign.splitlines()
        all_files = {os.path.join(root,name) for root,dirs,files in os.walk('./') for name in files}
        matches = {f for f in all_files if _match_gitignore(patterns, f)}
        to_cp_files = all_files - matches
        dist_print('Copying projects to '+ to_path + ' for backup')
        t0 = time.time()
        warning_flag = True
        for f in to_cp_files:
            dirs = os.path.join(to_path,'code',os.path.split(f[2:])[0])
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            os.system('cp %s %s'%(f,os.path.join(to_path,'code',f[2:])))
            elapsed_time = time.time() - t0
            if elapsed_time > 5 and warning_flag:
                dist_print('If the program is stuck, it might be copying large files in this directory. please don\'t set --auto_backup. Or please make you working directory clean, i.e, don\'t place large files like dataset, log results under this directory.')
                warning_flag = False




import datetime, os
def get_work_dir(cfg):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    hyper_param_str = '_lr_%1.0e_b_%d' % (cfg.learning_rate, cfg.batch_size)
    work_dir = os.path.join(cfg.log_path, now + hyper_param_str + cfg.note)
    return work_dir

def get_logger(work_dir, cfg):
    os.makedirs(work_dir, exist_ok=True)
    logger = DistSummaryWriter(work_dir)
    config_txt = os.path.join(work_dir, 'cfg.txt')
    if is_main_process():
        with open(config_txt, 'w') as fp:
            fp.write(str(cfg))

    return logger

def initialize_weights(*models):
    for model in models:
        real_init_weights(model)
def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)
            
import importlib
def get_model(cfg):
    try:
        module = importlib.import_module('model.model_' + cfg.dataset.lower())
    except ModuleNotFoundError:
        module = importlib.import_module('model.model_culane')
    return module.get_model(cfg)

def get_train_loader(cfg):
    if cfg.dataset == 'CULane':
        train_loader = TrainCollect(cfg.batch_size, 4, cfg.data_root, os.path.join(cfg.data_root, 'list/train_gt.txt'), get_rank(), get_world_size(), 
                                cfg.row_anchor, cfg.col_anchor, cfg.train_width, cfg.train_height, cfg.num_cell_row, cfg.num_cell_col, cfg.dataset, cfg.crop_ratio)
    elif cfg.dataset == 'Tusimple':

        train_loader = TrainCollect(cfg.batch_size, 4, cfg.data_root, os.path.join(cfg.data_root, 'train_gt.txt'), get_rank(), get_world_size(), 
                                cfg.row_anchor, cfg.col_anchor, cfg.train_width, cfg.train_height, cfg.num_cell_row, cfg.num_cell_col, cfg.dataset, cfg.crop_ratio)
    elif cfg.dataset == 'CurveLanes':
        train_loader = TrainCollect(cfg.batch_size, 4, cfg.data_root, os.path.join(cfg.data_root, 'train', 'train_gt.txt'), get_rank(), get_world_size(), 
                                cfg.row_anchor, cfg.col_anchor, cfg.train_width, cfg.train_height, cfg.num_cell_row, cfg.num_cell_col, cfg.dataset, cfg.crop_ratio)
    else:
        raise NotImplementedError
    return train_loader 

def inference(net, data_label, dataset):
    if dataset == 'CurveLanes':
        return inference_curvelanes(net, data_label)
    elif dataset in ['Tusimple', 'CULane']:
        return inference_culane_tusimple(net, data_label)
    else:
        raise NotImplementedError

def inference_culane_tusimple(net, data_label):
    pred = net(data_label['images'])

    if 'row_anchors' in pred:
        # Dynamic anchor mode: recompute GT labels at predicted anchor positions
        from model.anchor_predictor import interp_label_rows, interp_label_cols
        from data.constant import culane_row_anchor, culane_col_anchor

        raw_pts = data_label['raw_points']   # [B, 4, 35, 2]  (x_px, row_px)
        B = raw_pts.shape[0]

        # row_anchors are fractions in [0.42,1.0]; convert to pixel coords
        # CULane image height = 590
        img_h = 590
        img_w = 1640
        row_anchors_px = pred['row_anchors'] * img_h   # [B, num_row]
        col_anchors_px = pred['col_anchors'] * img_w   # [B, num_col]

        num_cell_row = pred['loc_row'].shape[1]   # num_grid_row
        num_cell_col = pred['loc_col'].shape[1]

        # Interpolate GT x at dynamic row positions → [B, num_lanes, num_row]
        x_at_rows = interp_label_rows(raw_pts, row_anchors_px)   # [B, L, num_row]
        labels_row = (x_at_rows / img_w * (num_cell_row - 1)).long()
        labels_row[x_at_rows < 0]               = -1
        labels_row[labels_row < 0]              = -1
        labels_row[labels_row > num_cell_row-1] = -1
        labels_row = labels_row.permute(0, 2, 1)   # [B, num_row, num_lanes]

        labels_row_float = (x_at_rows / img_w).permute(0, 2, 1)
        labels_row_float[labels_row_float < 0] = -1
        labels_row_float[labels_row_float > 1] = -1

        # Interpolate GT row at dynamic col positions → [B, num_lanes, num_col]
        row_at_cols = interp_label_cols(raw_pts, col_anchors_px)
        labels_col = (row_at_cols / img_h * (num_cell_col - 1)).long()
        labels_col[row_at_cols < 0]              = -1
        labels_col[labels_col < 0]               = -1
        labels_col[labels_col > num_cell_col-1]  = -1
        labels_col = labels_col.permute(0, 2, 1)  # [B, num_col, num_lanes]

        labels_col_float = (row_at_cols / img_h).permute(0, 2, 1)
        labels_col_float[labels_col_float < 0] = -1
        labels_col_float[labels_col_float > 1] = -1
    else:
        labels_row       = data_label['labels_row']
        labels_col       = data_label['labels_col']
        labels_row_float = data_label['labels_row_float']
        labels_col_float = data_label['labels_col_float']

    cls_out_ext_label     = (labels_row != -1).long()
    cls_out_col_ext_label = (labels_col != -1).long()

    res_dict = {
        'cls_out':             pred['loc_row'],
        'cls_label':           labels_row,
        'cls_out_col':         pred['loc_col'],
        'cls_label_col':       labels_col,
        'cls_out_ext':         pred['exist_row'],
        'cls_out_ext_label':   cls_out_ext_label,
        'cls_out_col_ext':     pred['exist_col'],
        'cls_out_col_ext_label': cls_out_col_ext_label,
        'labels_row_float':    labels_row_float,
        'labels_col_float':    labels_col_float,
    }
    if 'seg_out' in pred:
        res_dict['seg_out']   = pred['seg_out']
        res_dict['seg_label'] = data_label['seg_images']
    if 'row_logits' in pred:
        res_dict['row_logits'] = pred['row_logits']
        res_dict['col_logits'] = pred['col_logits']

    return res_dict
def inference_curvelanes(net, data_label):
    pred = net(data_label['images'])
    cls_out_ext_label = (data_label['labels_row'] != -1).long()
    cls_out_col_ext_label = (data_label['labels_col'] != -1).long()

    res_dict = {'cls_out': pred['loc_row'], 'cls_label': data_label['labels_row'], 'cls_out_col':pred['loc_col'],'cls_label_col':data_label['labels_col'],
                'cls_out_ext':pred['exist_row'], 'cls_out_ext_label':cls_out_ext_label, 'cls_out_col_ext':pred['exist_col'],
                'cls_out_col_ext_label':cls_out_col_ext_label, 'seg_label': data_label['seg_images'], 'seg_out_row': pred['lane_token_row'], 'seg_out_col': pred['lane_token_col'] }
    if 'seg_out' in pred.keys():
        res_dict['seg_out'] = pred['seg_out']
        res_dict['seg_label'] = data_label['segs']
    return res_dict

def calc_loss(loss_dict, results, logger, global_step, epoch):
    loss = 0

    for i in range(len(loss_dict['name'])):

        if loss_dict['weight'][i] == 0:
            continue
            
        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)

        if global_step % 20 == 0:
            logger.add_scalar('loss/'+loss_dict['name'][i], loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]

    return loss