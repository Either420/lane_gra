"""
Generate a minimal fake CULane dataset for testing without downloading real data.
Usage:
    python scripts/make_fake_culane.py --root /path/to/fake_culane --num_train 100 --num_test 20
"""
import os
import cv2
import numpy as np
import json
import argparse


H, W = 590, 1640
LANE_XS = [300, 600, 950, 1280]  # x positions of 4 fake lanes
ROW_ANCHORS = list(range(250, 600, 10))  # [250, 260, ..., 590], 35 values


def make_image(path):
    img = np.random.randint(50, 200, (H, W, 3), dtype=np.uint8)
    cv2.imwrite(path, img)


def make_label(path):
    label = np.zeros((H, W), dtype=np.uint8)
    for lane_idx, lx in enumerate(LANE_XS):
        label[:, max(0, lx - 4):min(W, lx + 4)] = lane_idx + 1
    cv2.imwrite(path, label)


def make_lines_txt(path):
    with open(path, 'w') as f:
        for lx in LANE_XS:
            pts = []
            for y in ROW_ANCHORS:
                pts.extend([str(lx), str(y)])
            f.write(' '.join(pts) + '\n')


def make_cache_entry():
    all_points = np.full((4, 35, 2), -99999.0, dtype=np.float32)
    all_points[:, :, 1] = np.tile(ROW_ANCHORS, (4, 1))
    for lane_idx, lx in enumerate(LANE_XS):
        all_points[lane_idx, :, 0] = float(lx)
    return all_points.tolist()


def create_fake_culane(root, num_train, num_test):
    driver = 'driver_23_30frame'
    video = 'fake_video.MP4'
    seg_dir = 'laneseg_label_w16'

    img_dir   = os.path.join(root, driver, video)
    label_dir = os.path.join(root, seg_dir, driver, video)
    list_dir  = os.path.join(root, 'list', 'test_split')

    for d in [img_dir, label_dir, list_dir]:
        os.makedirs(d, exist_ok=True)

    train_lines, test_lines = [], []
    cache_dict = {}

    for i in range(num_train + num_test):
        name = f'{i:05d}'
        img_rel   = f'{driver}/{video}/{name}.jpg'
        label_rel = f'{seg_dir}/{driver}/{video}/{name}.png'

        make_image(os.path.join(root, img_rel))
        make_label(os.path.join(root, label_rel))
        make_lines_txt(os.path.join(root, img_rel.replace('.jpg', '.lines.txt')))

        cache_dict[img_rel] = make_cache_entry()

        if i < num_train:
            train_lines.append(f'/{img_rel} /{label_rel}\n')
        else:
            test_lines.append(f'/{img_rel}\n')

    with open(os.path.join(root, 'list', 'train_gt.txt'), 'w') as f:
        f.writelines(train_lines)
    with open(os.path.join(root, 'list', 'test.txt'), 'w') as f:
        f.writelines(test_lines)

    splits = ['test0_normal', 'test1_crowd', 'test2_hlight', 'test3_shadow',
              'test4_noline', 'test5_arrow', 'test6_curve', 'test7_cross', 'test8_night']
    for split in splits:
        with open(os.path.join(list_dir, f'{split}.txt'), 'w') as f:
            f.writelines(test_lines[:max(1, len(test_lines) // len(splits))])

    with open(os.path.join(root, 'culane_anno_cache.json'), 'w') as f:
        json.dump(cache_dict, f)

    print(f'Done. Fake CULane dataset at: {root}')
    print(f'  train: {num_train} images,  test: {num_test} images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--num_train', type=int, default=100)
    parser.add_argument('--num_test',  type=int, default=20)
    args = parser.parse_args()
    create_fake_culane(args.root, args.num_train, args.num_test)
