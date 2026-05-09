import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('img_dir', help='图片文件夹路径')
parser.add_argument('output_txt', help='输出的txt文件路径')
args = parser.parse_args()

img_files = sorted(
    glob.glob(os.path.join(args.img_dir, '*.jpg')) +
    glob.glob(os.path.join(args.img_dir, '*.jpeg')) +
    glob.glob(os.path.join(args.img_dir, '*.png'))
)

assert len(img_files) > 0, '没有找到图片：' + args.img_dir

with open(args.output_txt, 'w') as f:
    for p in img_files:
        f.write(os.path.basename(p) + '\n')

print('已写入 %d 张图片路径到 %s' % (len(img_files), args.output_txt))

# python make_img_list.py KITTI/test KITTI/test/test.txt