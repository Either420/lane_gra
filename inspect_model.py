import torch
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'best_model/model1.pth'
ckpt = torch.load(path, map_location='cpu')
model = ckpt['model']

print('=== 所有层的参数形状 ===')
for k, v in sorted(model.items()):
    print(f'{k}: {list(v.shape)}')
