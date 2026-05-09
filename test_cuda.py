import torch
# 更详细的GPU信息
if torch.cuda.is_available():
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"计算能力: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    print(f"多处理器数量: {torch.cuda.get_device_properties(0).multi_processor_count}")
    
    # 监控显存使用
    print(f"已分配显存: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
    print(f"缓存显存: {torch.cuda.memory_reserved(0) / 1e6:.2f} MB")

def test_gpu():
    # 创建一个简单的张量并将其移动到GPU
    x = torch.rand(1000, 1000).cuda()
    print("张量已成功移动到GPU。")
    print(f"张量设备: {x.device}")

test_gpu()