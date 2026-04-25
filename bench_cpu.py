import torch
import time

# 模拟深度学习常用大矩阵
def test_pytorch_cpu():
    # 中等张量，贴合你日常调试/测试场景
    size = 2048
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # 预热
    for _ in range(3):
        _ = a @ b

    # 正式计时
    start = time.time()
    for _ in range(10):
        c = a @ b
    end = time.time()

    print(f"矩阵乘法 10次 耗时：{end - start:.2f} s")
    print(f"单次平均耗时：{(end - start)/10:.3f} s")

if __name__ == "__main__":
    test_pytorch_cpu()