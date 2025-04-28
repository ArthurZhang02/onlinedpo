import random
from typing import Dict, Any, Optional
import torch

class ReplayBuffer:
    def __init__(self, buffer_size: int):
        """
        初始化Replay Buffer
        
        参数:
            buffer_size (int): Buffer的最大容量
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self.current_size = 0
    
    def add(self, elements) -> bool:
        replaced = False
        batch_elements = torch.unbind(elements, dim=0)
        for elem in batch_elements:
            if self.current_size < self.buffer_size:
                self.buffer.append(elem.detach().clone())  # 关键修改：detach()+clone
                self.current_size += 1
            else:
                self.buffer.pop(0)
                self.buffer.append(elem.detach().clone())  # 关键修改
                replaced = True
        return replaced
        
    def sample(self, batch_size: int = 1, remove: bool = False, stack_dim: int = 0):
        if self.current_size == 0:
            return None
        
        actual_batch_size = min(batch_size, self.current_size)
        indices = random.sample(range(self.current_size), actual_batch_size)
        sampled_elements = [self.buffer[idx] for idx in indices]  # 直接取用，无需 clone

        if remove:
            for idx in sorted(indices, reverse=True):
                del self.buffer[idx]
            self.current_size -= actual_batch_size
        
        if batch_size == 1:
            return sampled_elements[0].clone().requires_grad_(True)
        
        # 返回 stack 后的张量，并确保无视图问题
        return torch.stack(sampled_elements, dim=stack_dim).clone().clone().requires_grad_(True)  # 最终 clone
        
    def __len__(self) -> int:
        """返回Buffer中当前元素的数量"""
        return self.current_size
if __name__ == '__main__':
    # 初始化Buffer，大小为3
    buffer = ReplayBuffer(3)

    # 添加元素
    print(buffer.add({"a": 1}))  # False (未满)
    print(buffer.add({"b": 2}))  # False (未满)
    print(buffer.add({"c": 3}))  # False (未满)
    print(buffer.add({"d": 4}))  # True (已满，替换最早的元素)

    # 采样元素
    print(buffer.sample())          # 随机返回一个元素，不删除
    print(buffer.sample(remove=True))  # 随机返回一个元素并从Buffer中删除

    # 查看当前Buffer大小
    print(len(buffer))  # 返回当前Buffer中的元素数量