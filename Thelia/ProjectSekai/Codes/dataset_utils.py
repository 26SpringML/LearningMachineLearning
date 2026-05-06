"""ProjectSekai 数据集工具模块
将自定义类放在独立模块中，以支持多进程数据加载
"""

from torch.utils.data import Dataset
from PIL import Image


class ProjectSekaiDataset(Dataset):
    """ProjectSekai 图像分类数据集"""
    
    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label
