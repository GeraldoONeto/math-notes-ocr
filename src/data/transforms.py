from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image

class ResizePad:
    def __init__(self, target_size):
        self.target_width, self.target_height = target_size

    def __call__(self, img):
        img.thumbnail((self.target_width, self.target_height), Image.Resampling.LANCZOS)

        delta_w = self.target_width - img.size[0]
        delta_h = self.target_height - img.size[1]
        padding = [0, 0, int(delta_w), int(delta_h)]

        return F.pad(img, padding, fill=255)
    

def get_transform(max_dim, min_dim, train=False):
    transform_list = []

    if train:
        transform_list.extend([
            transforms.RandomRotation(degrees=[-5,5], fill=255),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])

    transform_list.extend([
        transforms.Grayscale(),
        ResizePad(max_dim),
        transforms.ToTensor()
    ])

    return transforms.Compose(transform_list)