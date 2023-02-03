# -*- coding: utf-8 -*-
# @Time    : 2023/2/3 15:03
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_preprocess.py
# @Software: PyCharm

import timm
from PIL import Image
import torchvision.transforms as T
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

model_timm = timm.create_model('convnext_atto', checkpoint_path='./polls/model_best.pth.tar', num_classes=2)
model_timm.default_cfg['input_size'] = (3, 512, 512)
config = resolve_data_config({}, model=model_timm)
transform = create_transform(**config)

WSISEG_STATS = [0.3114, 0.3166, 0.3946], [0.2587, 0.2598, 0.2958]
image_tfm = T.Compose(
    [T.ToTensor(),
     T.Normalize(mean=WSISEG_STATS[0], std=WSISEG_STATS[1]),
     T.Resize((256, 256))
     ]
)


def data_preprocess_for_timm(img_path):

    img = Image.open(img_path).convert('RGB')

    return transform(img).unsqueeze(0).numpy(), image_tfm(img).unsqueeze_(0).numpy()
