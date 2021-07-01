import torch


class config(object):
    save_interval=2
    batch_size=8
    lrD=0.0002
    lrG=0.0002
    img_size=256
    pretrain_epoch=20
    train_epoch=100
    content_coe=10
    beta1=0.5
    beta2=0.999
    model_path='models'
    cartoon_root='shinkai/shinkai'
    photo_root='shinkai/scenery_photo'
    test_root='test'
    is_smooth=False
    generator_model='models/pretrain_epoch20.pth'
    discriminator_model=None

