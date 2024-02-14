import torch
import cv2
import numpy as np
import math
from PIL import Image
import os
# (1) select perlin scale

def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out
def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])

def load_image(image_path, trg_h, trg_w):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    if trg_h and trg_w :
        image = image.resize((trg_w, trg_h), Image.BICUBIC)
    img = np.array(image, np.uint8)
    return img

image = load_image('001_recon.png', 512, 512)

gen_imgs = os.listdir('gen_images')
for gen_img in gen_imgs:

    name, ext = os.path.splitext(gen_img)
    img = load_image('001_recon.png', 512, 512)

    anomal_img_path = os.path.join('gen_images', gen_img)
    anomal_img = load_image(anomal_img_path, 512,512)
    # (3.1) perlin noise
    perlin_scale = 6
    min_perlin_scale = 2
    random_scale_x = torch.randint(min_perlin_scale, perlin_scale, (1,))
    random_scale_y = torch.randint(min_perlin_scale, perlin_scale, (1,))
    perlin_scalex = 2 ** (random_scale_x.numpy()[0])
    perlin_scaley = 2 ** (random_scale_y.numpy()[0])

    perlin_noise = rand_perlin_2d_np((512, 512), (perlin_scalex, perlin_scaley))  # 0 ~ 1
    threshold = 0.5
    perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise),
                          np.zeros_like(perlin_noise))
    perlin_thr_np = np.expand_dims(perlin_thr, axis=2)  # [512,512,1]

    augmented_img = (1 - perlin_thr_np) * img + (perlin_thr_np) * anomal_img
    augmented_img_pil = Image.fromarray(augmented_img.astype(np.uint8))
    augmented_img_pil.save(os.path.join('test',f'{name}_augmented.png'))

    anomal_mask_pil = Image.fromarray((perlin_thr * 255).astype(np.uint8))
    anomal_mask_pil.save(os.path.join('test',f'{name}_mask.png'))


