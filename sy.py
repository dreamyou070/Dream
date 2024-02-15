import torch
import numpy as np
out_mask = torch.randn(1, 320, 64, 64)
out_mask_sm = torch.softmax(out_mask, dim=1)
# avg_pool2d
out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:],
                                                   kernel_size=7,
                                                   stride=1,
                                                   padding=7// 2).cpu().detach().numpy()
import einops
raw_map = torch.randn(8,64*64)
argmax_map = torch.max(raw_map, dim=0).values

print(argmax_map.shape)