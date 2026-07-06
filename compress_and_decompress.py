import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from models import (
    DCAE,
)
import warnings
import torch
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image
import pandas as pd
import numpy as np
import torch.nn as nn
import struct
warnings.filterwarnings("ignore")

print(torch.cuda.is_available())

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    bpp_dict = {}
    ans = 0
    for likelihoods in out_net['likelihoods']:
        fsize = out_net['likelihoods'][likelihoods].size()
        num = 1
        for s in fsize:
            num = num * s
        bpp_dict[likelihoods] = torch.log(out_net['likelihoods'][likelihoods]).sum() / (-math.log(2) * num_pixels)
        ans = ans + bpp_dict[likelihoods]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    # pad_layer = nn.ReflectionPad2d(padding)
    # x_padded = pad_layer(x)
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, padding

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )



def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--M", type=int, default=320,
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--save_path", type=str, help="Path to save")
    parser.add_argument("--data", type=str, help="Path to dataset")
    parser.add_argument("--mode", type=str, choices=['compress', "decompress"])
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args

def save_png_image(img, img_name, png_path):
    img = img.squeeze(0).permute(1,2,0).cpu().numpy()
    file_name, ext = os.path.splitext(img_name)
    new_img_name = file_name + '.png'
    decompressed_path = os.path.join(png_path, new_img_name)
    os.makedirs(os.path.dirname(decompressed_path), exist_ok=True)
    print(decompressed_path)
    plt.imsave(decompressed_path, img)
    
def save_bin(string, size, img_name, bin_path):
    file_name, ext = os.path.splitext(img_name)
    bin_name = file_name + '.bin'
    compress_path = os.path.join(bin_path,'bin/', bin_name)
    os.makedirs(os.path.dirname(compress_path), exist_ok=True)
    # print(type(string[0][0]))
    with open(compress_path, 'wb') as f:
        f.write(struct.pack(">H", size[0]))
        f.write(struct.pack(">H", size[1]))
        f.write(struct.pack(">I", len(string[0][0]))) 
        f.write(string[0][0])
        f.write(struct.pack(">I", len(string[1][0])))  
        f.write(string[1][0])
    
def calculate_padding(h, w, p=128):
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    padding_size = (new_h, new_w)
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    return padding_size, padding

def read_bin(bin_path):
    with open(bin_path, 'rb') as f:
        h = struct.unpack(">H", f.read(2))[0]
        w = struct.unpack(">H", f.read(2))[0]
        length_y = struct.unpack(">I", f.read(4))[0]
        string_y = f.read(length_y)
        length_z = struct.unpack(">I", f.read(4))[0]
        string_z = f.read(length_z)
        
    padding_size, padding = calculate_padding(h, w)
    z_shape = [padding_size[0]//64, padding_size[1]//64]
    
    string = [[string_y], [string_z]]
    return string, z_shape, padding
    

def main(argv):
    torch.backends.cudnn.enabled = False
    args = parse_args(argv)
    p = 128
    path = args.data
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"] and args.mode == "compress":
            img_list.append(file)
        elif file[-3:] in ["bin"] and args.mode == "decompress":
            # print(file)
            img_list.append(file)
        
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    base_path = args.save_path
    net = DCAE()
    net = net.to(device)
    net.eval()
    
    dictory = {}
    if args.checkpoint:  
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)
        
    net.update()
    if args.mode == "compress":
        for img_name in img_list:    
            with torch.no_grad():
                img_path = os.path.join(path, img_name)
                img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
                x = img.unsqueeze(0)
                x_size = x.size()[-2:]
                x_padded, padding = pad(x, p)
                x_padded.to(device)
                out_enc = net.compress(x_padded)
                save_bin(out_enc["strings"], x_size, img_name, base_path)

    elif args.mode == "decompress":
        for img_name in img_list:  
            with torch.no_grad():
                bin_path = os.path.join(path, img_name)
                string, shape, padding = read_bin(bin_path)
                out_dec = net.decompress(string, shape)
                out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                out_dec["x_hat"] = out_dec["x_hat"].clamp(0, 1)
                to_save_img = out_dec["x_hat"]
                save_png_image(to_save_img, img_name, base_path)

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])
    