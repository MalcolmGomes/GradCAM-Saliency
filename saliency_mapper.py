import torch
import time
import sys
import requests
import os
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.utils import make_grid, save_image
from utils import visualize_cam, Normalize
from gradcam import GradCAM, GradCAMpp

def generate_saliency_map(img, img_name):
    start = time.time()

    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    torch_img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).unsqueeze(0).float().div(255)
    torch_img = F.upsample(torch_img, size=(512, 512), mode='bilinear', align_corners=False)
    normed_torch_img = normalizer(torch_img)

    resnet = models.resnet101(pretrained=True)
    resnet.eval()
    cam_dict = dict()
    model_dict = dict(type='resnet', arch=resnet, layer_name='layer4', input_size=(512, 512))
    gradcam = GradCAM(model_dict, True)
    gradcam_pp = GradCAMpp(model_dict, True)
    
    images = []

    mask, _ = gradcam(normed_torch_img)
    heatmap, result = visualize_cam(mask, torch_img)
    mask_pp, _ = gradcam_pp(normed_torch_img)
    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)    
    images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))
    images = make_grid(torch.cat(images, 0), nrow=1)

    # Only going to use result_pp
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_name = img_name
    output_path = os.path.join(output_dir, output_name)
    save_image(result_pp, output_path)

    end = time.time()
    duration = round(end - start, 2)
    return output_path

if __name__ == '__main__':
    filename = 'malcolm.jpg'
    url = input('Enter image url: ')
    urlBool = False
    if url != '':
        urlBool = True
        filename = url.split('/')[-1]
        r = requests.get(url, allow_redirects=True)
        open(filename, 'wb').write(r.content)           
    img = Image.open(filename)          
    print("Running saliency mapper neural network on image:", filename)  
    result = generate_saliency_map(img, filename)
    Image.open(result).show()    
    if (urlBool): os.remove(filename)
    input("\nExecution Complete.")