import torch
import copy
import sys
sys.path.append('./RIDCP_dehazing')
from basicsr.archs.dehaze_vq_weight_arch import VQWeightDehazeNet
from PIL import Image
import cv2
import numpy as np
from basicsr.utils import img2tensor, tensor2img

path ='RIDCP_dehazing/examples/canon_input.png'
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_path = 'RIDCP_dehazing/pretrained_models/pretrained_RIDCP.pth'

    sr_model = VQWeightDehazeNet(codebook_params=[[64, 1024, 512]], \
                                 LQ_stage=True, use_weight=False , weight_alpha=-21.25).to(device)
    sr_model.load_state_dict(torch.load(weight_path)['params'], strict=False)

    img = Image.open('RIDCP_dehazing/examples/canon_input.png')
    img.show()

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img_tensor = img2tensor(img).to(device) / 255.
    img_tensor = img_tensor.unsqueeze(0)
    if img.max() > 255.0:
        img = img / 255.0
    if img.shape[-1] > 3:
        img = img[:, :, :3]

    output, _ = sr_model.test(img_tensor)
    output_img = tensor2img(output)
    pil_image = Image.fromarray(output_img[..., ::-1].astype('uint8')) ## # Convert BGR to RGB

    pil_image.show()

if __name__ == '__main__':
    main()