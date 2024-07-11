import numpy as np
import torch
import model.SRGAN as SRGANarch
import model.SRCNN as SRCNNarch


srgan_weights_path = './model/srgan.pth'
srcnn_weights_path = './model/srcnn.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


SRGAN_model = SRGANarch.RRDBNet(3, 3, 64, 23, gc=32)
SRGAN_model.load_state_dict(torch.load(srgan_weights_path), strict=True)
SRGAN_model = SRGAN_model.to(device)
SRGAN_model.eval()


SRCNN_model = SRCNNarch.Generator()
SRCNN_model.load_state_dict(torch.load(srcnn_weights_path), strict=True)
SRCNN_model = SRCNN_model.to(device)
SRCNN_model.eval()


def upscale(img, model):
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    
    with torch.no_grad():
        if model == 'SRGAN':
            output = SRGAN_model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        elif model == 'SRCNN':
            output = SRCNN_model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    return output


def srcnn_upscale(img):
    pass
