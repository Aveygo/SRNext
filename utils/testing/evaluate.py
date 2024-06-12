import torch
from utils.testing.ssim import SSIM
from archs.datasets.testset import TestSet

PIXEL_MAX = 255

class Evaluate:
    def __init__(self):
        self.testset = TestSet(batch_size=1).dataloader
        self.ssim_module = SSIM(data_range=PIXEL_MAX)

    def ssim(self, img1, img2):
        return self.ssim_module(img1, img2)

    def rgb_to_ycbcr(self, input):
        # FIXED
        output = torch.autograd.Variable(input.data.new(*input.size()))
        output[:, 0, :, :] = input[:, 0, :, :] * 65.481 + input[:, 1, :, :] * 128.553 + input[:, 2, :, :] * 24.966 + 16
        return output[:, 0, :, :]

    def psnr(self, img1, img2):
        
        mse = torch.mean((img1 - img2)**2)
        if mse == 0:
            return 100
        
        return 20*torch.log10((PIXEL_MAX ** 2)/torch.sqrt(mse))

    def __call__(self, model:torch.nn.Module):
        
        pnsr_results = []
        ssim_results = []
        with torch.no_grad():
            model.eval().cuda()
            for idx, (x, y) in enumerate(self.testset):

                x = x.cuda()
                y = y.cuda()
                pred = model(x)

                y =  y * PIXEL_MAX
                pred = pred * PIXEL_MAX

                pred = torch.nn.functional.interpolate(pred, (y.shape[2], y.shape[3]))
                
                pnsr_results.append(
                    self.psnr(self.rgb_to_ycbcr(pred), self.rgb_to_ycbcr(y)).item()
                )

                ssim_results.append(
                    self.ssim(pred, y).item()
                )

        #print(pnsr_results, ssim_results)

        pnsr = sum(pnsr_results) / len(pnsr_results)
        ssim = sum(ssim_results) / len(ssim_results)
        
        return pnsr, ssim