import torch, numpy as np, cv2, os, pyiqa
import os, torch, cv2, numpy as np, random, time
from utils.cached import Cached
from utils.bicubic import imresize
from torchstat import ModelStat, report_format

class TestSet(Cached):
    def __init__(self, src="datasets/bsd100"):
        super().__init__(batch_size=1, compute_cache=False)
        self.src = src
        self.img_names = [f for f in os.listdir(self.src) if os.path.isfile(os.path.join(self.src, f))]
    
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        src_pth = os.path.join(self.src, self.img_names[idx])
        src_img = cv2.imread(src_pth) / 255

        src_img = cv2.rotate(src_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        src_img = cv2.flip(src_img, 0)

        h, w = src_img.shape[:2]
        size = max(h, w)
        src_img = cv2.copyMakeBorder(src_img, (size - h) // 2, (size - h + 1) // 2, (size - w) // 2, (size - w + 1) // 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        lq_img = imresize(src_img, (src_img.shape[1]/4) / src_img.shape[0])
        hq_img = imresize(src_img, (lq_img.shape[1]*4) / src_img.shape[0])

        #lq_img = cv2.resize(src_img, (src_img.shape[1]//4, src_img.shape[0]//4), interpolation=cv2.INTER_CUBIC)
        #hq_img = cv2.resize(src_img, (lq_img.shape[1]*4, lq_img.shape[0]*4), interpolation=cv2.INTER_CUBIC)
        
        lq_img = torch.from_numpy(lq_img).float()
        hq_img = torch.from_numpy(hq_img).float()

        # [w, h, 3] -> [3, w, h]
        lq_img = torch.transpose(lq_img, 0, 2)
        hq_img = torch.transpose(hq_img, 0, 2)
        
        return lq_img, hq_img
    
class TestSetPair(Cached):
    def __init__(self, lq_src="datasets/Set14_lq", hq_src="datasets/Set14_hq"):
        super().__init__(batch_size=1, compute_cache=False)

        self.lq_img_names = [os.path.join(lq_src, f) for f in os.listdir(lq_src) if os.path.isfile(os.path.join(lq_src, f))]
        self.hq_img_names = [os.path.join(hq_src, f) for f in os.listdir(hq_src) if os.path.isfile(os.path.join(hq_src, f))]
    
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def __len__(self):
        return len(self.lq_img_names)

    def __getitem__(self, idx):

        lq_img = cv2.flip(cv2.rotate((cv2.imread(self.lq_img_names[idx]) / 255), cv2.ROTATE_90_COUNTERCLOCKWISE), 0)
        hq_img = cv2.flip(cv2.rotate((cv2.imread(self.hq_img_names[idx]) / 255), cv2.ROTATE_90_COUNTERCLOCKWISE), 0)
        
        h, w, _ = src_img.shape
        window = 8*4
        pad_w = (window - w % window) % window
        pad_h = (window - h % window) % window
        src_img = np.pad(src_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    
        lq_img = cv2.resize(src_img, (src_img.shape[1]//4, src_img.shape[0]//4), interpolation=cv2.INTER_AREA)
        hq_img = cv2.resize(src_img, (lq_img.shape[1]*4, lq_img.shape[0]*4), interpolation=cv2.INTER_AREA)
        
        lq_img = torch.from_numpy(lq_img).float()
        hq_img = torch.from_numpy(hq_img).float()

        # [w, h, 3] -> [3, w, h]
        lq_img = torch.transpose(lq_img, 0, 2)
        hq_img = torch.transpose(hq_img, 0, 2)
        
        return lq_img, hq_img

def pre_model(img_lq):
    return img_lq

def post_model(img_lq, output):
    return output

def swin_pre_model(img_lq):
    window_size = 8
    # pad input image to be a multiple of window_size
    _, _, h_old, w_old = img_lq.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
    return img_lq

def swin_post_model(img_lq, output):
    scale = 4
    _, _, h_old, w_old = img_lq.size()
    return output[..., :h_old * scale, :w_old * scale]

class Metrics:
    def __init__(self, dataset:TestSet):
        self.testset = dataset.dataloader
        self.pnsr = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to("cuda")
        self.ssim = pyiqa.create_metric('ssim').to("cuda")
    
    def __call__(self, model:torch.nn.Module, dst:str="RENDER", pre_model=pre_model, post_model=post_model):
        if not os.path.exists(dst):
            os.makedirs(dst)
        pnsrscores = []
        ssimscores = []

        with torch.no_grad():
            model.eval().cuda()
            for idx, (x, y) in enumerate(self.testset):
                x = x.cuda()
                y = y.cuda()
                pred = post_model(x, model(pre_model(x)))
                
                pnsrscores.append(self.pnsr(y, pred).item())
                ssimscores.append(self.ssim(y, pred).item())

                pred = torch.clip(pred[0], 0, 1)
                pred = pred.detach().cpu().numpy()
                pred = np.rollaxis(pred, 0, 3)
                pred = (pred*255).astype(np.uint8)
                cv2.imwrite(os.path.join(dst + f"{idx}.png"), pred)
        
        return np.mean(pnsrscores), np.mean(ssimscores)

class SamplesPerSec:
    def __init__(self):
        pass

    def __call__(self, model:torch.nn.Module, targets=[64, 128, 256]):
        times = {}
        model.cuda()

        for size in targets:
            times.setdefault(size, [])
            start = time.time()
            frames = 0
            
            while (time.time() - start) < 5:
                y = model(torch.randn((1, 3, size, size), device="cuda"))
                frames += 1

            times[size].append(frames / (time.time() - start))
        
        for size, dtimes in times.items():
            times[size] = round(np.mean(dtimes) * 1000)
        model.cpu()

        return times

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

class Eval:
    def __init__(self):
        pass

    def __call__(self, model:torch.nn.Module, dst:str="RENDER/", pre_model=pre_model, post_model=post_model):
        a=time.time()
        print("=" * 20 + " Throughput stats " + "=" * 20)
        print("SIZE | FPS")
        results = SamplesPerSec()(model, targets=[64, 128, 256])
        for size, t in results.items():
            print(f"{size} | {t}")
        
        print("=" * 20 + " Metrics " + "=" * 20)
        print(f"NAME | PNSR | SSIM")
        datasets = {"SET14": "datasets/Set14/Set14/original", "BSD100": "datasets/BSDS100/BSDS100", "URBAN100": "datasets/urban100/urban100", "MANGA109": "datasets/manga109/manga109"}
        #datasets = {"SET14": "datasets/Set14/Set14/original"}
        for name, src in datasets.items():  
            pnsr, ssim = Metrics(TestSet(src))(model, os.path.join(dst + f"{name}/"))
            print(f"{name} | {pnsr:.4f} | {ssim:.4f}")

        #pnsr, ssim = Metrics(TestSetPair())(model, os.path.join(dst + f"SET14/"))
        #print(f"SET14 (Matlab) | {pnsr:.4f} | {ssim:.4f}")
        
        print("=" * 20 + " Memory stats " + "=" * 20)
        with suppress_stdout():
            stats = ModelStat(model.cpu(), (3, 256, 256), 1)
            stats = report_format(stats._analyze_model())
        print(stats)
        
        print("=" * 20 + " DONE " + "=" * 20)
        print(f"Total time: {time.time() - a:.4f}")

class BiCubic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="bicubic")
        
if __name__ == "__main__":
    import numpy as np, random, time

    torch.manual_seed(42)

    from archs.models.srnext import SRNext
    model = SRNext().eval()
    #model.load_state_dict(torch.load("ckpts/SRNext.ckpt"), strict=True)
    model.load_state_dict(torch.load("ckpts/SRNext_Final.ckpt"), strict=True)
    Eval()(model, "RENDER/SRNEXT")

    #from archs.models.swinir import SwinIR
    #model = SwinIR().eval()
    #model.load_state_dict(torch.load("ckpts/SwinIR.ckpt"), strict=True)
    #model.load_state_dict(torch.load("ckpts/official_swinir.pth")["params"], strict=True)
    #Eval()(model, "RENDER/SWINIR", swin_pre_model, swin_post_model)

    #model = BiCubic()
    #Eval()(model, "RENDER/BICUBIC")

    #from archs.models.IMDN import IMDN
    #model = IMDN(upscale=4).eval()
    #state_dict = torch.load("ckpts/IMDN_x4.pth")
    #new_state_dict = {}
    #for key in state_dict.keys():
    #    new_key = key.replace('module.', '', 1)
    #    new_state_dict[new_key] = state_dict[key]
    #model.load_state_dict(new_state_dict, strict=True)
    #Eval()(model, "RENDER/IMDN")

    #from archs.models.carn import Net as Carn
    #model = Carn()
    #model.load_state_dict(torch.load("ckpts/carn.pth"), strict=True)
    #Eval()(model, "RENDER/Carn")





        




    