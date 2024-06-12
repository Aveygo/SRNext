import os, torch, cv2, numpy as np, random
from utils.cached import Cached
from utils.bicubic import imresize

class TestSet(Cached):
    def __init__(self, src="datasets/Set14", batch_size=1, use_cache=True, hq_size=256, lq_size=64, pre_crop=False):
        super().__init__(batch_size=batch_size, compute_cache=use_cache)
        self.src = src
        self.use_cache = use_cache
        self.hq_size = hq_size
        self.lq_size = lq_size
        self.pre_crop = pre_crop
        self.img_names = [f for f in os.listdir(self.src) if os.path.isfile(os.path.join(self.src, f))]
    
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):        
        if self.use_cache:
            batch = self.from_cache(idx)
            if not batch is None:
                return batch

        src_pth = os.path.join(self.src, self.img_names[idx % len(self.img_names)])
        src_img = cv2.imread(src_pth) / 255

        src_img = cv2.rotate(src_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.pre_crop:
            height, width, _ = src_img.shape
            crop_size = int(self.hq_size * (1 + np.random.rand()))

            if height-1 < crop_size or width-1 < crop_size:
                return self.__getitem__(idx)

            start_y = np.random.randint(0, height - crop_size)
            start_x = np.random.randint(0, width - crop_size)
            src_img = src_img[start_y:start_y+crop_size, start_x:start_x+crop_size]

        #hq_img = cv2.resize(src_img, (self.hq_size, self.hq_size), interpolation=cv2.INTER_AREA)
        #lq_img = cv2.resize(hq_img, (self.lq_size, self.lq_size), interpolation=cv2.INTER_AREA)

        if np.random.randint(0, 2) == 0:
            src_img = cv2.flip(src_img, 0)

        hq_img = src_img
        lq_img = src_img

        hq_scale = self.hq_size / hq_img.shape[0]
        hq_img = imresize(hq_img, hq_scale)

        lq_scale = self.lq_size / hq_img.shape[0]
        lq_img = imresize(hq_img, lq_scale)

        lq_img = torch.from_numpy(lq_img).float()
        hq_img = torch.from_numpy(hq_img).float()

        # [w, h, 3] -> [3, w, h]
        lq_img = torch.transpose(lq_img, 0, 2)
        hq_img = torch.transpose(hq_img, 0, 2)
        
        return lq_img, hq_img

