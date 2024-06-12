import os, torch, cv2, numpy as np
from utils.cached import Cached

class PairedImageDataset(Cached):
    """
    Simple dataset that takes a lq folder and a hq folder and combines into a single batch
    """
    def __init__(self, lq_pth:str="datasets/lq", hq_pth:str="datasets/hq", batch_size=16, hq_size=256, use_cache=True, lq_size=64, pre_crop=False):
        super().__init__(batch_size=batch_size, compute_cache=use_cache)
        self.lq_pth = lq_pth
        self.hq_pth = hq_pth
        self.hq_size = hq_size
        self.lq_size = lq_size
        self.pre_crop = pre_crop
        self.use_cache = use_cache

        
        # Find all the files in the lq folder, we assume that the hq folder contains the same file names
        self.lq_img_names = [f for f in os.listdir(self.lq_pth) if os.path.isfile(os.path.join(self.lq_pth, f))]
        self.hq_img_names = [f for f in os.listdir(self.hq_pth) if os.path.isfile(os.path.join(self.hq_pth, f))]
        self.current_imgs = []

        self.lq_img_names.sort()
        self.hq_img_names.sort()

    def __len__(self):
        return len(self.lq_img_names)

    def __getitem__(self, idx):

        # Cache works to speed up training times, but can only work once all
        # samples have been seen
        if self.use_cache:
            batch = self.from_cache(idx)
            if not batch is None:
                return batch

        # Read the images
        lq_img = cv2.imread(os.path.join(self.lq_pth, self.lq_img_names[idx]))
        hq_img = cv2.imread(os.path.join(self.hq_pth, self.hq_img_names[idx]))

        if self.pre_crop:
            height, width, _ = hq_img.shape

            if height-1 < self.hq_size or width-1 < self.hq_size:
                return self.__getitem__(idx)

            start_y = np.random.randint(0, height - self.hq_size)
            start_x = np.random.randint(0, width - self.hq_size)
            hq_img = hq_img[start_y:start_y+self.hq_size, start_x:start_x+self.hq_size]
            lq_img = lq_img[start_y//4:start_y//4+self.lq_size, start_x//4:start_x//4+self.lq_size]

        
        if not (hq_img.shape == (self.hq_size, self.hq_size, 3)):
            return self.__getitem__(idx)
        
        if not (lq_img.shape == (self.lq_size, self.lq_size, 3)):
            return self.__getitem__(idx)
        
        if np.random.randint(0, 2) == 0:
            hq_img = cv2.flip(hq_img, 0)
            lq_img = cv2.flip(lq_img, 0)
        
        if np.random.randint(0, 2) == 0:
            hq_img = cv2.flip(hq_img, 1)
            lq_img = cv2.flip(lq_img, 1)

        # Convert to pytorch
        lq_img = torch.from_numpy(lq_img).float() / 255
        hq_img = torch.from_numpy(hq_img).float() / 255

        # Roll axis to what tensorflow expects: [w, h, 3] -> [3, w, h]        
        lq_img = torch.transpose(lq_img, 0, 2)
        hq_img = torch.transpose(hq_img, 0, 2)
        
        # Returns images as batch pair
        return lq_img, hq_img