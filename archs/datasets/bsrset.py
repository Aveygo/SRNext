import os, torch, cv2, numpy as np
from utils.cached import Cached
from utils.bsr import degradation_bsrgan

class BSRImageDataset(Cached):
    def __init__(self, src="datasets/Flickr2K", batch_size=32, use_cache=True, hq_size=256):
        super().__init__(batch_size=batch_size)
        assert not src is None, "Must provide source directory to create BSR dataset"
        self.src = src
        self.use_cache = use_cache
        self.hq_size = hq_size
        self.img_names = [f for f in os.listdir(self.src) if os.path.isfile(os.path.join(self.src, f))]
        
    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.img_names)-1)
        
        if self.use_cache:
            batch = self.from_cache(idx)
            if not batch is None:
                return batch

        src_img = cv2.imread(os.path.join(self.src, self.img_names[idx])) / 255

        height, width, _ = src_img.shape
        crop_size = int(self.hq_size * 1.2)

        if height < crop_size or width < crop_size:
            return self.__getitem__(idx)

        start_y = np.random.randint(0, height - crop_size)
        start_x = np.random.randint(0, width - crop_size)
        #print(start_y,start_y+crop_size, start_x,start_x+crop_size)
        src_img = src_img[start_y:start_y+crop_size, start_x:start_x+crop_size]

        try:
            print("start")
            lq_img, hq_img = degradation_bsrgan(src_img, 4, lq_patchsize=self.hq_size // 4)
            print("Got image")
        except:
            print("re-reading")
            return self.__getitem__(idx)
        
        lq_img = torch.from_numpy(lq_img).float()
        hq_img = torch.from_numpy(hq_img).float()

        # [w, h, 3] -> [3, w, h]
        lq_img = torch.transpose(lq_img, 0, 2)
        hq_img = torch.transpose(hq_img, 0, 2)
        
        return lq_img, hq_img

