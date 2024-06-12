import os, pickle, zipfile, hashlib, inspect, time, torch
from torch.utils.data import Dataset, DataLoader

class Cached(Dataset):
    """
    Handy class that caches any pytorch dataset.
    Makes it much faster to load a dataset from disk, and avoids the overhead of loading it again.
    The only drawback is that the cache is "fixed" (eg random augmentations are not applied).
    """
    def __init__(self, batch_size:int=4, build_dir:str="datasets/cache/", compute_cache=True):
        # Create the build folder
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        self.build_dir = build_dir
        self.data_dir = None
        self._zip = None
        self._built = False
        self.batch_size = batch_size
        self.compute_cache = compute_cache
    
    def create_dataloader(self):
        g = torch.Generator()
        g.manual_seed(0)
        return DataLoader(self, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8, worker_init_fn=self.seed_worker, generator=g) 

    def seed_worker(self, worker_id):
        pass

    @property
    def dataloader(self) -> DataLoader:
        if not hasattr(self, "_dlr"):
            self._data_dir = os.path.join(self.build_dir, self._compute_hash() + '.zip')
            if not os.path.exists(self._data_dir):
                self.build_cache()
            setattr(self, "_dlr", self.create_dataloader())

        return getattr(self, "_dlr")
        
    def _compute_hash(self) -> str:
        # If the dataset changes, then we need to rebuild
        with open(inspect.getfile(self.__class__), "r") as f:
            content = "\n".join(f.readlines()).encode()
        return hashlib.sha256(content).hexdigest()[:8]

    def build_cache(self): 
        if not self.compute_cache:
            #print("Not using cache...")
            return
        
        # Start loading the batches into the zip file as pkl objects
        print(f"Building cache: {self._data_dir}")
        self._zip = zipfile.ZipFile(self._data_dir, 'w')
        a = time.time()
        for i in range(5): self[i]
        print(f"Est. total time: ~{round((time.time() - a) / 5 * len(self) / 60)} minutes")
        
        for idx, batch in enumerate(self):
            self._zip.writestr(f"batch_{idx}", pickle.dumps(batch))
            print(f"{idx / len(self) * 100:.2f}%", end="\r")
        self._zip.close()
        self._zip = zipfile.ZipFile(self._data_dir, 'r')
        print("")
        
    def from_cache(self, idx):
        # State: Not even open
        if not self._built:
            return None

        # Load the batch from the zip file
        return pickle.loads(self._zip.read(f"batch_{idx}"))