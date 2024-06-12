import os, pickle, zipfile, hashlib, inspect, time
from torch.utils.data import Dataset, DataLoader

class Cached(Dataset):
    """
    Handy class that caches any pytorch dataset.
    Makes it much faster to load a dataset from disk, and avoids the overhead of loading it again.
    The only drawback is that the cache is "fixed" (eg random augmentations are not applied).
    """
    def __init__(self, batch_size:int=4, build_dir:str="/home/dmitri/Datasets/Cache/"):
        # Create the build folder
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        self.build_dir = build_dir
        self.data_dir = None
        self._zip = None
        self._built = 0
        self.batch_size = batch_size
    
    def create_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8) 

    @property
    def dataloader(self) -> DataLoader:
        if not hasattr(self, "_dlr"):
            setattr(self, "_dlr", self.create_dataloader())

        return getattr(self, "_dlr")
        
    def _compute_hash(self) -> str:
        # If the dataset changes, then we need to rebuild
        with open(inspect.getfile(self.__class__), "r") as f:
            content = "\n".join(f.readlines()).encode()
        return hashlib.sha256(content).hexdigest()[:8]

    def build_cache(self):
        # Start loading the batches into the zip file as pkl objects
        print("Building cache")
        self._zip = zipfile.ZipFile(self._data_dir, 'w')
        a = time.time()
        for i in range(5): self[i]
        print(f"Est. total time: ~{round((time.time() - a) / 5 * len(self) / 60)} minutes")
        
        for idx, batch in enumerate(self):
            self._zip.writestr(f"batch_{idx}", pickle.dumps(batch))
            print(f"{idx / len(self) * 100:.2f}%", end="\r")
        self._zip.close()
        print("")
        
    def from_cache(self, idx):
        # State: Not even open
        if self._built == 0:
            self._data_dir = os.path.join(self.build_dir, self._compute_hash() + '.zip')

            if not os.path.exists(self._data_dir):
                # Need to build the dataset
                self._built = 2
                self.build_cache()

            # Check if all batch files present
            self._zip = zipfile.ZipFile(self._data_dir, 'r')
            was_completed = len(self._zip.namelist()) == len(self)
            self._zip.close()
            
            if was_completed:
                # Built and ready to use           
                self._built = 1
            else:
                # Need to build the dataset
                self._built = 2
                self.build_cache()
            
            # Open zip file for reading finalized batches
            self._zip = zipfile.ZipFile(self._data_dir, 'r')
        
        # State: Building, dataset must create datapoints, not cache
        if self._built == 2:
            return None

        # Load the batch from the zip file
        return pickle.loads(self._zip.read(f"batch_{idx}"))