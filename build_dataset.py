"""
Script to take an existing folder of images, eg: Flickr2k, and export them to hq/lq folders
"""
from archs.datasets.bsrset import BSRImageDataset
import os, cv2, numpy as np, torch

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description ='Export BSR outputs to hq/lq folders')
    parser.add_argument('src', type=str)
    parser.add_argument('lq_dst', type=str)
    parser.add_argument('hq_dst', type=str)
    args = parser.parse_args()

    print(args.src)
    print(args.lq_dst)
    print(args.hq_dst)
    
    dataset = BSRImageDataset(src=args.src, use_cache=False, hq_size=1024)
    for idx, batch in enumerate(dataset):
        print('batch shape:', batch[0].shape, batch[1].shape)
        
        lq_img = batch[0]
        hq_img = batch[1]

        lq_img = torch.transpose(lq_img, 2, 0).detach().cpu()
        hq_img = torch.transpose(hq_img, 2, 0).detach().cpu()

        cv2.imwrite(os.path.join(args.lq_dst, str(idx) + '.png'), (lq_img.numpy() * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(args.hq_dst, str(idx) + '.png'), (hq_img.numpy() * 255).astype(np.uint8))
        
    print('Done')
