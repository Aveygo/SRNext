import torch, lightning as L, numpy as np, cv2, lpips
from utils.testing.evaluate import Evaluate
from utils.perceptual_loss import  VGGPerceptualLoss
from utils.ssim_loss import SSIM

class Simple(L.LightningModule):
    """
    "Simple" experiment -> train uNext to predict hq from lq
    """
    def __init__(self, generator:torch.nn.Module, use_vgg:bool=False, use_ssim:bool=False):
        super().__init__()
        self.test = Evaluate()
        
        self.use_vgg = use_vgg
        if use_vgg:
            self.vgg = VGGPerceptualLoss().cuda()
        
        self.use_ssim = use_ssim
        if use_ssim:
            self.ssim = SSIM().cuda()

        self.generator = generator
        self.name = f"{type(self.generator).__name__}_{__class__.__name__}"

        self.initial_lr = 2e-4
        self.min_lr = 1e-5

    def training_step(self, batch, batch_idx):
        # Get the lq/hq image pair from the batch
        x, y = batch

        # Predict hq from lq using the model 
        pred = self.generator(x)
        
        # Compare prediction with lq 
        loss = torch.nn.functional.l1_loss(pred, y)

        # Log the training loss
        self.log("train_loss", torch.log(loss))

        # Log perceptual loss
        if self.use_vgg:
            vgg_loss = self.vgg(pred, y) * 1e-2
            self.log("vgg_loss", torch.log(vgg_loss))
            loss += vgg_loss
        
        # Log ssim loss
        if self.use_ssim:
            ssim_loss = -1 * self.ssim(pred, y)
            #ssim_loss = torch.acos(self.ssim(pred, y))
            self.log("ssim_loss", torch.log(ssim_loss))
            loss += ssim_loss

        # Log the learning rate
        #self.log('learning_rate', self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0])

        # Log a sample image every 10 batches 
        if batch_idx % 10 == 0:
            self.image(x, y, pred)

        # Return the loss for logging
        return loss
    
    def on_train_epoch_end(self, *arg, **kwargs):
        # Calculate eval
        pnsr, ssim = self.test(self.generator)
        self.log("pnsr", pnsr, on_epoch=True)
        self.log("ssim", ssim, on_epoch=True)
        print(f"PNSR {pnsr:.2f}, SSIM {ssim:.2f}")
        
        # Save generator
        torch.save(self.generator.state_dict(), f"ckpts/{self.name}_Latest.ckpt")

    def image(self, lq: torch.Tensor, hq: torch.Tensor, pred: torch.Tensor):
        # Save lq and pred for model stats
        self.lq = lq.cpu().detach().numpy()
        self.pred = pred.cpu().detach().numpy()

        # Pytorch has [b, 3, w, h], but cv2 expects [w, h, 3]
        lq = np.rollaxis(self.lq[0], 0, 3)
        hq = np.rollaxis(hq[0].cpu().detach().numpy(), 0, 3)
        pred = np.rollaxis(self.pred[0], 0, 3)

        # Lq (might) be small so we upsample it to hq so it aligns 
        lq = cv2.resize(lq, (hq.shape[0], hq.shape[1]), interpolation=cv2.INTER_NEAREST)

        # Join lq, hq, and pred horizontally
        shown = np.concatenate((lq, hq, pred), axis=1)

        # Resize to ensure quick saving
        shown = cv2.resize(shown, (hq.shape[0]*3, hq.shape[1]), interpolation=cv2.INTER_NEAREST)

        # Convert to an image data type
        shown = (np.clip(shown, 0, 1) * 255).astype(np.uint8)

        # Save and log
        cv2.imwrite("./latest.png", shown)
    
    def lr_lambda(self, epoch):
        lr = self.initial_lr * (0.5 ** (epoch // 50))
        # Ensure the learning rate doesn't go below the minimum threshold
        return max(lr, self.min_lr) / self.initial_lr

    def configure_optimizers(self):
        # Optimizer for backpropagation
        #optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, betas=(0.9, 0.99))
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6, T_mult=2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda)
        return [optimizer], [scheduler]

