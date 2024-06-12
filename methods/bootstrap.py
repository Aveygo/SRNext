import torch, lightning as L, numpy as np, cv2
from utils.testing.evaluate import Evaluate
from utils.perceptual_loss import VGGPerceptualLoss

class Bootstrap(L.LightningModule):
    def __init__(self, generator:torch.nn.Module, bootstrapper:torch.nn.Module, use_vgg:bool=False):
        super().__init__()
        self.test = Evaluate()

        self.generator = generator
        self.bootstrapper = bootstrapper

        self.name = f"{type(self.generator).__name__}_{__class__.__name__}"

        pnsr, ssim = self.test(self.bootstrapper)
        print(f"Bootstrapper - pnsr: {pnsr:.4f} ssim: {ssim:.4f}")

        self.use_vgg = use_vgg
        if use_vgg:
            self.vgg = VGGPerceptualLoss()

    def training_step(self, batch, batch_idx):
        # Get the lq/hq image pair from the batch
        x, y = batch

        # Predict hq from lq using the model 
        pred = self.generator(x)

        # Get the bootstraped hq from lq using the bootstrapper
        bs = self.bootstrapper(x)
        
        # Compare prediction with the bootstraped hq and calculate the mse loss
        loss = torch.nn.functional.l1_loss(pred, bs)

        # Mix the ground truth into the calculation
        loss += torch.nn.functional.l1_loss(pred, y)

        # Log the training loss
        self.log("train_loss", loss)

        if self.use_vgg:
            vgg_loss = self.vgg(pred, bs)
            self.log("vgg_loss", vgg_loss)
            loss += vgg_loss

        # Log a sample image every 10 batches 
        if batch_idx % 10 == 0:
            self.image(x, y, bs, pred)

        # Return the loss for logging
        return loss
    
    def on_train_epoch_end(self, *arg, **kwargs):
        # Calculate eval
        pnsr, ssim = self.test(self.generator)
        self.log("pnsr", pnsr, on_epoch=True)
        self.log("ssim", ssim, on_epoch=True)

        # When the epoch ends, save the model
        torch.save(self.generator.state_dict(), f"ckpts/{self.name}_Latest.ckpt")

    def image(self, lq: torch.Tensor, hq: torch.Tensor, boot: torch.Tensor, pred: torch.Tensor):

        # Pytorch has [b, 3, w, h], but cv2 expects [w, h, 3]
        lq = np.rollaxis(lq[0].cpu().detach().numpy(), 0, 3)
        hq = np.rollaxis(hq[0].cpu().detach().numpy(), 0, 3)
        boot = np.rollaxis(boot[0].cpu().detach().numpy(), 0, 3)
        pred = np.rollaxis(pred[0].cpu().detach().numpy(), 0, 3)

        # Lq might be small so we upsample it to hq so it aligns 
        lq = cv2.resize(lq, (hq.shape[0], hq.shape[1]), interpolation=cv2.INTER_NEAREST)

        # Join lq, hq, boot, and pred horizontally
        shown = np.concatenate((lq, hq, boot, pred), axis=1)

        # Resize to ensure quick saving
        shown = cv2.resize(shown, (hq.shape[0]*4, hq.shape[1]), interpolation=cv2.INTER_NEAREST)

        # Convert to an image data type
        shown = (np.clip(shown, 0, 1) * 255).astype(np.uint8)

        # Save and log
        cv2.imwrite("./latest.png", shown)
        
    def configure_optimizers(self):
        # Optimizer for backpropagation
        return torch.optim.AdamW(self.generator.parameters(), lr=1e-4)

