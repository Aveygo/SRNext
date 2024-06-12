import torch, lightning as L, numpy as np, cv2
from utils.perceptual_loss_any import PerceptualLoss
from utils.testing.evaluate import Evaluate
import torch.nn.functional as F

class GAN(L.LightningModule):
    def __init__(self, generator:torch.nn.Module, discriminator:torch.nn.Module):
        super().__init__()
        self.test = Evaluate()
        self.generator = generator
        self.discriminator = discriminator
        
        self.huber_loss = torch.nn.HuberLoss()
        self.gan_loss = torch.nn.BCEWithLogitsLoss().cuda()

        self.automatic_optimization = False
        self.vgg = PerceptualLoss(discriminator)

        self.g_name = f"{type(self.generator).__name__}_{__class__.__name__}"
        self.d_name = f"{type(self.discriminator).__name__}_{__class__.__name__}"

    def forward(self, x):
        return self.generator(x)
    
    def positive_loss(self, left, right):
        return self.gan_loss(left - right.mean(0, keepdim=True), torch.ones_like(right))
    
    def negative_loss(self, left, right):
        return self.gan_loss(left - right.mean(0, keepdim=True), torch.zeros_like(right))

    def step_g(self, batch):
        crushed, original = batch

        #print(crushed.shape)
        predict = self.generator(crushed)

        pixel_loss = self.huber_loss(predict, original)
        
        #d_input_left =  torch.cat([predict,  F.interpolate(crushed, scale_factor=4, mode="bilinear")], dim=1)
        #d_input_right = torch.cat([original, F.interpolate(crushed, scale_factor=4, mode="bilinear")], dim=1)

        vgg_loss = self.vgg(predict, original) *1e-1

        gan_loss = self.positive_loss(
            self.discriminator(predict),
            self.discriminator(original).detach()
        ) * 1e-1

        self.log("Pixel Loss", pixel_loss, on_step=True)
        self.log("Gan Loss", gan_loss, on_step=True)
        self.log("Percept. Loss", vgg_loss, on_step=True)

        return crushed, original, predict, pixel_loss + gan_loss + vgg_loss

    def step_d(self, batch):
        crushed, original = batch
        predict = self.generator(crushed)
        
        pred_real = self.discriminator(original)
        pred_fake = self.discriminator(predict.detach())
        #pred_real = self.discriminator(torch.cat([original,         F.interpolate(crushed, scale_factor=4, mode="bilinear")], dim=1))
        #pred_fake = self.discriminator(torch.cat([predict.detach(), F.interpolate(crushed, scale_factor=4, mode="bilinear")], dim=1))

        real_loss = self.positive_loss(pred_real, pred_fake)
        fake_loss = self.negative_loss(pred_fake, pred_real)

        self.log("Real Loss", real_loss, on_step=True)
        self.log("Fake Loss", fake_loss, on_step=True)

        return crushed, original, predict, (real_loss + fake_loss)/2

    def training_step(self, batch, batch_idx):
        
        g_opt, d_opt = self.optimizers()

        g_opt.zero_grad()
        crushed, original, predict, loss = self.step_g(batch)
        loss.backward()
        g_opt.step()

        d_opt.zero_grad()
        crushed, original, predict, loss = self.step_d(batch)
        loss.backward()
        d_opt.step()
        
        if batch_idx % 10 == 0:
            self.image(crushed, original, predict)

        return loss
    
    def on_train_epoch_end(self, *arg, **kwargs):
        # Calculate eval
        pnsr, ssim = self.test(self.generator)
        self.log("pnsr", pnsr, on_epoch=True)
        self.log("ssim", ssim, on_epoch=True)

        # When the epoch ends, save the models
        torch.save(self.generator.state_dict(), f"ckpts/{self.g_name}_Latest.ckpt")
        torch.save(self.discriminator.state_dict(), f"ckpts/{self.d_name}_Latest.ckpt")

    def image(self, lq: torch.Tensor, hq: torch.Tensor, pred: torch.Tensor):

        # Pytorch has [b, 3, w, h], but cv2 expects [w, h, 3]
        lq = np.rollaxis(lq[0].cpu().detach().numpy(), 0, 3)
        hq = np.rollaxis(hq[0].cpu().detach().numpy(), 0, 3)
        pred = np.rollaxis(pred[0].cpu().detach().numpy(), 0, 3)

        # Lq might be small so we upsample it to hq so it aligns 
        lq = cv2.resize(lq, (hq.shape[0], hq.shape[1]), interpolation=cv2.INTER_NEAREST)

        # Join lq, hq, and pred horizontally
        shown = np.concatenate((lq, hq, pred), axis=1)

        # Resize to ensure quick saving
        shown = cv2.resize(shown, (hq.shape[0]*3, hq.shape[1]), interpolation=cv2.INTER_NEAREST)

        # Convert to an image data type
        shown = (np.clip(shown, 0, 1) * 255).astype(np.uint8)

        # Save and log
        cv2.imwrite("./latest.png", shown)
        
    def configure_optimizers(self):
        # Optimizer for backpropagation
        return [
            torch.optim.AdamW(self.generator.parameters(), lr=1e-6, betas=(0.9, 0.95), weight_decay=0.05, amsgrad=True),
            torch.optim.AdamW(self.discriminator.parameters(), lr=1e-6, betas=(0.9, 0.95), weight_decay=0.05, amsgrad=True)
        ], []

