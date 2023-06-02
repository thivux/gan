from typing import Any, Optional

import numpy as np

import torchvision.utils as vutils
from torchvision.transforms import ToPILImage

import torch
from torch.optim import Adam
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric


class DCGANLitModule(LightningModule):
    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        lr: float, 
        z_dim: int = 64,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['generator', 'discriminator'])

        self.generator = generator
        self.discriminator = discriminator

        # metric
        self.gen_metric = MeanMetric()
        self.disc_metric = MeanMetric()

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def on_train_start(self) -> None:
        # cache fixed noise for visualization
        n_samples = 16
        self.fixed_noise = self.get_noise(n_samples=n_samples, z_dim=self.hparams.z_dim, fixed=True)

        # log sample real images for visualization
        train_loader = self.trainer.train_dataloader
        x, _ = next(iter(train_loader))
        samples = x[:n_samples]
        grid = vutils.make_grid(samples, nrow=int(np.sqrt(n_samples)), normalize=True)
        self.logger.log_image(key='real images', images=[ToPILImage()(grid)])

    def get_noise(self, n_samples, z_dim, fixed=False):
        if fixed:
            torch.manual_seed(7749)
        return torch.randn(n_samples, z_dim, device=self.device)

    def get_disc_loss(self, real, fake):
        fake_pred = self.discriminator(fake.detach()) 
        real_pred = self.discriminator(real)

        fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
        real_loss = self.criterion(real_pred, torch.ones_like(real_pred))
        disc_loss = (fake_loss + real_loss) / 2
        return disc_loss

    def get_gen_loss(self, fake):
        fake_pred = self.discriminator(fake)
        gen_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))
        return gen_loss
    
    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        real = batch[0]

        noise = self.get_noise(z_dim=self.hparams.z_dim, n_samples=len(real))
        fake = self.generator(noise)

        if optimizer_idx == 0:
            gen_loss = self.get_gen_loss(fake)
            self.gen_metric(gen_loss)
            self.log('gen_loss', self.gen_metric, on_step=False, on_epoch=True, prog_bar=True)
            return gen_loss
        elif optimizer_idx == 1:
            disc_loss = self.get_disc_loss(real, fake)
            self.disc_metric(disc_loss)
            self.log('disc_loss', self.disc_metric, on_step=False, on_epoch=True, prog_bar=True)
            return disc_loss

    def on_train_epoch_end(self) -> None:
        fake = self.generator(self.fixed_noise)
        grid = vutils.make_grid(fake, nrow=int(np.sqrt(len(fake))), normalize=True)
        self.logger.log_image(key='fake images', images=[ToPILImage()(grid)], step=self.current_epoch)

    def test_step(self, *args: Any, **kwargs: Any):
        pass

    def configure_optimizers(self):
        gen_opt = Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        disc_opt = Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        return [gen_opt, disc_opt], []


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    import pyrootutils

    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs/model")

    @hydra.main(version_base="1.3", config_path=config_path, config_name="gan.yaml")
    def main(cfg: DictConfig):
        dcgan = hydra.utils.instantiate(cfg)
        noise = dcgan.get_noise(128, 64)
        print(dcgan(noise).shape)

    main()
    