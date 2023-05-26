from typing import Any, List

import torch
from torch.optim import Adam
from pytorch_lightning import LightningModule


class GANLitModule(LightningModule):
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

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def get_noise(self, z_dim, n_samples):
        return torch.randn(n_samples, z_dim, device=self.generator.device)

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
        # flatten the input images
        x, y = batch
        real = x.view(len(x), -1) # len(x) = x.shape[0] = batch_size

        noise = self.get_noise(z_dim=self.hparams.z_dim, n_samples=len(x))
        fake = self.generator(noise)

        if optimizer_idx == 0:
            gen_loss = self.get_gen_loss(fake)
            self.log('gen_loss', gen_loss, on_step=False, on_epoch=True, prog_bar=True)
            return gen_loss
        elif optimizer_idx == 1:
            disc_loss = self.get_disc_loss(real, fake)
            self.log('disc_loss', disc_loss, on_step=False, on_epoch=True, prog_bar=True)
            return disc_loss

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def configure_optimizers(self):
        gen_opt = Adam(self.generator.parameters(), lr=self.hparams.lr)
        disc_opt = Adam(self.discriminator.parameters(), lr=self.hparams.lr)
        return [gen_opt, disc_opt], []


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import pyrootutils

    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs/model")
    # output_path = path / "outputs"
    # print(f"path: {path}")
    # print(f'config path: {config_path}')
    # print(f'output path: {output_path}')

    @hydra.main(version_base="1.3", config_path=config_path, config_name="gan.yaml")
    def main(cfg: DictConfig):
        gan = hydra.utils.instantiate(cfg)
        print(gan.device)

    main()
    