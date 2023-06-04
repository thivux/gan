from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms
from torch import Tensor

class FASHIONMNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        img_dims: int = 1,
        img_size: int = 28, 
        batch_size: int = 128,
        n_classes: int = 10,
        num_workers: int = 0,
        transform: Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.transforms = transform

    def prepare_data(self):
        FashionMNIST(self.hparams.data_dir, train=True, download=True)
        FashionMNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if stage == "fit" or stage is None:
            self.data_train = FashionMNIST(self.hparams.data_dir, train=True, transform=self.transforms)

        if stage == "test" or stage is None:
            self.data_test = FashionMNIST(self.hparams.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
    
    def get_n_classes(self):
        return self.hparams.n_classes

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "fashion_mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
    _.setup()
    features, labels = next(iter(_.train_dataloader()))
    print(features.shape)
    print(len(labels))
