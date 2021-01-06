import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning.core.lightning import LightningModule
from Testing.Research.config.ConfigProvider import ConfigProvider
from pytorch_lightning import Trainer, seed_everything
from torch import optim
from Testing.Research.data_modules.MNISTDataModule import MNISTDataModule
from Testing.Research.data_modules.MyDataModule import MyDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from Testing.Research.config.paths import tb_logs_folder
from Testing.Research.config.paths import classifier_checkpoints_path
from Testing.Research.config.paths import vae_checkpoints_path
from Testing.Research.autoencoding.VAEFC_Lightning import VAEFC
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import os
import glob
from typing import List, Optional, Any
import torch
import pytorch_lightning as pl
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
from PIL import Image


class LatentSpaceClassifierLightning(LightningModule):
    def __init__(self, config, trained_vae, latent_dim):
        super(LatentSpaceClassifierLightning, self).__init__()
        self._config = config
        self._trained_vae = trained_vae
        self._trained_vae.eval()

        self.fc1 = nn.Linear(latent_dim, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, self._config.n_clusters)

        self._criterion = nn.CrossEntropyLoss()

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.val_confusion = pl.metrics.classification.ConfusionMatrix(num_classes=self._config.n_clusters)

        self.logger: Optional[TensorBoardLogger] = None

    def forward(self, x):
        decoded, mu_batch, logvar = self._trained_vae(x)  # TODO cancel decoder part evaluation

        x = F.relu(self.fc1(mu_batch))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # prediction = F.softmax(x)
        # return prediction
        log_probs = x
        return log_probs

    def training_step(self, batch, batch_index):
        if self._config.dataset == "toy":
            (orig_batch, noisy_batch), label_batch = batch
            # TODO put in the noise here and not in the dataset?
        elif self._config.dataset == "mnist":
            orig_batch, label_batch = batch
            orig_batch = orig_batch.reshape(-1, 28 * 28)
            noisy_batch = orig_batch
        else:
            raise ValueError("wrong config.dataset")

        log_probs = self.forward(orig_batch)
        loss = self._criterion(log_probs, label_batch)

        self.train_acc.update(log_probs, label_batch)
        self.log(r'Classifier loss\train', loss, on_step=True, on_epoch=True)

        return loss

    def training_step_end(self, outputs):
        return outputs

    def training_epoch_end(self, outs):
        tb = self.logger.experiment
        tb.add_scalars('Classifier accuracy', {"train": self.train_acc.compute()}, global_step=self.current_epoch)

    def validation_step(self, batch, batch_index):
        if self._config.dataset == "toy":
            (orig_batch, noisy_batch), label_batch = batch
            # TODO put in the noise here and not in the dataset?
        elif self._config.dataset == "mnist":
            orig_batch, label_batch = batch
            orig_batch = orig_batch.reshape(-1, 28 * 28)
            noisy_batch = orig_batch
        else:
            raise ValueError("wrong config.dataset")

        log_probs = self.forward(orig_batch)
        loss = self._criterion(log_probs, label_batch)

        self.val_acc.update(log_probs, label_batch)
        self.val_confusion.update(log_probs, label_batch)

        self.log(r'Classifier loss\validation', loss, on_step=True, on_epoch=True)
        return {"loss": loss, "labels": label_batch}

    def validation_step_end(self, outputs):
        return outputs

    def validation_epoch_end(self, outs):
        tb = self.logger.experiment

        # accuracy
        tb.add_scalars('Classifier accuracy', {"validation": self.val_acc.compute()}, global_step=self.current_epoch)

        # confusion matrix
        conf_mat = self.val_confusion.compute().detach().cpu().numpy().astype(np.int)
        df_cm = pd.DataFrame(
            conf_mat,
            index=np.arange(self._config.n_clusters),
            columns=np.arange(self._config.n_clusters))
        plt.figure()
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
        buf = io.BytesIO()
        # plt.show()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)
        plt.close('all')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self._config.learning_rate)
        return optimizer

    def test_step(self, batch, batch_index):
        tb = self.logger.experiment
        if self._config.dataset == "toy":
            (orig_batch, noisy_batch), label_batch = batch
            # TODO put in the noise here and not in the dataset?
        elif self._config.dataset == "mnist":
            orig_batch, label_batch = batch
            orig_batch = orig_batch.reshape(-1, 28 * 28)
            noisy_batch = orig_batch
        else:
            raise ValueError("wrong config.dataset")
        # recon_batch, mu_batch, logvar_batch = self._trained_vae(orig_batch)

        log_probs = self.forward(orig_batch)
        loss = self._criterion(log_probs, label_batch)

        self.test_acc.update(log_probs, label_batch)

        self.log('Classifier_test_accuracy_step', self.test_acc.compute(), on_step=True, on_epoch=False)
        return loss

    def test_step_end(self, outputs):
        # update and log
        # loss = self._criterion(outputs['preds'], outputs['target'])
        # self.log('train_metric', loss)
        return outputs

    def test_epoch_end(self, outs):
        tb = self.logger.experiment
        self.log('Classifier_test_accuracy_epoch', self.test_acc.compute())


def train_latent_classifier():
    config = ConfigProvider.get_config()
    seed_everything(config.random_seed)

    if config.dataset == "toy":
        datamodule = MyDataModule(config)
        latent_dim = config.latent_dim_toy
        enc_layer_sizes = config.enc_layer_sizes_toy + [latent_dim]
        dec_layer_sizes = [latent_dim] + config.dec_layer_sizes_toy
    elif config.dataset == "mnist":
        datamodule = MNISTDataModule(config)
        latent_dim = config.latent_dim_mnist
        enc_layer_sizes = config.enc_layer_sizes_mnist + [latent_dim]
        dec_layer_sizes = [latent_dim] + config.dec_layer_sizes_mnist
    else:
        raise ValueError("undefined config.dataset. Allowed are either 'toy' or 'mnist'")

    # model = VAEFC(config=config, encoder_layer_sizes=enc_layer_sizes, decoder_layer_sizes=dec_layer_sizes)

    last_vae = max(glob.glob(os.path.join(os.path.abspath(vae_checkpoints_path), r"**/*.ckpt"), recursive=True), key=os.path.getctime)
    trained_vae = VAEFC.load_from_checkpoint(last_vae, config=config, encoder_layer_sizes=enc_layer_sizes, decoder_layer_sizes=dec_layer_sizes)

    logger = TensorBoardLogger(save_dir=tb_logs_folder, name='Classifier', default_hp_metric=False)
    logger.hparams = config  # TODO only put here relevant stuff

    checkpoint_callback = ModelCheckpoint(dirpath=classifier_checkpoints_path)
    trainer = Trainer(deterministic=config.is_deterministic,
                      # auto_lr_find=config.auto_lr_find,
                      # log_gpu_memory='all',
                      # min_epochs=99999,
                      max_epochs=config.num_epochs,
                      default_root_dir=classifier_checkpoints_path,
                      logger=logger,
                      callbacks=[checkpoint_callback],
                      gpus=1
                      )
    # trainer.tune(model)

    classifier = LatentSpaceClassifierLightning(config, trained_vae, latent_dim=latent_dim)
    trainer.fit(classifier, datamodule=datamodule)
    best_model_path = checkpoint_callback.best_model_path
    print("done training classifier with lightning")
    print(f"best model path = {best_model_path}")
    return trainer


def run_trained_classifier(trainer):
    trainer.test()


if __name__ == "__main__":
    trainer = train_latent_classifier()
    run_trained_classifier(trainer)
