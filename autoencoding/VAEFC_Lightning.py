import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Any
from pytorch_lightning.core.lightning import LightningModule
from Testing.Research.config.ConfigProvider import ConfigProvider
from pytorch_lightning import Trainer, seed_everything
from torch import optim
import os
from pytorch_lightning.loggers import TensorBoardLogger
# import tfmpl
import matplotlib.pyplot as plt
import matplotlib
from Testing.Research.data_modules.MyDataModule import MyDataModule
from Testing.Research.data_modules.MNISTDataModule import MNISTDataModule
import torchvision
from Testing.Research.config.paths import tb_logs_folder
from Testing.Research.config.paths import vae_checkpoints_path
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


# TODO use better tensorboard
# @tfmpl.figure_tensor
# def draw_scatter(scaled, colors):
#     """Draw scatter plots. One for each color."""
#     figs = tfmpl.create_figures(len(colors), figsize=(4,4))
#     for idx, f in enumerate(figs):
#         ax = f.add_subplot(111)
#         ax.axis('off')
#         ax.scatter(scaled[:, 0], scaled[:, 1], c=colors[idx])
#         f.tight_layout()
#
#     return figs


class VAEFC(LightningModule):
    # see https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
    # for possible upgrades, see https://arxiv.org/pdf/1602.02282.pdf
    # https://stats.stackexchange.com/questions/332179/how-to-weight-kld-loss-vs-reconstruction-loss-in-variational-auto-encoder
    def __init__(self, encoder_layer_sizes: List, decoder_layer_sizes: List, config):
        super(VAEFC, self).__init__()
        self._config = config
        self.logger: Optional[TensorBoardLogger] = None
        self.save_hyperparameters()

        assert len(encoder_layer_sizes) >= 3, "must have at least 3 layers (2 hidden)"
        # encoder layers
        self._encoder_layers = nn.ModuleList()
        for i in range(1, len(encoder_layer_sizes) - 1):
            enc_layer = nn.Linear(encoder_layer_sizes[i - 1], encoder_layer_sizes[i])
            self._encoder_layers.append(enc_layer)

        # predict mean and covariance vectors
        self._mean_layer = nn.Linear(encoder_layer_sizes[
                                         len(encoder_layer_sizes) - 2],
                                     encoder_layer_sizes[len(encoder_layer_sizes) - 1])
        self._logvar_layer = nn.Linear(encoder_layer_sizes[
                                           len(encoder_layer_sizes) - 2],
                                       encoder_layer_sizes[len(encoder_layer_sizes) - 1])

        # decoder layers
        self._decoder_layers = nn.ModuleList()
        for i in range(1, len(decoder_layer_sizes)):
            dec_layer = nn.Linear(decoder_layer_sizes[i - 1], decoder_layer_sizes[i])
            self._decoder_layers.append(dec_layer)

        self._recon_function = nn.MSELoss(reduction='mean')

    def _encode(self, x):
        for i in range(len(self._encoder_layers)):
            layer = self._encoder_layers[i]
            x = F.relu(layer(x))

        mean_output = self._mean_layer(x)
        logvar_output = self._logvar_layer(x)
        return mean_output, logvar_output

    def _reparametrize(self, mu, logvar):
        if not self.training:
            return mu
        std = logvar.mul(0.5).exp_()
        if std.is_cuda:
            eps = torch.FloatTensor(std.size()).cuda().normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        reparameterized = eps.mul(std).add_(mu)
        return reparameterized

    def _decode(self, z):
        for i in range(len(self._decoder_layers) - 1):
            layer = self._decoder_layers[i]
            z = F.relu((layer(z)))

        decoded = self._decoder_layers[len(self._decoder_layers) - 1](z)
        # decoded = F.sigmoid(self._decoder_layers[len(self._decoder_layers)-1](z))
        return decoded

    def _loss_function(self, recon_x, x, mu, logvar, reconstruction_function):
        """
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """
        binary_cross_entropy = reconstruction_function(recon_x, x)  # mse loss TODO see if mse or cross entropy
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kld = torch.sum(kld_element).mul_(-0.5)
        # KL divergence Kullback–Leibler divergence, regularization term for VAE
        # It is a measure of how different two probability distributions are different from each other.
        # We are trying to force the distributions closer while keeping the reconstruction loss low.
        # see https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73

        # read on weighting the regularization term here:
        # https://stats.stackexchange.com/questions/332179/how-to-weight-kld-loss-vs-reconstruction-loss-in-variational
        # -auto-encoder
        return binary_cross_entropy + kld * self._config.regularization_factor

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
        noisy_batch = noisy_batch.view(noisy_batch.size(0), -1)

        recon_batch, mu, logvar = self.forward(noisy_batch)

        loss = self._loss_function(
            recon_batch,
            orig_batch, mu, logvar,
            reconstruction_function=self._recon_function
        )
        # self.logger.experiment.add_scalars("losses", {"train_loss": loss})
        tb = self.logger.experiment
        tb.add_scalars("losses", {"train_loss": loss}, global_step=self.current_epoch)
        # self.logger.experiment.add_scalar("train_loss", loss, self.current_epoch)
        if batch_index == len(self.train_dataloader()) - 2:
            # https://pytorch.org/docs/stable/_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_embedding
            # noisy_batch = noisy_batch.detach()
            # recon_batch = recon_batch.detach()
            # last_batch_plt = matplotlib.figure.Figure()  # read https://github.com/wookayin/tensorflow-plot
            # ax = last_batch_plt.add_subplot(1, 1, 1)
            # ax.scatter(orig_batch[:, 0], orig_batch[:, 1], label="original")
            # ax.scatter(noisy_batch[:, 0], noisy_batch[:, 1], label="noisy")
            # ax.scatter(recon_batch[:, 0], recon_batch[:, 1], label="reconstructed")
            # ax.legend(loc="upper left")
            # self.logger.experiment.add_figure(f"original last batch, epoch {self.current_epoch}", last_batch_plt)
            # tb.add_embedding(orig_batch, global_step=self.current_epoch, metadata=label_batch)
            pass
        self.logger.experiment.flush()
        return loss

    def validation_step(self, batch, batch_idx):
        if self._config.dataset == "toy":
            (orig_batch, noisy_batch), label_batch = batch
            # TODO put in the noise here and not in the dataset?
        elif self._config.dataset == "mnist":
            orig_batch, label_batch = batch
            orig_batch = orig_batch.reshape(-1, 28 * 28)
            noisy_batch = orig_batch
        else:
            raise ValueError("invalid dataset")

        noisy_batch = noisy_batch.view(noisy_batch.size(0), -1)

        recon_batch, mu, logvar = self.forward(noisy_batch)

        loss = self._loss_function(
            recon_batch,
            orig_batch, mu, logvar,
            reconstruction_function=self._recon_function
        )

        tb = self.logger.experiment
        # can probably speed up training by waiting for epoch end for data copy from gpu
        # see https://sagivtech.com/2017/09/19/optimizing-pytorch-training-code/
        tb.add_scalars("losses", {"val_loss": loss}, global_step=self.current_epoch)
        if batch_idx == len(self.val_dataloader()) - 2:
            orig_batch -= orig_batch.min()
            orig_batch /= orig_batch.max()
            recon_batch -= recon_batch.min()
            recon_batch /= recon_batch.max()

            orig_grid = torchvision.utils.make_grid(orig_batch.view(-1, 1, 28, 28))
            val_recon_grid = torchvision.utils.make_grid(recon_batch.view(-1, 1, 28, 28))

            tb.add_image("original_val", orig_grid, global_step=self.current_epoch)
            tb.add_image("reconstruction_val", val_recon_grid, global_step=self.current_epoch)
            pass

        outputs = {"val_loss": loss, "recon_batch": recon_batch, "label_batch": label_batch,
                   "label_img": orig_batch.view(-1, 1, 28, 28)}
        return outputs

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        first_batch_dict = outputs[-1]
        self.log(name="VAEFC_val_loss_epoch_end", value={"val_loss": first_batch_dict["val_loss"]})

    def test_step(self, batch, batch_idx):
        if self._config.dataset == "toy":
            (orig_batch, noisy_batch), label_batch = batch
            # TODO put in the noise here and not in the dataset?
        elif self._config.dataset == "mnist":
            orig_batch, label_batch = batch
            orig_batch = orig_batch.reshape(-1, 28 * 28)
            noisy_batch = orig_batch
        else:
            raise ValueError("invalid dataset")
        noisy_batch = noisy_batch.view(noisy_batch.size(0), -1)

        recon_batch, mu, logvar = self.forward(noisy_batch)

        loss = self._loss_function(
            recon_batch,
            orig_batch, mu, logvar,
            reconstruction_function=self._recon_function
        )

        tb = self.logger.experiment
        tb.add_scalars("losses", {"test_loss": loss}, global_step=self.global_step)

        return {"test_loss": loss, "mus": mu, "labels": label_batch, "images": orig_batch}

    def test_epoch_end(self, outputs: List):
        tb = self.logger.experiment

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log(name="test_epoch_end", value={"test_loss_avg": avg_loss})

        tb.add_embedding(
            mat=torch.cat([o["mus"] for o in outputs]),
            metadata=torch.cat([o["labels"] for o in outputs]).detach().cpu().numpy(),
            label_img=torch.cat([o["images"] for o in outputs]).view(-1, 1, 28, 28),
            global_step=self.global_step,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self._config.learning_rate)
        return optimizer

    def forward(self, x):
        mu, logvar = self._encode(x)
        z = self._reparametrize(mu, logvar)
        decoded = self._decode(z)
        return decoded, mu, logvar


def train_vae():
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

    model = VAEFC(config=config, encoder_layer_sizes=enc_layer_sizes, decoder_layer_sizes=dec_layer_sizes)

    logger = TensorBoardLogger(save_dir=tb_logs_folder, name='VAEFC', default_hp_metric=False)
    logger.hparams = config  # TODO only put here relevant stuff

    checkpoint_callback = ModelCheckpoint(dirpath=vae_checkpoints_path)
    trainer = Trainer(deterministic=config.is_deterministic,
                      # auto_lr_find=config.auto_lr_find,
                      # log_gpu_memory='all',
                      # min_epochs=99999,
                      max_epochs=config.num_epochs,
                      default_root_dir=vae_checkpoints_path,
                      logger=logger,
                      callbacks=[checkpoint_callback],
                      gpus=1
                      )
    # trainer.tune(model)
    trainer.fit(model, datamodule=datamodule)
    best_model_path = checkpoint_callback.best_model_path
    print("done training vae with lightning")
    print(f"best model path = {best_model_path}")
    return trainer


def run_trained_vae(trainer):
    # https://pytorch-lightning.readthedocs.io/en/latest/test_set.html
    # (1) load the best checkpoint automatically (lightning tracks this for you)
    trainer.test()

    # (2) don't load a checkpoint, instead use the model with the latest weights
    # trainer.test(ckpt_path=None)

    # (3) test using a specific checkpoint
    # trainer.test(ckpt_path='/path/to/my_checkpoint.ckpt')

    # (4) test with an explicit model (will use this model and not load a checkpoint)
    # trainer.test(model)


if __name__ == "__main__":
    trainer = train_vae()
    run_trained_vae(trainer)
