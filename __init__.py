import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from config import log_dir, train_dir, valid_dir, test_dir, batch_size
from src.datamodule.amass_data_module import AMASSDataModule
from src.ligtning_module.unet_1d import UNet1D

if __name__ == '__main__':
    # # Initialize the TransformerVAE model
    # model = TransformerVAE(input_dim=156 + 52, output_dim=156, latent_dim=256,
    #                        num_frames=frames_per_sample)  # Adjust as needed
    model = UNet1D(in_channels=156 + 52, out_channels=156)

    # Configure the Trainer
    logger = CSVLogger(save_dir=log_dir, name="TransformerVAE")
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",  # Monitor training loss for saving the best model
        save_top_k=1,
        mode="min",
        filename="transformer_vae-{epoch:02d}-{train_loss:.4f}",
    )
    trainer = pl.Trainer(
        max_epochs=10,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=1,  # Print logs for every batch
        enable_progress_bar=True,  # Progress will be shown in the notebook
    )

    data_module = AMASSDataModule(train_dir, valid_dir, test_dir, batch_size=batch_size)

    # Assuming you have a DataModule instance called `data_module`
    trainer.fit(model, data_module)
