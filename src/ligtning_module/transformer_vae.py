import pytorch_lightning as pl
import torch
from torch import nn


class TransformerVAE(pl.LightningModule):
    def __init__(self, input_dim, output_dim, latent_dim, num_frames, nhead=4, num_layers=2, hidden_dim=256, lr=1e-3):
        """
        Transformer-based Variational Autoencoder with global latent space.

        Args:
            input_dim (int): Number of joints (spatial dimension).
            output_dim (int): Number of joints in the output sequence.
            latent_dim (int): Dimensionality of the global latent space.
            num_frames (int): Number of frames in the temporal dimension.
            nhead (int): Number of attention heads in the Transformer.
            num_layers (int): Number of Transformer layers.
            hidden_dim (int): Hidden dimension for Transformer layers.
            lr (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        self.lr = lr

        # Encoder: maps input to global latent distribution
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.encoder_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 2,
                activation="relu"
            ),
            num_layers=num_layers
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_pre = nn.Linear(latent_dim, hidden_dim)
        # Decoder: maps global latent space to full sequence
        self.decoder_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 2,
                activation="relu"
            ),
            num_layers=num_layers
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def encode(self, x):
        # Flatten temporal frames into a sequence
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # (BATCH, N_FRAMES, HIDDEN) -> (N_FRAMES, BATCH, HIDDEN)
        x = self.encoder_transformer(x)
        x = x.mean(dim=0)  # Aggregate across frames -> (BATCH, HIDDEN)
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        return mu, logvar

    def decode(self, z, seq_length):
        # Expand latent vector to match sequence length
        z = z.unsqueeze(1).repeat(1, seq_length, 1)  # (BATCH, 1, LATENT) -> (BATCH, N_FRAMES, LATENT)
        z = self.decoder_pre(z)  # (BATCH, N_FRAMES, LATENT) -> (BATCH, N_FRAMES, HIDDEN)
        z = z.permute(1, 0, 2)  # (BATCH, N_FRAMES, HIDDEN) -> (N_FRAMES, BATCH, HIDDEN)
        z = self.decoder_transformer(z, z)
        z = z.permute(1, 0, 2)  # Back to (BATCH, N_FRAMES, LATENT)
        return self.decoder(z)

    def sample_latent(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        mu, logvar = self.encode(x)
        z = self.sample_latent(mu, logvar)

        # Decode
        x_recon = self.decode(z, x.size(1))
        return x_recon, mu, logvar

    def vae_loss(self, x_recon, x, mu, logvar):
        # Reconstruction loss
        recon_loss = nn.MSELoss()(x_recon, x)

        # KL Divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def training_step(self, batch, batch_idx):
        x_masked = torch.cat((batch['masked_poses'], batch['mask']), dim=-1)
        y = batch['poses']

        x_recon, mu, logvar = self.forward(x_masked)
        loss = self.vae_loss(x_recon, y, mu, logvar)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
