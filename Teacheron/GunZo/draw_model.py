import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
from torch.nn.functional import grid_sample, affine_grid

"""
One-shot generalization in deep generative models
"""

class DRAWModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.T = params['T'] # Number of steps
        self.A = params['A'] # Width
        self.B = params['B'] # Height
        self.C = params['C'] # Height
        self.z_size = params['z_size'] 
        self.read_N = params['read_N']
        self.write_N = params['write_N']
        self.enc_size = params['enc_size']
        self.dec_size = params['dec_size']
        self.device = params['device']
        self.channel = params['channel']

        # Stores the generated image for each time step.
        self.cs = [0] * self.T
        
        # To store appropriate values used for calculating the latent loss (KL-Divergence loss)
        self.logsigmas = [0] * self.T
        self.sigmas = [0] * self.T
        self.mus = [0] * self.T

        self.encoder = nn.LSTMCell(self.read_N*self.read_N*self.read_N*self.channel + self.dec_size, self.enc_size)

        # To get the mean and standard deviation for the distribution of z.
        self.fc_mu = nn.Linear(self.enc_size, self.z_size)
        self.fc_sigma = nn.Linear(self.enc_size, self.z_size)

        self.decoder = nn.LSTMCell(self.z_size, self.dec_size)

#         self.fc_write = nn.Linear(self.dec_size, self.write_N*self.write_N*self.channel)
        self.fc_w1 = nn.Linear(self.dec_size, 4)
        self.fc_w2 = nn.Linear(self.dec_size, self.write_N*self.write_N*self.write_N*self.channel)
        
        # To get the attention parameters (3 in total)
        self.fc_read = nn.Linear(self.dec_size, 4)
        self.fc_attention = nn.Linear(self.dec_size, 5)

    def forward(self, x):
        self.batch_size = x.size(0)

        # requires_grad should be set True to allow backpropagation of the gradients for training.
        h_enc_prev = torch.zeros(self.batch_size, self.enc_size, requires_grad=True, device=self.device)
        h_dec_prev = torch.zeros(self.batch_size, self.dec_size, requires_grad=True, device=self.device)

        enc_state = torch.zeros(self.batch_size, self.enc_size, requires_grad=True, device=self.device)
        dec_state = torch.zeros(self.batch_size, self.dec_size, requires_grad=True, device=self.device)

        for t in range(self.T):
            c_prev = torch.zeros(self.batch_size, self.A*self.B*self.C*self.channel, requires_grad=True, device=self.device) if t == 0 else self.cs[t-1]
            # Read function
            r_t = self.read(x, h_dec_prev, (self.batch_size, self.channel, self.read_N, self.read_N, self.read_N)) # Use h_dec_prev to get affine transformation matrix
            # Encoder LSTM
            h_enc, enc_state = self.encoder(torch.cat((r_t, h_dec_prev), dim=1), (h_enc_prev, enc_state))
            # Sample
            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(h_enc)
            # Decoder LSTM
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))

            # Write and adding canvas functions
            self.cs[t] = c_prev + self.write(h_dec, (self.batch_size, self.channel, self.A, self.B, self.C))

            h_enc_prev = h_enc
            h_dec_prev = h_dec

    def read(self, x, h_dec_prev, out_dims):
        # params (s, x, y)
        params = self.fc_read(h_dec_prev)
    
        theta = torch.zeros(3,4).repeat(x.shape[0], 1, 1).to(x.device)
        # set scaling
        theta[:, 0, 0] = theta[:, 1, 1] = theta[:, 2, 2]= params[:,0]
        # set translation
        theta[:, :, -1] = params[:, 1:]
      
        grid = affine_grid(theta, torch.Size(out_dims))        
        out = grid_sample(x.view(x.size(0), 1, 16, 16, 16), grid)
        out = out.view(out.size(0), -1)
        
        return out # size (64, 125)
    
    def write(self, h_dec, out_dims):
        # params (s, x, y)
        params = self.fc_w1(h_dec)
        x = self.fc_w2(h_dec)

        theta = torch.zeros(3, 4).repeat(x.size(0), 1, 1).to(x.device)
        # set scaling
        theta[:, 0, 0] = theta[:, 1, 1] = theta[:, 2, 2] = 1 / (params[:, 0] + 1e-9)
        # set translation
        theta[:, :, -1] = - params[:, 1:] / (params[:, 0].view(-1, 1) + 1e-9)

        grid = affine_grid(theta, torch.Size(out_dims))
        
        out = grid_sample(x.view(x.size(0), 1, 5, 5, 5), grid)
        out = out.view(out.size(0), -1)

        return out # size should be (64, 4096 of 1x16x16x16)
    

    def sampleQ(self, h_enc):
        e = torch.randn(self.batch_size, self.z_size, device=self.device)

        # Equation 1.
        mu = self.fc_mu(h_enc)
        # Equation 2.
        log_sigma = self.fc_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        
        z = mu + e * sigma

        return z, mu, log_sigma, sigma


    def loss(self, x):
        self.forward(x)

        criterion = nn.BCELoss()
        x_recon = torch.sigmoid(self.cs[-1])
        # Reconstruction loss.
        # Only want to average across the mini-batch, hence, multiply by the image dimensions.
        Lx = criterion(x_recon, x) * self.A * self.B * self.C * self.channel
        # Latent loss.
        Lz = 0

        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]

            kl_loss = 0.5*torch.sum(mu_2 + sigma_2 - 2*logsigma, 1) - 0.5*self.T
            Lz += kl_loss

        Lz = torch.mean(Lz)
        net_loss = Lx + Lz

        return net_loss

    def generate(self, num_output):
        self.batch_size = num_output
        h_dec_prev = torch.zeros(num_output, self.dec_size, device=self.device)
        dec_state = torch.zeros(num_output, self.dec_size  , device=self.device)

        for t in range(self.T):
            c_prev = torch.zeros(self.batch_size, self.A*self.B*self.C*self.channel, device=self.device) if t == 0 else self.cs[t-1]
            z = torch.randn(self.batch_size, self.z_size, device=self.device)
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            self.cs[t] = c_prev + self.write(h_dec, (self.batch_size, self.channel, self.A, self.B, self.C))
            h_dec_prev = h_dec

        imgs = []

        for img in self.cs:
            # The image dimension is A x B x C
            print(img.size())
            img = img.view(-1, self.channel, self.A, self.B, self.C)
            imgs.append(vutils.make_grid(torch.sigmoid(img).detach().cpu(), nrow=int(np.sqrt(int(num_output))), padding=1, normalize=True, pad_value=1))

        return imgs