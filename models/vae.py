#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianEncoder(object):
    def __init__(self, encode_dims, dropout):
        self.encoder = nn.ModuleDict({
            f'enc_{i}':nn.Linear(encode_dims[i],encode_dims[i+1]) 
            for i in range(len(encode_dims)-2)
        })
        self.dropout = nn.Dropout(p=dropout)
        self.fc_mu = nn.Linear(encode_dims[-2],encode_dims[-1])
        self.fc_logvar = nn.Linear(encode_dims[-2],encode_dims[-1])
    def encode(self, x):
        hid = x
        for i,layer in self.encoder.items():
            hid = F.relu(self.dropout(layer(hid)))
        mu, log_var = self.fc_mu(hid), self.fc_logvar(hid)
        return mu, log_var

class Decoder(object):
    def __init__(self, decode_dims, dropout):
        self.decoder = nn.ModuleDict({
            f'dec_{i}':nn.Linear(decode_dims[i],decode_dims[i+1])
            for i in range(len(decode_dims)-1)
        })
        self.dropout = nn.Dropout(p=dropout)
    def decode(self, z):
        hid = z
        for i,(_,layer) in enumerate(self.decoder.items()):
            hid = layer(hid)
            if i<len(self.decoder)-1:
                hid = F.relu(self.dropout(hid))
        return hid

# VAE model
class VAE(nn.Module, GaussianEncoder, Decoder):
    def __init__(self, encode_dims=[2000,1024,512,20],decode_dims=[20,1024,2000],dropout=0.0):
        super(VAE, self).__init__()
        GaussianEncoder.__init__(self, encode_dims, dropout)
        Decoder.__init__(self, decode_dims, dropout)
        self.fc1 = nn.Linear(encode_dims[-1], encode_dims[-1])

    def forward(self, x, collate_fn=None):
        mu, log_var = self.encode(x)
        _theta = self.reparameterize(mu, log_var)
        _theta = self.fc1(_theta) 
        if collate_fn!=None:
            theta = collate_fn(_theta)
        else:
            theta = _theta
        x_reconst = self.decode(theta)
        return x_reconst, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

class StickBreakingVAE(VAE):
    def __init__(self, encode_dims=[2000,1024,512,20],decode_dims=[20,1024,2000],dropout=0.0):
        super(StickBreakingVAE, self).__init__(encode_dims[:-1], decode_dims, dropout)
        self.fc1 = nn.Linear(encode_dims[-2], encode_dims[-1]-1)
        self.latent_dim = encode_dims[-1]

    def forward(self, x):
        mu, log_var = self.encode(x)
        eta = self.reparameterize(mu, log_var)
        eta = torch.sigmoid(self.fc1(eta))
        n_samples = eta.shape[0]
        eta = torch.hstack([eta,torch.ones((n_samples,1))])
        assert(eta.shape[1] == self.latent_dim)
        theta = torch.zeros((n_samples, self.latent_dim)) # stickbreaking
        theta[:,0] = eta[:,0]
        for k in range(1,self.latent_dim):
            theta[:,k] = eta[:,k] * (1-eta)[:,:k].prod(axis=1)
        x_reconst = self.decode(theta)
        return x_reconst, mu, log_var

class RecurrentStickBreakingVAE(VAE):
    def __init__(self, encode_dims=[2000,1024,512],decode_dims=[1024,2000],dropout=0.0):
        # encode_dims and decode_dims should not include n_topic dimension
        super(RecurrentStickBreakingVAE, self).__init__(encode_dims, decode_dims, dropout)
        self.rnn_sb = nn.LSTM(input_size=0, hidden_size=encode_dims[-1], num_layers=1)
        self.rnn_topic = nn.LSTM(input_size=0, hidden_size=decode_dims[0], num_layers=1)

    def init_input(self, n_samples, latent_dim):
        # empty placeholder input
        return torch.zeros(latent_dim, n_samples, 0)

    def forward(self, x, latent_dim):
        mu, log_var = self.encode(x)
        eta = self.reparameterize(mu, log_var) #(N,H)
        n_samples = eta.shape[0]
        hid, _ = self.rnn_sb(self.init_input(n_samples,latent_dim-1)) #(K-1, N, H)
        eta = torch.sigmoid((hid*eta).sum(-1).transpose(0,1)) #(N,K-1)
        eta = torch.hstack([eta,torch.ones((n_samples,1))])
        assert(eta.shape[1] == latent_dim)
        theta = torch.zeros((n_samples,latent_dim)) # stickbreaking
        theta[:,0] = eta[:,0]
        for k in range(1,latent_dim):
            theta[:,k] = eta[:,k] * (1-eta)[:,:k].prod(axis=1)
        x_reconst = self.decode(theta)
        return x_reconst, mu, log_var

    def decode(self, theta):
        # RSB-TF for unbounded number of topics
        t, _ = self.rnn_topic(self.init_input(theta.shape[0], theta.shape[1])) #(K, N, H)
        z = (t.transpose(0,2)*theta).sum(-1).transpose(0,1) #(N,H)
        return super().decode(z)

if __name__ == '__main__':
    model = VAE(encode_dims=[1024,512,256,20],decode_dims=[20,128,768,1024])
    model = model.cuda()
    inpt = torch.randn(234,1024).cuda()
    out,mu,log_var = model(inpt)
    print(out.shape)
    print(mu.shape)
