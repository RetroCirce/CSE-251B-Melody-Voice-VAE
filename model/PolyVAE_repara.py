# PolyVAE
import torch
from torch import Size, nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import Parameter
from torch.distributions import Normal
import numpy as np

class PolyVAE_repara(nn.Module):
    def __init__(
        self, input_dims = 90, 
        hidden_dims = 512, z_dims = 1024, 
        seq_len = 320, beat_num = 20, tick_num = 16, 
        decay = 1000):
        super(PolyVAE_repara, self).__init__()
        # encoder
        self.vocab_dims = 10
        self.layer_num = 2
        self.embedding = nn.Embedding(input_dims, self.vocab_dims)
        self.encoder_gru = nn.GRU(
            self.vocab_dims, 
            hidden_dims, 
            num_layers = self.layer_num, 
            batch_first = True, 
            bidirectional = True, 
            dropout = 0.2
        )
        self.linear_mu = nn.Linear(hidden_dims * 2 * self.layer_num, z_dims)
        # self.linear_var = nn.Linear(hidden_dims * 2 * self.layer_num, z_dims)
        self.linear_log_var = nn.Linear(hidden_dims * 2 * self.layer_num, z_dims)
        # hierarchical_decoder
        self.beat_layer_num = 2
        self.tick_layer_num = 2
        self.z_to_beat_hidden = nn.Sequential(
            nn.Linear(z_dims, hidden_dims * self.beat_layer_num),
            nn.SELU()
        )
        self.beat_0 = Parameter(data = torch.zeros(1))
        self.beat_gru = nn.GRU(
            1, hidden_dims, num_layers = self.beat_layer_num, 
            dropout = 0.2, batch_first = True)
        self.beat_to_tick_hidden = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims * self.tick_layer_num),
            nn.SELU()
        )
        self.beat_to_tick_input = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.SELU()
        )
        self.tick_0 = Parameter(data = torch.zeros(self.vocab_dims))
        self.d_embedding = nn.Embedding(input_dims, self.vocab_dims)
        self.tick_gru = nn.GRU(
            self.vocab_dims + hidden_dims, hidden_dims, num_layers = self.tick_layer_num, 
            dropout = 0.2, batch_first = True)
        self.tick_to_note = nn.Sequential(
            nn.Linear(hidden_dims, input_dims),
            nn.ReLU()
        )
        # parameter initialization
        self.input_dims = input_dims
        self.z_dims = z_dims
        self.hidden_dims = hidden_dims
        self.seq_len = seq_len    
        self.beat_num = beat_num
        self.tick_num = tick_num
        # teacher forcing hyperparameters
        self.iteration = 0
        self.eps = 1.0
        self.decay = torch.FloatTensor([decay])
        
    def encoder(self, rx):
        rx = self.embedding(rx) #rx: [64, 160]
        rx = self.encoder_gru(rx)[-1]
        rx = rx.transpose(0,1).contiguous()
        rx = rx.view(rx.size(0), -1)
        r_mu = self.linear_mu(rx)
        # r_var = self.linear_var(rx)
        r_log_var = self.linear_log_var(rx)
        r_var = torch.exp(r_log_var)#[64, 1024]
        std_z = torch.randn(r_var.size()).cuda() #[64, 1024]
        assert not std_z.requires_grad
        #r_dis = Normal(r_mu, r_var)
        self.z_mu=r_mu
        self.z_var=r_var

        # return r_dis
        return r_mu + r_var * std_z

    def final_decoder(self, z, gd, is_train = True):
        gd = self.d_embedding(gd) #[64, 160, 10]
        beat_out = self.forward_beat(z)
        recon = self.forward_tick(beat_out, gd, is_train)
        return recon

    def forward_beat(self, z):
        batch_size = z.size(0)
        h_beat = self.z_to_beat_hidden(z) #[64, 1024]
        h_beat = h_beat.view(batch_size, self.beat_layer_num, -1)
        h_beat = h_beat.transpose(0,1).contiguous() #[2, 64, 512]
        beat_input = self.beat_0.unsqueeze(0).expand(
            batch_size, self.beat_num, 1
        ) #[64, 10, 1]
        beat_out, _ = self.beat_gru(beat_input, h_beat) #[64, 10, 512]
#         print("beat_out",beat_out.size())
        return beat_out

    def forward_tick(self, beat_out, gd, is_train = True):
        ys = []
        batch_size = beat_out.size(0) #tick_0 [10]
        tick_input = self.tick_0.unsqueeze(0).expand(batch_size, self.vocab_dims)#[64, 10]
        tick_input = tick_input.unsqueeze(1) #[64,1,10]
        y = tick_input
#         print(beat_out)
        for i in range(self.beat_num):
            h_tick = self.beat_to_tick_hidden(beat_out[:, i, :]) #[64, 1024]
            h_tick = h_tick.view(batch_size, self.tick_layer_num, -1)
            h_tick = h_tick.transpose(0,1).contiguous() #[2,64, 512]
#             print(h_tick)
            c_tick = self.beat_to_tick_input(beat_out[:, i, :]).unsqueeze(1) #[64,1, 512]
            for j in range(self.tick_num): # tick_num=16
                y = torch.cat((y, c_tick), -1)
#                 print("y size:",y.size())
                y, h_tick = self.tick_gru(y, h_tick)
                y = y.contiguous().view(y.size(0), -1) #[64, 512]
                y = self.tick_to_note(y) #[64, 130]
#                 print("after embed:", y.size())
                ys.append(y)
                y = y.argmax(-1)
#                 print("argmax y:",y)
                y = self.d_embedding(y)
                if self.training and is_train:
                    p = torch.rand(1).item()
                    if p < self.eps:
                        y = gd[:, i * self.tick_num + j, :]
#                         print("yes")
#                     else:
#                         print("no")
                    # update the eps after one batch
                    self.eps = self.decay / (self.decay + torch.exp(self.iteration / self.decay))
                y = y.unsqueeze(1)
#                 print("next input:",y.size())
#                 print(y)
#                 print(gd[:, i * self.tick_num + j,:])
        return torch.stack(ys, 1)
    def forward(self, x, gd):
        # x: [batch, seq_len, 1] with input number range
        # gd: [batch, seq_len, 1] groundtruth of the melody sequence
#         print("vae forward", self.training)
        if self.training:
            self.iteration += 1
        # r_dis = self.encoder(x)
        # z = r_dis.rsample()
        z = self.encoder(x) # [64, 1024]
        recon = self.final_decoder(z, gd)
        # return recon, r_dis, self.iteration
        return recon, z, self.iteration, self.z_mu, self.z_var
