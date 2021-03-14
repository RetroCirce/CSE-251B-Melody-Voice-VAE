# %%

# In this file
# we train our proposed PolyVAE

import torch
import os
import numpy as np
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from DynamicDataset import DynamicDataset
from model import PolyVAE
from torch.utils.data import Dataset, DataLoader, TensorDataset


class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]


###############################
# initial parameters
s_dir = ""
batch_size = 64
n_epochs = 1
data_path = [s_dir + "data/poly_train_dynamic.npy",
             s_dir + "data/poly_validate_dynamic.npy",
             s_dir + "data/poly_train_dynamic.npy"]
save_path = ""
lr = 1e-4
decay = 0.9999
hidden_dims = 512
z_dims = 1024
vae_beta = 0.1
input_dims = 130
beat_num = 20
tick_num = 16
seq_len = beat_num * tick_num
save_period = 1
##############################


# %%

# input data
train_set = np.load(data_path[0], allow_pickle=True)
validate_set = np.load(data_path[1], allow_pickle=True)

train_x = []
for i, data in enumerate(train_set):
    temp = []
    for d in data["layers"]:
        temp += d
    train_x.append(temp)
train_x = np.array(train_x)
# print(train_x.shape)

validate_x = []
for i, data in enumerate(validate_set):
    temp = []
    for d in data["layers"]:
        temp += d
    validate_x.append(temp)
validate_x = np.array(validate_x)
# print(train_x.shape)
# train_x = torch.from_numpy(train_x).long()
# validate_x = torch.from_numpy(validate_x).long()
#
# print(train_x.size())
# print(validate_x.size())

# train_set = TensorDataset(train_x)
# validate_set = TensorDataset(validate_x)

train_set = DynamicDataset(train_x)
validate_set = DynamicDataset(validate_x)

train_dl = DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=True
)
validate_dl = DataLoader(
    dataset=validate_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    drop_last=True
)

# %%

# import model
model = PolyVAE(input_dims, hidden_dims, z_dims, seq_len, beat_num, tick_num, 4000)
optimizer = optim.Adam(model.parameters(), lr=lr)
if decay > 0:
    scheduler = MinExponentialLR(optimizer, gamma=decay, minimum=1e-5)
if torch.cuda.is_available():
    print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print('Using: CPU')

# %%

# # process validete data from the dataloder
# validate_data = []
# for i, d in enumerate(validate_set):
#     validate_data.append(d[0])
# print(len(validate_data))


# %%

# loss function
def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N


def loss_function(recon, target, r_dis, beta):
    CE = F.cross_entropy(recon.view(-1, recon.size(-1)), target, reduction="mean")
    #     rhy_CE = F.nll_loss(recon_rhythm.view(-1, recon_rhythm.size(-1)), target_rhythm, reduction = "mean")
    normal1 = std_normal(r_dis.mean.size())
    KLD1 = kl_divergence(r_dis, normal1).mean()
    max_indices = recon.view(-1, recon.size(-1)).max(-1)[-1]
    #     print(max_indices)
    correct = max_indices == target
    acc = torch.sum(correct.float()) / target.size(0)
    return acc, CE + beta * (KLD1)


# %%

# start training
logs = []
device = torch.device(torch.cuda.current_device())
iteration = 0
step = 0
for epoch in range(n_epochs):
    print("epoch: %d\n__________________________________________" % (epoch), flush=True)
    mean_loss = 0.0
    mean_acc = 0.0
    v_mean_loss = 0.0
    v_mean_acc = 0.0
    total = 0
    for i, d in enumerate(train_dl):
        # validate display
        x = d['data']
        lens = d['lens']
        model.train()
        j = i % len(validate_set)
        v_x = validate_set[j]['data'].unsqueeze(0)
        v_lens = validate_set[j]['lens'].unsqueeze(0)

        x = x.to(device=device, non_blocking=True)
        # lens = lens.to(device=device, non_blocking=True)
        v_x = v_x.to(device=device, non_blocking=True)
        # v_lens = v_lens.to(device=device, non_blocking=True)

        optimizer.zero_grad()
        recon, r_dis, iteration = model(x, lens)

        acc, loss = loss_function(recon, x.view(-1), r_dis, vae_beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        mean_loss += loss.item()
        mean_acc += acc.item()

        model.eval()
        with torch.no_grad():
            v_recon, v_r_dis, _ = model(v_x, v_lens)
            v_acc, v_loss = loss_function(v_recon, v_x.view(-1), v_r_dis, vae_beta)
            v_mean_loss += v_loss.item()
            v_mean_acc += v_acc.item()
        step += 1
        total += 1
        if decay > 0:
            scheduler.step()
        print("batch %d loss: %.5f acc: %.5f | val loss %.5f acc: %.5f iteration: %d"
              % (i, loss.item(), acc.item(), v_loss.item(), v_acc.item(), iteration), flush=True)
    mean_loss /= total
    mean_acc /= total
    v_mean_loss /= total
    v_mean_acc /= total
    print("epoch %d loss: %.5f acc: %.5f | val loss %.5f acc: %.5f iteration: %d"
          % (epoch, mean_loss, mean_acc, v_mean_loss, v_mean_acc, iteration), flush=True)
    logs.append([mean_loss, mean_acc, v_mean_loss, v_mean_acc, iteration])
    if (epoch + 1) % save_period == 0:
        filename = "sketchvae-" + 'loss_' + str(mean_loss) + "_" + str(epoch + 1) + "_" + str(iteration) + ".pt"
        torch.save(model.cpu().state_dict(), save_path + filename)
        model.cuda()
    np.save("sketchvae-log.npy", logs)

# %%


