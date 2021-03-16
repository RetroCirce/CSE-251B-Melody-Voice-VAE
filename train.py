# %%

# In this file
# we train our proposed PolyVAE

import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR

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
n_epochs = 200
data_path = [s_dir + "data/poly_train_dynamic.npy",
             s_dir + "data/poly_validate_dynamic.npy",
             s_dir + "data/poly_train_dynamic.npy"]
save_path = ""
lr = 1e-4
decay = 0.9999
hidden_dims = 512
z_dims = 1024
vae_beta = 0.1
input_dims = 90
beat_num = 20
tick_num = 16
seq_len = beat_num * tick_num
save_period = 1
experiment_name = "dynamic"
torch.cuda.set_device(1)
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


def loss_function(recon, target, r_dis, beta, ignore_index=-100):
    CE = F.cross_entropy(recon.view(-1, recon.size(-1)), target, reduction="mean", ignore_index=ignore_index)
    #     rhy_CE = F.nll_loss(recon_rhythm.view(-1, recon_rhythm.size(-1)), target_rhythm, reduction = "mean")
    normal1 = std_normal(r_dis.mean.size())
    KLD1 = kl_divergence(r_dis, normal1).mean()
    max_indices = recon.view(-1, recon.size(-1)).max(-1)[-1]
    #     print(max_indices)
    correct = max_indices == target
    correct_without_pad = correct[target != 0]
    acc = torch.sum(correct.float()) / target.size(0)
    acc_without_pad = torch.sum(correct_without_pad.float()) / torch.sum(target != 0)
    return acc, acc_without_pad, CE + beta * (KLD1)


# %%

def save_model(model, optimizer):
    root_model_path = os.path.join(experiment_dir, 'best_model.pt')
    model_dict = model.state_dict()
    state_dict = {'model': model_dict, 'optimizer': optimizer.state_dict()}
    torch.save(state_dict, root_model_path)


def load_model(model, optimizer):
    state_dict = torch.load(os.path.join(experiment_dir, 'best_model.pt'))
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])


def record_stats(train_loss, train_acc, val_loss, val_acc):
    training_losses.append(train_loss)
    training_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    plot_stats()


def plot_stats():
    e = len(training_losses)
    x_axis = np.arange(1, e + 1, 1)
    plt.figure()
    plt.plot(x_axis, training_losses, label="Training Loss")
    plt.plot(x_axis, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.legend(loc='best')
    plt.savefig(os.path.join(experiment_dir, "loss_plot.png"))
    plt.close()

    plt.figure()
    plt.plot(x_axis, training_accs, label="Training Accuracy")
    plt.plot(x_axis, val_accs, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.legend(loc='best')
    plt.savefig(os.path.join(experiment_dir, "acc_plot.png"))
    plt.close()


# %%

# rename the experiment dir name every time when you tune the hyperparameters
experiment_dir = os.path.join("experiment_data/", experiment_name)
os.makedirs(experiment_dir, exist_ok=True)

training_losses = []
val_losses = []
training_accs = []
val_accs = []

best_loss = 1000

# start training
logs = []
device = torch.device(torch.cuda.current_device())
iteration = 0
for epoch in range(n_epochs):
    print("epoch: %d\n__________________________________________" % (epoch), flush=True)
    mean_loss = []
    mean_acc = []
    mean_acc_without_pad = []

    total = 0
    with tqdm(train_dl) as t:
        for i, d in enumerate(t):
            # validate display
            x = d['data']
            lens = d['lens']
            model.train()
            j = i % len(validate_set)
            # v_x = validate_set[j]['data'].unsqueeze(0)
            # v_lens = validate_set[j]['lens'].unsqueeze(0)

            x = x.to(device=device, non_blocking=True)
            # lens = lens.to(device=device, non_blocking=True)
            # v_x = v_x.to(device=device, non_blocking=True)
            # v_lens = v_lens.to(device=device, non_blocking=True)

            optimizer.zero_grad()
            recon, r_dis, iteration = model(x, lens)

            # add ignore_index=0 for training without padding loss
            acc, acc_without_pad, loss = loss_function(recon, x.view(-1), r_dis, vae_beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            mean_loss.append(loss.item())
            mean_acc.append(acc.item())
            mean_acc_without_pad.append(acc_without_pad.item())
            total += 1
            if decay > 0:
                scheduler.step()
            t.set_postfix({
                'train_loss': f'{float(loss):.3f}',
                'train_acc': f'{float(acc):.3f}',
                'train_acc_without_pad': f'{float(acc_without_pad):.3f}'
            })


    v_mean_loss = []
    v_mean_acc = []
    v_mean_acc_without_pad = []
    model.eval()
    with torch.no_grad():
        with tqdm(validate_dl) as t:
            for i, d in enumerate(t):
                v_x = d['data']
                v_lens = d['lens']
                v_x = v_x.to(device=device, non_blocking=True)

                v_recon, v_r_dis, _ = model(v_x, v_lens)
                v_acc, v_acc_without_pad, v_loss = loss_function(v_recon, v_x.view(-1), v_r_dis, vae_beta)
                v_mean_loss.append(v_loss.item())
                v_mean_acc.append(v_acc.item())
                v_mean_acc_without_pad.append(v_acc_without_pad.item())
                t.set_postfix({
                    'valid_loss': f'{float(v_loss):.3f}',
                    'valid_acc': f'{float(v_acc):.3f}',
                    'valid_acc_without_pad': f'{float(v_acc_without_pad):.3f}',
                })

            # print("batch %d loss: %.5f acc: %.5f | val loss %.5f acc: %.5f iteration: %d"
            #       % (i,loss.item(), acc.item(), v_loss.item(),v_acc.item(),iteration),flush = True)
    mean_loss = sum(mean_loss) / len(mean_loss)
    mean_acc = sum(mean_acc) / len(mean_acc)
    mean_acc_without_pad = sum(mean_acc_without_pad) / len(mean_acc_without_pad)
    v_mean_loss = sum(v_mean_loss) / len(v_mean_loss)
    v_mean_acc = sum(v_mean_acc) / len(v_mean_acc)
    v_mean_acc_without_pad = sum(v_mean_acc_without_pad) / len(v_mean_acc_without_pad)

    record_stats(mean_loss, mean_acc, v_mean_loss, v_mean_acc)

    print("epoch %d loss: %.5f acc: %.5f acc without pad: %.5f | val loss %.5f acc: %.5f acc without pad: %.5f iteration: %d"
        % (epoch, mean_loss, mean_acc, mean_acc_without_pad, v_mean_loss, v_mean_acc, v_mean_acc_without_pad, iteration), flush=True)
    logs.append([mean_loss, mean_acc, v_mean_loss, v_mean_acc, iteration])
    if (epoch + 1) % save_period == 0:
        filename = "sketchvae-" + 'loss_' + str(mean_loss) + "_" + str(epoch + 1) + "_" + str(iteration) + ".pt"
        torch.save(model.cpu().state_dict(), os.path.join(experiment_dir, filename))
        model.cuda()
    np.save("sketchvae-log.npy", logs)

    if v_mean_loss < best_loss:
        save_model(model, optimizer)

# %%

# os.makedirs(experiment_dir, exist_ok=True)
# record_stats(mean_loss, mean_acc, v_mean_loss, v_mean_acc)

# %%

