import torch
import os
import numpy as np
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from model import PolyVAE_repara
from torch.utils.data import Dataset, DataLoader, TensorDataset

class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]
###############################
# initial parameters
s_dir = "/home/zexue/Downloads/CSE251B/CSE-251B-Melody-Voice-VAE/"
batch_size = 256
n_epochs = 200
data_path = [s_dir + "data/poly_train_fix.npy",
             s_dir + "data/poly_validate_fix.npy",
             s_dir + "data/poly_test_fix.npy"]
save_path = ""
lr = 1e-3
decay = 0.9999
hidden_dims = 512
z_dims = 1024
vae_beta = 0.1
input_dims = 90
seq_len = 20 * 16
beat_num = 20
tick_num = 16
save_period = 10
##############################
train_set = np.load(data_path[0], allow_pickle = True)
validate_set = np.load(data_path[1],allow_pickle = True) 
test_set = np.load(data_path[2],allow_pickle = True) 

train_x = []
for i,data in enumerate(train_set):
    temp = []
    for d in data["layers"]:
        temp += d
    train_x.append(temp)
train_x = np.array(train_x)
# print(train_x.shape)

validate_x = []
for i,data in enumerate(validate_set):
    temp = []
    for d in data["layers"]:
        temp += d
    validate_x.append(temp)
validate_x = np.array(validate_x)
# print(train_x.shape)

test_x = []
for i,data in enumerate(test_set):
    temp = []
    for d in data["layers"]:
        temp += d
    test_x.append(temp)
test_x = np.array(test_x)


train_x = torch.from_numpy(train_x).long()
validate_x = torch.from_numpy(validate_x).long()
test_x = torch.from_numpy(test_x).long()

print(train_x.size())
print(validate_x.size())
print(test_x.size())

train_set = TensorDataset(train_x)
validate_set = TensorDataset(validate_x)
test_set = TensorDataset(test_x)

train_set = DataLoader(
    dataset = train_set,
    batch_size = batch_size, 
    shuffle = True, 
    num_workers = 0, 
    pin_memory = True, 
    drop_last = True
)
validate_set = DataLoader(
    dataset = validate_set,
    batch_size = batch_size, 
    shuffle = False, 
    num_workers = 0, 
    pin_memory = True, 
    drop_last = True
)

test_set = DataLoader(
    dataset = test_set,
    batch_size = batch_size, 
    shuffle = False, 
    num_workers = 0, 
    pin_memory = True, 
    drop_last = True
)
model = PolyVAE_repara(input_dims, hidden_dims, z_dims, seq_len, beat_num, tick_num, 4000)
optimizer = optim.Adam(model.parameters(), lr = lr)
if decay > 0:
    scheduler = MinExponentialLR(optimizer, gamma = decay, minimum = 1e-5)
if torch.cuda.is_available():
    print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print('Using: CPU')

validate_data = []
for i,d in enumerate(validate_set):
    validate_data.append(d[0])
print(len(validate_data))

def loss_function(recon, target, z_mean, z_std, beta):
    CE = F.cross_entropy(recon.view(-1, recon.size(-1)), target, reduction = "mean")
#     rhy_CE = F.nll_loss(recon_rhythm.view(-1, recon_rhythm.size(-1)), target_rhythm, reduction = "mean")
    
    # normal1 =  std_normal(r_dis.mean.size())
    # KLD1 = kl_divergence(r_dis, normal1).mean()
    mean_sq=z_mean * z_mean
    std_sq=z_std * z_std
    KLD1 = 0.5 * torch.mean(mean_sq + std_sq - torch.log(std_sq) - 1)

    max_indices = recon.view(-1, recon.size(-1)).max(-1)[-1]
#     print(max_indices)
    correct = max_indices == target
    acc = torch.sum(correct.float()) / target.size(0)
    return acc, CE + beta * (KLD1)

device = torch.device(torch.cuda.current_device())

logs = []
iteration = 0
step = 0
best_acc=0
for epoch in range(n_epochs):
    print("epoch: %d\n__________________________________________" % (epoch), flush = True)
    mean_loss = 0.0
    mean_acc = 0.0
    v_mean_loss = 0.0
    v_mean_acc = 0.0
    total = 0
    for i, d in enumerate(train_set):
        # validate display
        x = gd = d[0]
        model.train()
        j = i % len(validate_data)
        v_x = v_gd = validate_data[j]
        
        x = x.to(device = device,non_blocking = True)
        gd = gd.to(device = device,non_blocking = True)
        v_x = v_x.to(device = device,non_blocking = True)
        v_gd = v_gd.to(device = device,non_blocking = True)
            
        optimizer.zero_grad()
        recon, r_dis, iteration, z_mu, z_var = model(x, gd)
        
        acc, loss = loss_function(recon, gd.view(-1), z_mu, z_var, vae_beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        mean_loss += loss.item()
        mean_acc += acc.item()
        
        model.eval()
        with torch.no_grad():
            v_recon, v_r_dis, _, v_z_mu, v_z_var = model(v_x, v_gd)
            v_acc, v_loss = loss_function(v_recon, v_gd.view(-1), v_z_mu, v_z_var, vae_beta)
            v_mean_loss += v_loss.item()
            v_mean_acc += v_acc.item()
        step += 1
        total += 1
        if decay > 0:
            scheduler.step()
        if i % 400 == 0:
            print("batch %d loss: %.5f acc: %.5f | val loss %.5f acc: %.5f iteration: %d"  
                % (i,loss.item(), acc.item(), v_loss.item(),v_acc.item(),iteration),flush = True)
    mean_loss /= total
    mean_acc /= total
    v_mean_loss /= total
    v_mean_acc /= total
    print("epoch %d loss: %.5f acc: %.5f | val loss %.5f acc: %.5f iteration: %d"  
              % (epoch, mean_loss, mean_acc, v_mean_loss, v_mean_acc, iteration),flush = True)
    logs.append([mean_loss,mean_acc,v_mean_loss,v_mean_acc,iteration])
    if v_mean_acc > best_acc:
        filename = "best-" + 'loss_' + str(mean_loss) + "_" + str(epoch+1) + "_" + str(iteration) + ".pt"
        torch.save(model.cpu().state_dict(),save_path + filename)
        best_acc=v_mean_acc
        model.cuda()
np.save("320-log.npy", logs)
        


# test
model.eval()
device = torch.device(torch.cuda.current_device())
mean_acc = 0.0
mean_loss=0.0
ys = []
gds = []
output = []
for i, d in enumerate(validate_set):
    # validate display
    x = gd = d[0]
        
    x = x.to(device = device,non_blocking = True)
    gd = gd.to(device = device,non_blocking = True)
            
    recon, r_dis, iteration, z_mu, z_var = model(x, gd)
        
    acc,v_loss = loss_function(recon, gd.view(-1), z_mu, z_var, vae_beta)
    mean_acc += acc.item()
    mean_loss+=v_loss.item()
#     z = r_dis.rsample()
    pred = recon.argmax(-1)
    output.append({"gd":x.cpu().detach().numpy(), "pred":pred.cpu().detach().numpy(), "acc": acc.item()})
#     print(pred.size())
#     print(v_gd.size())
#     output.append([pred.cpu().detach().numpy(), v_gd.cpu().detach().numpy()])
#     print(v_loss.item())

    # np.save("dismeasurevae_gen_exp.npy", output)
print("******validation acc: ", mean_acc/(i+1), "validation loss: ", mean_loss/(i+1))

model.eval()
device = torch.device(torch.cuda.current_device())
mean_acc = 0.0
mean_loss=0.0
ys = []
gds = []
output = []
for i, d in enumerate(test_set):
    # validate display
    x = gd = d[0]
        
    x = x.to(device = device,non_blocking = True)
    gd = gd.to(device = device,non_blocking = True)
            
    recon, r_dis, iteration, z_mu, z_var = model(x, gd)
        
    acc,v_loss = loss_function(recon, gd.view(-1), z_mu, z_var, vae_beta)
    mean_acc += acc.item()
    mean_loss+=v_loss.item()
#     z = r_dis.rsample()
    pred = recon.argmax(-1)
    output.append({"gd":x.cpu().detach().numpy(), "pred":pred.cpu().detach().numpy(), "acc": acc.item()})
#     print(pred.size())
#     print(v_gd.size())
#     output.append([pred.cpu().detach().numpy(), v_gd.cpu().detach().numpy()])
#     print(v_loss.item())

    # np.save("dismeasurevae_gen_exp.npy", output)
print("******test acc: ", mean_acc/(i+1), "validation loss: ", mean_loss/(i+1))