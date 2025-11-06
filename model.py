import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_in, num_hid, num_out):
        super().__init__()
        self.linear1 = nn.Linear(num_in, num_hid)
        self.linear2 = nn.Linear(num_hid, num_hid)
        self.linear3 = nn.Linear(num_hid, num_hid)
        self.linear4 = nn.Linear(num_hid, num_out)

    def forward(self, x, t):

        # concatenate t and x along the last dimension
        # t may be (bs, 1) or scalar; ensure it matches x’s batch dim
        if t.ndim == 0:
            t = t.expand(x.shape[0], 1)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)
        h = torch.cat([t, x], dim=-1)

        h = F.relu(self.linear1(h))
        h = F.silu(self.linear2(h))  # SiLU = Swish
        h = F.silu(self.linear3(h))
        h = self.linear4(h)
        return h
    
    def derivative_dsdx(self, x, t):

        x_ = x.clone().detach().requires_grad_(True)
        t_ = t.clone().detach().requires_grad_(True)
        out = self.forward(x_, t_)
        y = torch.sum(out)
        d = torch.autograd.grad(y, x_, create_graph= True)[0]
        return d
    
    def derivative_dsdt(self, x, t):

        x_ = x.clone().detach().requires_grad_(True)
        t_ = t.clone().detach().requires_grad_(True)
        out = self.forward(x_, t_)
        y = torch.sum(out)
        d = torch.autograd.grad(y, t_, create_graph= True)[0]
        return d
    
    def grads_wrt_t_x(self, x, t):
        t = t.clone().detach().requires_grad_(True)
        x = x.clone().detach().requires_grad_(True)
        y = self.forward(x, t).sum()
        dx, dt = torch.autograd.grad(y, (x, t), create_graph=True)
        return dx, dt
    
    def action_loss(self, img_batch, noise_batch, timeline):

        #boundary terms
        boundary_term = torch.mean(self.forward(img_batch, torch.tensor(0)) - self.forward(noise_batch, torch.tensor(1)))

        #intermediate terms
        time_steps = 0

        for i, t in enumerate(timeline):

            inter = t*img_batch + (1-t)*noise_batch

            t_vec = t.expand(img_batch.shape[0], 1)

            dsdx = self.derivative_dsdx(inter, t_vec)
            dsdt = self.derivative_dsdt(inter, t_vec)

            step = torch.mean(torch.sum(dsdx**2, dim=1, keepdim=True)/2 + dsdt)

            time_steps += step

        # Monte Carlo integral approximation:
        time_term = time_steps/timeline.shape[0]

        return torch.mean(boundary_term + time_term)
    
        
    


if __name__ == "__main__":

    device = torch.device('cpu')

    model = MLP(3,1,1)

    #data
    t_0 = torch.zeros((100,1), requires_grad= True)
    x_0 = torch.ones((100,2), requires_grad= True)
    
    t_1 = torch.zeros((100,1), requires_grad= True)
    x_1 = torch.ones((100,2), requires_grad= True)

    t_inter = torch.zeros((100,1), requires_grad= True) + (1/2)
    x_inter = torch.ones((100,2), requires_grad= True)
    

    x_data = [x_0, x_1, x_inter]
    t_data = [t_0, t_1, t_inter]

    for name, param in model.named_parameters():
        print(name, param.shape)
        print(param) 

    action_loss = model.action_loss(x_data, t_data)

    print(f"Action Loss: {action_loss}")





    
