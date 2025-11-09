import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_in, num_hid, num_out):
        super().__init__()
        self.linear1 = nn.Linear(num_in, num_hid)
        self.linear2 = nn.Linear(num_hid, num_hid)
        self.linear3 = nn.Linear(num_hid, num_hid)
        self.linear4 = nn.Linear(num_hid, num_hid)
        self.linear5 = nn.Linear(num_hid, num_out)

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
        h = F.silu(self.linear4(h))
        h = self.linear5(h)
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
    
    def action_loss(self, img_batch, noise_batch, timeline):

        #boundary terms
        boundary_term = torch.mean(self.forward(noise_batch, torch.tensor(0)) - self.forward(img_batch, torch.tensor(1)) )

        #intermediate terms
        time_steps = 0

        for t in timeline:

            inter = t*img_batch + (1-t)*noise_batch

            t_vec = t.expand(img_batch.shape[0], 1)

            dsdx = self.derivative_dsdx(inter, t_vec)
            dsdt = self.derivative_dsdt(inter, t_vec)

            step = torch.mean(torch.sum(dsdx**2, dim=1, keepdim=True)/2 + dsdt)

            time_steps += step

        # Monte Carlo integral approximation:
        time_term = time_steps/timeline.shape[0]

        return torch.mean(boundary_term + time_term)
    
    def action_loss_2(self, img_batch, noise_batch, timeline):
        """
        Assumes t=0 -> noise, t=1 -> data.
        Monte Carlo integral is averaged over the provided timeline points.
        """
        device = img_batch.device
        B, D = img_batch.shape
        K = timeline.numel()

        # ---- boundary term: s(noise,0) - s(data,1) ----
        t0 = torch.zeros(B, 1, device=device)
        t1 = torch.ones(B, 1, device=device)
        s_noise = self.forward(noise_batch, t0)   # (B,1)
        s_data  = self.forward(img_batch,  t1)    # (B,1)
        boundary = (s_noise - s_data).mean()

        # ---- build all interpolates for all t in one go ----
        # shapes: (K, B, D)
        t_grid = timeline.to(device).view(K, 1, 1)
        inter  = t_grid * img_batch.unsqueeze(0) + (1.0 - t_grid) * noise_batch.unsqueeze(0)

        # flatten to big batch: (K*B, D)
        inter_flat = inter.reshape(K * B, D)
        t_big = timeline.to(device).repeat_interleave(B).view(K * B, 1)

        # we need grads wrt inputs and parameters → create_graph=True
        inter_flat.requires_grad_(True)
        t_big.requires_grad_(True)

        # ---- single forward over all (x,t) ----
        out = torch.sum(self.forward(inter_flat, t_big)) # (K*B, 1)

        # ---- single autograd call to get both grads ----
        dsdx, dsdt = torch.autograd.grad(
            outputs=out,
            inputs=(inter_flat, t_big),
            create_graph=True,
        )

        # integrand: 0.5*||∇_x s||^2 + ∂_t s
        # result per-sample, then mean over all K*B
        Arndt = dsdt.squeeze(1)
        Billy = (dsdx**2).sum(dim=1)
        integrand = 0.5 * ((dsdx**2).sum(dim=1)) + dsdt.squeeze(1)
        time_term = integrand.mean()  # already averages over K and B

        return boundary + time_term
    
        
    


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





    
