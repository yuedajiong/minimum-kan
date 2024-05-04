import torch

def B_batch(x, grid, k, extend=True):  #compute x on B-spline bases  #x shape: (size, x);  grid shape: (size, grid)/ number of splines;  k: piecewise polynomial order of splines  #engineering: to-optimize performance
    def extend_grid(grid, k_extend=0):  # pad k to left and right  # grid shape: (batch, grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        return grid
    if extend == True:
        grid = extend_grid(grid, k_extend=k)
    grid = grid.unsqueeze(dim=2)
    x = x.unsqueeze(dim=1)
    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k-1, extend=False)  #k阶数很大的时候递归就麻烦了
        value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
    return value

class KALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, grid_number=5, k=3, noise_scale=0.1, scale_base=1.0, scale_spline=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, +1], sp_trainable=True, sb_trainable=True):
        def curve2coef(y_eval, x_eval, grid, k): #converting B-spline curves to B-spline coefficients using least squares.  # x_eval: (size, batch); y_eval: (size, batch); grid: (size, grid); k: scalar
            return torch.linalg.lstsq(B_batch(x_eval, grid, k).permute(0, 2, 1), y_eval.unsqueeze(dim=2)).solution[:, :, 0]  # sometimes 'cuda' version may diverge

        super().__init__()
        self.in_dim, self.out_dim, self.k, self.base_fun = in_dim, out_dim, k, base_fun
        self.size = in_dim*out_dim
        self.weight_sharing = torch.arange(self.size)
        self.mask = torch.ones(self.size)

        self._grid = torch.einsum('i,j->ij', torch.ones(self.size), torch.linspace(grid_range[0], grid_range[1], steps=grid_number + 1))  #shape:(in*out, grid_number+1)  range[-1,+1]  distribution:evenly
        self.coef = torch.nn.Parameter(curve2coef((torch.rand(self.size, self._grid.shape[1])-1/2)*noise_scale/grid_number,  self._grid, self._grid, k))  #shape:(size, coef)
        self.scale_base = torch.nn.Parameter(torch.ones(self.size, ) * scale_base).requires_grad_(sb_trainable)
        self.scale_spline = torch.nn.Parameter(torch.ones(self.size, ) * scale_spline).requires_grad_(sp_trainable)

    def forward(self, x): #x:[-1,in_dim]
        def coef2curve(coef, x_eval,grid,k):  #converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis)  # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
            return torch.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid, k))  #B_batch: (size, coef, batch), summer over coef

        i = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim)).reshape(x.shape[0], self.size).permute(1,0)  # x: shape(batch, in_dim) => shape(out_dim*in_dim, batch)  #engineering: optimizable
        c = coef2curve(coef=self.coef[self.weight_sharing],  x_eval=i, grid=self._grid[self.weight_sharing], k=self.k).permute(1,0)  # shape(size, batch)
        a = self.scale_base.unsqueeze(dim=0) * self.base_fun(i).permute(1,0) + self.scale_spline.unsqueeze(dim=0) * c
        m = self.mask[None, :] * a
        y = torch.sum(m.reshape(x.shape[0], self.out_dim, self.in_dim), dim=2)  # shape(batch, out_dim)
        return y  #KAN_Y = sequential: sum { #_mask * [ $_scale_base * base_fun_silu(X) + $_scale_spline * $coef * spline(X, #grid, #k) ] } + $_bias  #$:parameter: _:optional #:fixed    #b-spline

class KA(torch.nn.Module):
    def __init__(self, layer_width=[2,1,1], grid_number=5, k=3, noise_scale=0.1, noise_scale_base=0.1, base_fun=torch.nn.SiLU(), bias_trainable=True, grid_eps=1.0, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True):
        super().__init__()
        self.act_all, self.bias_all = torch.nn.ModuleList(), torch.nn.ModuleList()
        import math
        for l in range(len(layer_width)-1):
            spline_batch = KALayer(in_dim=layer_width[l], out_dim=layer_width[l+1], grid_number=grid_number, k=k, noise_scale=noise_scale, scale_base=1/math.sqrt(layer_width[l])+(torch.randn(layer_width[l]*layer_width[l+1],)*2-1)*noise_scale_base, scale_spline=1.0, base_fun=base_fun, grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable, sb_trainable=sb_trainable)
            self.act_all.append(spline_batch)     
            bias = torch.nn.Linear(layer_width[l+1], 1, bias=False).requires_grad_(bias_trainable); bias.weight.data *= 0.0  #== torch.nn.Parameter(torch.zeros(1, layer_width[l+1])).requires_grad_(bias_trainable) 如果没有复杂的网咯连接可以直接就放在layer中
            self.bias_all.append(bias)

    def forward(self, x):
        for act_one, bias_one in zip(self.act_all, self.bias_all):
            x = act_one(x) + bias_one.weight
        return x

if __name__ == '__main__':  # KA_Y = _W1*silu(X) + W2*spline(X) + _bias      # MLP_Y = silu(W*X + bias)
    torch.manual_seed(0)
    ka = KA()
    I = torch.rand(3,2)
    O = ka.forward(I)
    print('I:', I.tolist(),'    ','O:', O.tolist())
