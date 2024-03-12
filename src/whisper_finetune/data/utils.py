import torch


class TimeWarpAugmenter:
    def __init__(self, W=50):
        '''
        Initialize the TimeWarpAugmenter with the strength of warp (W).
        '''
        self.W = W
        
    def __call__(self, specs):
        '''
        Apply time warp augmentation when the class is called.

        param:
        specs: spectrogram of size (batch, channel, freq_bin, length)
        '''
        return self.time_warp(specs)
    
    @staticmethod
    def h_poly(t):
        tt = t.unsqueeze(-2)**torch.arange(4, device=t.device).view(-1,1)
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
        ], dtype=t.dtype, device=t.device)
        return A @ tt

    @staticmethod
    def hspline_interpolate_1D(x, y, xs):
        '''
        Input x and y must be of shape (batch, n) or (n)
        '''
        m = (y[..., 1:] - y[..., :-1]) / (x[..., 1:] - x[..., :-1])
        m = torch.cat([m[...,[0]], (m[...,1:] + m[...,:-1]) / 2, m[...,[-1]]], -1)
        idxs = torch.searchsorted(x[..., 1:], xs)
        # print(torch.abs(x.take_along_dim(idxs+1, dim=-1) - x.gather(dim=-1, index=idxs+1)))
        dx = (x.gather(dim=-1, index=idxs+1) - x.gather(dim=-1, index=idxs))
        hh = TimeWarpAugmenter.h_poly((xs - x.gather(dim=-1, index=idxs)) / dx)
        return hh[...,0,:] * y.gather(dim=-1, index=idxs) \
            + hh[...,1,:] * m.gather(dim=-1, index=idxs) * dx \
            + hh[...,2,:] * y.gather(dim=-1, index=idxs+1) \
            + hh[...,3,:] * m.gather(dim=-1, index=idxs+1) * dx
        # dx = (x.take_along_dim(idxs+1, dim=-1) - x.take_along_dim(idxs, dim=-1))
        # hh = h_poly((xs - x.take_along_dim(idxs, dim=-1)) / dx)
        # return hh[...,0,:] * y.take_along_dim(idxs, dim=-1) \
        #     + hh[...,1,:] * m.take_along_dim(idxs, dim=-1) * dx \
        #     + hh[...,2,:] * y.take_along_dim(idxs+1, dim=-1) \
        #     + hh[...,3,:] * m.take_along_dim(idxs+1, dim=-1) * dx

    def time_warp(self, specs, W=80):
        '''
        Timewarp augmentation by https://github.com/IMLHF/SpecAugmentPyTorch/blob/master/spec_augment_pytorch.py

        param:
        specs: spectrogram of size (batch, channel, freq_bin, length)
        W: strength of warp
        '''
        device = specs.device
        specs = specs.unsqueeze(0).unsqueeze(0)
        batch_size, _, num_rows, spec_len = specs.shape

        warp_p = torch.randint(W, spec_len - W, (batch_size,), device=device)

        # Uniform distribution from (0,W) with chance to be up to W negative
        # warp_d = torch.randn(1)*W # Not using this since the paper author make random number with uniform distribution
        warp_d = torch.randint(-W, W, (batch_size,), device=device)
        # print("warp_d", warp_d)
        x = torch.stack([torch.tensor([0], device=device).expand(batch_size),
                        warp_p, torch.tensor([spec_len-1], device=device).expand(batch_size)], 1)
        y = torch.stack([torch.tensor([-1.], device=device).expand(batch_size),
                        (warp_p-warp_d)*2/(spec_len-1.)-1., torch.tensor([1.], device=device).expand(batch_size)], 1)
        # print((warp_p-warp_d)*2/(spec_len-1.)-1.)

        # Interpolate from 3 points to spec_len
        xs = torch.linspace(0, spec_len-1, spec_len, device=device).unsqueeze(0).expand(batch_size, -1)
        ys = TimeWarpAugmenter.hspline_interpolate_1D(x, y, xs)

        grid = torch.cat(
            (ys.view(batch_size,1,-1,1).expand(-1,num_rows,-1,-1),
            torch.linspace(-1, 1, num_rows, device=device).view(-1,1,1).expand(batch_size,-1,spec_len,-1)), -1)

        return torch.nn.functional.grid_sample(specs, grid, align_corners=True).squeeze(0).squeeze(0)