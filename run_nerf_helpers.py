import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hash_encoding import HashEmbedder, SHEncoder

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, args, i=0):
    if i == -1:
        return nn.Identity(), 3
    elif i==0:
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }
        
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        out_dim = embedder_obj.out_dim
    elif i==1:
        embed = HashEmbedder(bounding_box=args.bounding_box, \
                            log2_hashmap_size=args.log2_hashmap_size, \
                            finest_resolution=args.finest_res)
        out_dim = embed.out_dim
    elif i==2:
        embed = SHEncoder()
        out_dim = embed.out_dim
    return embed, out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


# Small NeRF for Hash embeddings
class NeRFSmall(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=3, input_ch_views=3,
                 ):
        super(NeRFSmall, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # sigma
        h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma, geo_feat = h[..., 0], h[..., 1:]
        
        # color
        h = torch.cat([input_views, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
            
        # color = torch.sigmoid(h)
        color = h
        outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], -1)

        return outputs

class HashNeRF3RGB(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=3, input_ch_views=3,
                 ):
        super(HashNeRF3RGB, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.sigma_net = nn.ModuleList(sigma_net)
        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.color_net_r = self._create_mlp()
        self.color_net_g = self._create_mlp()
        self.color_net_b = self._create_mlp()
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma, geo_feat = h[..., 0], h[..., 1:]
        h_color = torch.cat([input_views, geo_feat], dim=-1)
        h_color_r = h_color_g = h_color_b = h_color
        for l in range(self.num_layers):
            h_color_r = self.color_net_r[l](h_color_r)
            h_color_g = self.color_net_g[l](h_color_g)
            h_color_b = self.color_net_b[l](h_color_b)
            if l != self.num_layers - 1:
                h_color_r = F.relu(h_color_r, inplace=True)
                h_color_g = F.relu(h_color_g, inplace=True)
                h_color_b = F.relu(h_color_b, inplace=True)
        
        # Concatenate outputs along the last dimension
        color = torch.cat([h_color_r, h_color_g, h_color_b], dim=-1)
        outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], -1)
        # print(f'!!!!!!shape of outputs: {outputs.shape}')
        return outputs
    def _create_mlp(self):
        color_net =  []
        for l in range(self.num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim
            
            if l == self.num_layers_color - 1:
                out_dim = 1 # 1 channel
            else:
                out_dim = self.hidden_dim
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
        return nn.ModuleList(color_net)

# class HashNeRFDRGB(nn.Module):
#     def __init__(self,
#                  num_layers=3,
#                  hidden_dim=64,
#                  geo_feat_dim=15,
#                  num_layers_color=4,
#                  hidden_dim_color=64,
#                  input_ch=3, input_ch_views=3,
#                  ):
#         super(HashNeRFDRGB, self).__init__()

#         self.input_ch = input_ch
#         self.input_ch_views = input_ch_views

#         self.num_layers = num_layers + num_layers_color - 2
#         self.hidden_dim = hidden_dim
#         self.geo_feat_dim = geo_feat_dim

#         net = []
#         for l in range(self.num_layers):
#             if l == 0:
#                 in_dim = self.input_ch
#             else:
#                 in_dim = hidden_dim
#             if l == self.num_layers - 1:
#                 out_dim = 4 # 1 sigma + 3RGM
#             else:
#                 out_dim = hidden_dim
            
#             net.append(nn.Linear(in_dim, out_dim, bias=False))
#         self.net = nn.ModuleList(net)
    
#     def forward(self, x):
#         input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
#         h = input_pts
#         for l in range(self.num_layers):
#             # print(f'l={l}')
#             h = self.net[l](h)
#             if l != self.num_layers - 1:
#                 h = F.relu(h, inplace=True)
#         # print(f'!!!!!!shape of h: {h.shape}')
#         return h
    
class HashNeRFDRGB(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=3, input_ch_views=3,
                 ):
        super(HashNeRFDRGB, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        self.num_layers = num_layers + num_layers_color - 2
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        net = []
        for l in range(self.num_layers):
            if l == 0:
                in_dim = self.input_ch + self.input_ch_views
            else:
                in_dim = hidden_dim
            if l == self.num_layers - 1:
                out_dim = 4 # 1 sigma + 3RGM
            else:
                out_dim = hidden_dim
            
            net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        # input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = x
        for l in range(self.num_layers):
            # print(f'l={l}')
            h = self.net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        # print(f'!!!!!!shape of h: {h.shape}')
        return h

# class for learnable focal
class LearnFocal(nn.Module):
    def __init__(self, H, W, req_grad):
        super(LearnFocal, self).__init__()
        self.H = H
        self.W = W
        self.focal = nn.Parameter(torch.tensor(400.0, dtype=torch.float32), requires_grad=req_grad)
    def forward(self):
        # return self.focal
        K = torch.tensor([
                [self.focal, 0, 0.5*self.W],
                [0, self.focal, 0.5*self.H],
                [0, 0, 1]
            ])
        return K
    
# class for learnable poses
class LearnPoses(nn.Module):
    def __init__(self, num_poses, req_grad):
        super(LearnPoses, self).__init__()
        self.num_poses = num_poses
        self.poses = nn.Parameter(torch.full(size=(num_poses, 3, 4), fill_value=0.02, dtype=torch.float32), requires_grad=req_grad)
    def forward(self):
        return self.poses

# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    # t = -(near + rays_o[...,2]) / rays_d[...,2]
    # rays_o = rays_o + t[...,None] * rays_d
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    is_zero = torch.abs(rays_d[..., 2]) < 1e-8  # Check if denominator is close to zero
    t[is_zero] = 0  # Set t to 0 where rays_d[..., 2] is close to zero
    rays_o = rays_o + t[..., None] * rays_d
    
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    # print(f'!!!!!!!debug in rdc_rays: {torch.isnan(rays_o).any()}')
    # print(f'!!!!!!!debug: {rays_o.shape}')
    # print(f'!!!!!!!debug: {rays_o[0]}')
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
