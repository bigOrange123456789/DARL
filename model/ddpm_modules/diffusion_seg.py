import math
import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from . import loss

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise(): return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn, segment_fn,#这两个函数分别表示 UNet、SPADEGenerator
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.segment_fn = segment_fn
        self.conditional = conditional
        self.loss_type = loss_type
        if schedule_opt is not None:
            pass

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()
        self.loss_gan = loss.GANLoss('lsgan').to(device)
        self.loss_cyc = torch.nn.L1Loss()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):

        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10)) #1#

        shape = x_in['B'].shape
        b = shape[0]
        img = torch.randn(shape, device=device)
        ret_img = img
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)

        t = torch.full((b,), 0, device=device, dtype=torch.long)
        A_latent = self.denoise_fn(x_in['A'], t)
        segm_V = self.segment_fn(torch.cat([x_in['A'], A_latent], dim=1))
        B_latent = self.denoise_fn(x_in['B'], t)
        fractal = torch.eye(2)[:, torch.clamp_min(x_in['F'][:, 0], 0).type(torch.long)].transpose(0, 1)
        synt_A = self.segment_fn(torch.cat([x_in['B'], B_latent], dim=1), fractal.to(device))

        if continous:
            return ret_img, synt_A, segm_V
        else:
            return ret_img[-1], synt_A, segm_V

    def p_sample_segment(self, x_in):
        device = self.betas.device
        x_start_ = x_in['A']
        segm_V = torch.zeros_like(x_start_)
        for opt1 in range(2):
            for opt2 in range(2):
                x_start = x_start_[:, :, opt1::2, opt2::2]
                b= x_start.shape[0]
                t = torch.full((b,), 0, device=device, dtype=torch.long)
                x_latent = self.denoise_fn(x_start, t)
                segm_V[:, :, opt1::2, opt2::2] = self.segment_fn(torch.cat([x_start, x_latent], dim=1))
        return segm_V

    @torch.no_grad()
    def sample(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    @torch.no_grad()
    def segment(self, x_in):
        return self.p_sample_segment(x_in)

    def q_sample(self, x_start, t, noise=None): #貌似q为逆向去噪过程
        noise = default(noise, lambda: torch.randn_like(x_start))

        # fix gama
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_in, noise=None):#好像是用来计算某个损失 #貌似p为正向加噪过程
        '''
            x:{
                A <class 'Tensor'> Size([1, 1, 256, 256])
                B <class 'Tensor'> Size([1, 1, 256, 256])
                F <class 'Tensor'> Size([1, 1, 256, 256])
                P <class 'list'> ['./data/Dataset_XCAD/train\\trainC\\003_PPA_-29_PSA_29_2.png']
            }
        '''
        a_start = x_in['A'] # [1, 1, 256, 256] #造影图
        device = a_start.device # cuda:0
        [b, c, h, w] = a_start.shape
        noise = default(noise, lambda: torch.randn_like(a_start))# noise: None->[1, 1, 256, 256]

        #### A path #### 论文中的A分割路径
        t_a = torch.randint(0, 200, (b,), device=device).long() #生成b个0~200之间的随机整数 t_a = tensor([129]) #加噪步数
        A_noisy = self.q_sample(x_start=a_start, t=t_a, noise=noise) #加噪后的图片   [1, 1, 256, 256] 
        A_latent = self.denoise_fn(A_noisy, t_a) #所含的噪声   [1, 1, 256, 256]
        segm_V = self.segment_fn(torch.cat([A_noisy, A_latent], dim=1)) #造影图的分割结果

        #### B path #### 论文中的B合成路径
        t_b = torch.randint(0, self.num_timesteps, (b,), device=device).long() #加噪步数
        b_start = x_in['B'] #背景图
        B_noisy = self.q_sample(x_start=b_start, t=t_b, noise=noise) #加噪后的图片
        B_latent = self.denoise_fn(B_noisy, t_b) #所含的噪声
        # print("torch.eye(2):"                       ,torch.eye(2)      )
        # print("x_in['F']:"                                           ,x_in['F'].shape      )
        # print("x_in['F'][:, 0]:"                                     ,x_in['F'][:, 0].shape)
        # print("torch.clamp_min(x_in['F'][:, 0], 0):"                 ,torch.clamp_min(x_in['F'][:, 0], 0).shape)
        # print("torch.clamp_min(x_in['F'][:, 0], 0).type(torch.long):",torch.clamp_min(x_in['F'][:, 0], 0).type(torch.long).shape)
        '''
            x_in['F']:                                            torch.Size([1, 1, 256, 256])
            x_in['F'][:, 0]:                                      torch.Size([1, 256, 256])
            torch.clamp_min(x_in['F'][:, 0], 0):                  torch.Size([1, 256, 256])
            torch.clamp_min(x_in['F'][:, 0], 0).type(torch.long): torch.Size([1, 256, 256])

            torch.clamp(input, min, max, out=None)
                将input的值限制在[min, max]之间，并返回结果。

            torch.eye()
                生成对角线全1, 其余部分全0的二维数组
            transpose()
                求矩阵的转置
                transpose(0, 1)应该是用于改变矩阵中元素的坐标次序，我认为这里的作用应该是只保留前两个维度
        '''
        # print('torch.clamp_min(x_in['F'][:, 0], 0).type(torch.long)',torch.clamp_min(x_in['F'][:, 0], 0).type(torch.long))
        # fractal = torch.eye(2)[:, torch.clamp_min(x_in['F'][:, 0], 0).type(torch.long)].transpose(0, 1)
        fractal = torch.eye(2).to(device)[:, torch.clamp_min(x_in['F'][:, 0], 0).type(torch.long)].transpose(0, 1) #用于确保整体几何信息的分形图
        # [1, 2, 256, 256] # print('fractal',fractal.device,fractal.shape,type(fractal))
        # 整句代码的作用是：生成一个2x2的单位矩阵，然后根据x_in['F']张量的第一列中的值（经过限制不小于0并转换为长整型）来选择单位矩阵的列，最后将得到的矩阵进行转置。这样的操作可能用于某些特定的矩阵操作或数据预处理步骤。
        synt_A = self.segment_fn(torch.cat([B_noisy, B_latent], dim=1), fractal.to(device)) #合成的造影图
        # [1, 1, 256, 256] # print('synt_A',synt_A.device,synt_A.shape,type(synt_A))
        # return

        #### Cycle path ####
        f_noisy = self.q_sample(x_start=synt_A, t=t_a, noise=noise) #在合成图中添加噪声
        # [1, 1, 256, 256] # print('f_noisy',f_noisy.device,f_noisy.shape,type(f_noisy))
        # print('t_a',t_a.shape,type(t_a),t_a)
        # print('f_noisy',f_noisy.shape,type(f_noisy),f_noisy)
        # return
        f_recon = self.denoise_fn.forward_lzc(f_noisy, t_a) # self.denoise_fn(f_noisy, t_a) #所含的噪声
        print('lzc-diffusion_seg.py 313:为了测试，这里注释了许多代码。')
        return
        '''
            f_noisy # 加噪的合成图
                torch.Size([1, 1, 256, 256]) 
                <class 'torch.Tensor'> 
                tensor([[[[...]]]],device='cuda:0', grad_fn=<AddBackward0>)
            t_a     # 加噪步数为107
                torch.Size([1]) 
                <class 'torch.Tensor'> 
                tensor([107], device='cuda:0')
        '''
        # print('lzc-diffusion_seg.py p_losses():为了测试，这里注释了许多代码。')
        # return
        # [1, 1, 256, 256] # print('f_recon',f_recon.device,f_recon.shape,type(f_recon))
        recn_F = self.segment_fn(torch.cat([f_noisy, f_recon], dim=1)) #合成图的分割结果
        # [1, 1, 256, 256] # print('recn_F',recn_F.device,recn_F.shape,type(recn_F))

        l_dif = self.loss_func(noise, B_latent) # 真实噪声 和 预测噪声 之间的差异
        # print('l_dif',l_dif.device,l_dif.shape,type(l_dif),l_dif)
        l_dif = l_dif.sum() / int(b * c * h * w)
        # print('l_dif',l_dif.device,l_dif.shape,type(l_dif),l_dif)
        l_cyc = self.loss_cyc(recn_F, x_in['F']) # 分割结果 和 分形标签 之间的差异 
        # print('l_cyc',l_cyc.device,l_cyc.shape,type(l_cyc),l_cyc)
        '''
            l_dif: cuda:0 torch.Size([]) tensor(88027.5469, device='cuda:0', grad_fn=<MseLossBackward0>)
            l_dif: cuda:0 torch.Size([]) tensor(1.3432, device='cuda:0', grad_fn=<DivBackward0>)
            l_cyc: cuda:0 torch.Size([]) tensor(0.8366, device='cuda:0', grad_fn=<MeanBackward0>)
        '''
        return [A_noisy, A_latent, B_noisy, B_latent, segm_V, synt_A, recn_F], [l_dif, l_cyc]

    def forward(self, x, *args, **kwargs):
        # print('args',args)
        # print('kwargs',kwargs)
        # print('x[A]',type(x['A']) ,x['A'].shape)
        # print('x[B]',type(x['B']) ,x['B'].shape)
        # print('x[F]',type(x['F']) ,x['F'].shape)
        # print('x[P]',type(x['P']) ,x['P'])
        '''
            x:{
                A <class 'Tensor'> Size([1, 1, 256, 256])
                B <class 'Tensor'> Size([1, 1, 256, 256])
                F <class 'Tensor'> Size([1, 1, 256, 256])
                P <class 'list'> ['./data/Dataset_XCAD/train\\trainC\\000_PPA_44_PSA_00_8.png']
            }
            args: ()
            kwargs: {}
        '''
        # print('lzc-tag.diffusion_seg.py.forword')
        return self.p_losses(x, *args, **kwargs)

