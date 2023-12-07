from einops.einops import rearrange
import torch

class ForwardPredictionLatentLog():
    def __init__(self,model):
        self.model = model
    def log(self, batch,N=8, n_row=4, sample=True, ddim_steps=200,ddim_eta=1.,**kwargs):
        assert ddim_steps is not None
        log = dict()
        z, c, x, xrec, xc = self.model.get_input(batch, self.model.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        import torch.nn.functional as F
        # original images as 128x128
        cond_small = F.interpolate(xc,(128,128),mode='bilinear')
        # first image reconstruction, target image is x and xrec
        first_image_decoded = self.model.first_stage_model.decode(c['c_concat'][0])
        raw_imgs = rearrange(cond_small,'b (t c) h w -> b c h (t w)',t=2)
        log['raw_imgs'] = raw_imgs[N:]
        nc_concat = torch.cat((c['c_concat'][0],c['c_concat'][0]),0)
        nc_crossattn = torch.cat((c['c_crossattn'][0],-c['c_crossattn'][0]),0)
        new_c = {'c_concat': [nc_concat],'c_crossattn': [nc_crossattn]}
        samples, z_denoise_row = self.model.sample_log(cond=new_c,batch_size=2*N,ddim=True, ddim_steps=ddim_steps,eta=ddim_eta)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = rearrange(x_samples,'(t b) c h w -> b c h (t w)',t = 2)
        preds = torch.cat([first_image_decoded,xrec,x_samples],-1)
        log['preds'] = preds
        return log
