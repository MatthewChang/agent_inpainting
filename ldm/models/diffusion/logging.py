from einops.einops import rearrange, repeat
import torch
import torch.nn.functional as F

class NoHandsLog():
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
        # original images as 128x128
        lc = z.shape[1]
        B = c.shape[0]
        encoded_imgs = rearrange(c,'b (t c) h w -> (b t) c h w',c=lc)
        recon_imgs = self.model.first_stage_model.decode(encoded_imgs)
        out_ims = rearrange(recon_imgs,'(b t) c h w -> b c h (t w)',b=B)
        samples, z_denoise_row = self.model.sample_log(cond=c,batch_size=N,ddim=True, ddim_steps=ddim_steps,eta=ddim_eta)
        x_samples = self.model.decode_first_stage(samples)
        preds = torch.cat([out_ims,xrec,x_samples],-1)
        if 'loss_mask' in batch:
            lm = batch['loss_mask'][:N]
            upscaled = F.interpolate(lm.unsqueeze(1).float(),x_samples.shape[-2:],mode='bilinear')
            upscaled = (upscaled == 1)
            loss_mask = repeat(upscaled,'b 1 h w -> b 3 h w')
            preds = torch.cat([preds,loss_mask.float()],-1)
        log['preds'] = preds
        return log

class InpaintLog():
    def __init__(self,model):
        self.model = model
    def log(self, batch,N=8, n_row=4, sample=True, ddim_steps=200,ddim_eta=1.,**kwargs):
        assert ddim_steps is not None
        log = dict()
        z, con, x, xrec, xc = self.model.get_input(batch, self.model.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        # original images as 128x128
        lc = z.shape[1]
        B = con.shape[0]
        im_channels = torch.cat((con[:,:3],con[:,4:]),1)
        in_mask = con[:,3]
        encoded_imgs = rearrange(im_channels,'b (t c) h w -> (b t) c h w',c=lc)
        recon_imgs = self.model.first_stage_model.decode(encoded_imgs)
        out_ims = rearrange(recon_imgs,'(b t) c h w -> b c h (t w)',b=B)
        samples, z_denoise_row = self.model.sample_log(cond=con,batch_size=N,ddim=True, ddim_steps=ddim_steps,eta=ddim_eta)
        x_samples = self.model.decode_first_stage(samples)
        in_mask_up = F.interpolate(in_mask.unsqueeze(1).float(),x_samples.shape[-2:],mode='bilinear')
        in_mask_up = repeat(in_mask_up,'b 1 h w -> b 3 h w')
        preds = torch.cat([out_ims,in_mask_up.float(),xrec,x_samples],-1)
        if 'loss_mask' in batch:
            lm = batch['loss_mask'][:N]
            upscaled = F.interpolate(lm.unsqueeze(1).float(),x_samples.shape[-2:],mode='bilinear')
            upscaled = (upscaled == 1)
            loss_mask = repeat(upscaled,'b 1 h w -> b 3 h w')
            preds = torch.cat([preds,loss_mask.float()],-1)
        log['preds'] = preds
        return log

class InpaintLog3d():
    def __init__(self,model):
        self.model = model
    def log(self, batch,N=8, n_row=4, sample=True, ddim_steps=200,ddim_eta=1.,**kwargs):
        assert ddim_steps is not None
        log = dict()
        z, con, x, xrec, xc = self.model.get_input(batch, self.model.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        # original images as 128x128
        lc = z.shape[1]
        B = con.shape[0]
        in_mask = con[:,-1,-1]
        encoded_imgs = rearrange(con[:,:,:3],'b t c h w -> (b t) c h w',c=lc)
        recon_imgs = self.model.first_stage_model.decode(encoded_imgs)
        out_ims = rearrange(recon_imgs,'(b t) c h w -> b c h (t w)',b=B)
        samples, z_denoise_row = self.model.sample_log(cond=con,batch_size=N,ddim=True, ddim_steps=ddim_steps,eta=ddim_eta)
        x_samples = self.model.decode_first_stage(samples)
        in_mask_up = F.interpolate(in_mask.unsqueeze(1).float(),x_samples.shape[-2:],mode='bilinear')
        in_mask_up = repeat(in_mask_up,'b 1 h w -> b 3 h w')
        preds = torch.cat([out_ims,in_mask_up.float(),xrec,x_samples],-1)
        if 'loss_mask' in batch:
            lm = batch['loss_mask'][:N]
            upscaled = F.interpolate(lm.unsqueeze(1).float(),x_samples.shape[-2:],mode='bilinear')
            upscaled = (upscaled == 1)
            loss_mask = repeat(upscaled,'b 1 h w -> b 3 h w')
            preds = torch.cat([preds,loss_mask.float()],-1)
        log['preds'] = preds
        return log
