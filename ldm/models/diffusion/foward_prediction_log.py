from einops.einops import rearrange
import torch

class ForwardPredictionLog():
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
        first_image_decoded = self.model.first_stage_model.decode(c)
        raw_imgs = torch.cat((xc[:N],x),axis=-1)
        log['raw_imgs'] = raw_imgs
        samples, z_denoise_row = self.model.sample_log(cond=c,batch_size=N,ddim=True, ddim_steps=ddim_steps,eta=ddim_eta)
        x_samples = self.model.decode_first_stage(samples)
        x_samples.shape
        first_image_decoded.shape
        preds = torch.cat([first_image_decoded,xrec,x_samples],axis=-1)
        log['preds'] = preds
        return log
