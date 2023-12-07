from funcy.compat import lfilter, lmap
from funcy.seqs import lremove
import torch
import re


inpaint_conv_keys = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.input_blocks.1.0.in_layers.2.weight",
    "model.diffusion_model.input_blocks.1.0.out_layers.3.weight",
    "model.diffusion_model.input_blocks.2.0.in_layers.2.weight",
    "model.diffusion_model.input_blocks.2.0.out_layers.3.weight",
    "model.diffusion_model.input_blocks.3.0.in_layers.2.weight",
    "model.diffusion_model.input_blocks.3.0.out_layers.3.weight",
    "model.diffusion_model.input_blocks.4.0.in_layers.2.weight",
    "model.diffusion_model.input_blocks.4.0.out_layers.3.weight",
    "model.diffusion_model.input_blocks.4.0.skip_connection.weight",
    "model.diffusion_model.input_blocks.5.0.in_layers.2.weight",
    "model.diffusion_model.input_blocks.5.0.out_layers.3.weight",
    "model.diffusion_model.input_blocks.6.0.in_layers.2.weight",
    "model.diffusion_model.input_blocks.6.0.out_layers.3.weight",
    "model.diffusion_model.input_blocks.7.0.in_layers.2.weight",
    "model.diffusion_model.input_blocks.7.0.out_layers.3.weight",
    "model.diffusion_model.input_blocks.7.0.skip_connection.weight",
    "model.diffusion_model.input_blocks.8.0.in_layers.2.weight",
    "model.diffusion_model.input_blocks.8.0.out_layers.3.weight",
    "model.diffusion_model.input_blocks.9.0.in_layers.2.weight",
    "model.diffusion_model.input_blocks.9.0.out_layers.3.weight",
    "model.diffusion_model.input_blocks.10.0.in_layers.2.weight",
    "model.diffusion_model.input_blocks.10.0.out_layers.3.weight",
    "model.diffusion_model.input_blocks.10.0.skip_connection.weight",
    "model.diffusion_model.input_blocks.11.0.in_layers.2.weight",
    "model.diffusion_model.input_blocks.11.0.out_layers.3.weight",
    "model.diffusion_model.middle_block.0.in_layers.2.weight",
    "model.diffusion_model.middle_block.0.out_layers.3.weight",
    "model.diffusion_model.middle_block.2.in_layers.2.weight",
    "model.diffusion_model.middle_block.2.out_layers.3.weight",
    "model.diffusion_model.output_blocks.0.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.0.0.out_layers.3.weight",
    "model.diffusion_model.output_blocks.0.0.skip_connection.weight",
    "model.diffusion_model.output_blocks.1.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.1.0.out_layers.3.weight",
    "model.diffusion_model.output_blocks.1.0.skip_connection.weight",
    "model.diffusion_model.output_blocks.2.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.2.0.out_layers.3.weight",
    "model.diffusion_model.output_blocks.2.0.skip_connection.weight",
    "model.diffusion_model.output_blocks.2.2.in_layers.2.weight",
    "model.diffusion_model.output_blocks.2.2.out_layers.3.weight",
    "model.diffusion_model.output_blocks.3.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.3.0.out_layers.3.weight",
    "model.diffusion_model.output_blocks.3.0.skip_connection.weight",
    "model.diffusion_model.output_blocks.4.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.4.0.out_layers.3.weight",
    "model.diffusion_model.output_blocks.4.0.skip_connection.weight",
    "model.diffusion_model.output_blocks.5.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.5.0.out_layers.3.weight",
    "model.diffusion_model.output_blocks.5.0.skip_connection.weight",
    "model.diffusion_model.output_blocks.5.2.in_layers.2.weight",
    "model.diffusion_model.output_blocks.5.2.out_layers.3.weight",
    "model.diffusion_model.output_blocks.6.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.6.0.out_layers.3.weight",
    "model.diffusion_model.output_blocks.6.0.skip_connection.weight",
    "model.diffusion_model.output_blocks.7.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.7.0.out_layers.3.weight",
    "model.diffusion_model.output_blocks.7.0.skip_connection.weight",
    "model.diffusion_model.output_blocks.8.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.8.0.out_layers.3.weight",
    "model.diffusion_model.output_blocks.8.0.skip_connection.weight",
    "model.diffusion_model.output_blocks.8.2.in_layers.2.weight",
    "model.diffusion_model.output_blocks.8.2.out_layers.3.weight",
    "model.diffusion_model.output_blocks.9.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.9.0.out_layers.3.weight",
    "model.diffusion_model.output_blocks.9.0.skip_connection.weight",
    "model.diffusion_model.output_blocks.10.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.10.0.out_layers.3.weight",
    "model.diffusion_model.output_blocks.10.0.skip_connection.weight",
    "model.diffusion_model.output_blocks.11.0.in_layers.2.weight",
    "model.diffusion_model.output_blocks.11.0.out_layers.3.weight",
    "model.diffusion_model.output_blocks.11.0.skip_connection.weight",
]

ema_conv_keys = [
    "model_ema.diffusion_modelinput_blocks00weight",
    "model_ema.diffusion_modelinput_blocks10in_layers2weight",
    "model_ema.diffusion_modelinput_blocks10out_layers3weight",
    "model_ema.diffusion_modelinput_blocks20in_layers2weight",
    "model_ema.diffusion_modelinput_blocks20out_layers3weight",
    "model_ema.diffusion_modelinput_blocks30in_layers2weight",
    "model_ema.diffusion_modelinput_blocks30out_layers3weight",
    "model_ema.diffusion_modelinput_blocks40in_layers2weight",
    "model_ema.diffusion_modelinput_blocks40out_layers3weight",
    "model_ema.diffusion_modelinput_blocks40skip_connectionweight",
    "model_ema.diffusion_modelinput_blocks50in_layers2weight",
    "model_ema.diffusion_modelinput_blocks50out_layers3weight",
    "model_ema.diffusion_modelinput_blocks60in_layers2weight",
    "model_ema.diffusion_modelinput_blocks60out_layers3weight",
    "model_ema.diffusion_modelinput_blocks70in_layers2weight",
    "model_ema.diffusion_modelinput_blocks70out_layers3weight",
    "model_ema.diffusion_modelinput_blocks70skip_connectionweight",
    "model_ema.diffusion_modelinput_blocks80in_layers2weight",
    "model_ema.diffusion_modelinput_blocks80out_layers3weight",
    "model_ema.diffusion_modelinput_blocks90in_layers2weight",
    "model_ema.diffusion_modelinput_blocks90out_layers3weight",
    "model_ema.diffusion_modelinput_blocks100in_layers2weight",
    "model_ema.diffusion_modelinput_blocks100out_layers3weight",
    "model_ema.diffusion_modelinput_blocks100skip_connectionweight",
    "model_ema.diffusion_modelinput_blocks110in_layers2weight",
    "model_ema.diffusion_modelinput_blocks110out_layers3weight",
    "model_ema.diffusion_modelmiddle_block0in_layers2weight",
    "model_ema.diffusion_modelmiddle_block0out_layers3weight",
    "model_ema.diffusion_modelmiddle_block2in_layers2weight",
    "model_ema.diffusion_modelmiddle_block2out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks00in_layers2weight",
    "model_ema.diffusion_modeloutput_blocks00out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks00skip_connectionweight",
    "model_ema.diffusion_modeloutput_blocks10in_layers2weight",
    "model_ema.diffusion_modeloutput_blocks10out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks10skip_connectionweight",
    "model_ema.diffusion_modeloutput_blocks20in_layers2weight",
    "model_ema.diffusion_modeloutput_blocks20out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks20skip_connectionweight",
    "model_ema.diffusion_modeloutput_blocks22in_layers2weight",
    "model_ema.diffusion_modeloutput_blocks22out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks30in_layers2weight",
    "model_ema.diffusion_modeloutput_blocks30out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks30skip_connectionweight",
    "model_ema.diffusion_modeloutput_blocks40in_layers2weight",
    "model_ema.diffusion_modeloutput_blocks40out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks40skip_connectionweight",
    "model_ema.diffusion_modeloutput_blocks50in_layers2weight",
    "model_ema.diffusion_modeloutput_blocks50out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks50skip_connectionweight",
    "model_ema.diffusion_modeloutput_blocks52in_layers2weight",
    "model_ema.diffusion_modeloutput_blocks52out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks60in_layers2weight",
    "model_ema.diffusion_modeloutput_blocks60out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks60skip_connectionweight",
    "model_ema.diffusion_modeloutput_blocks70in_layers2weight",
    "model_ema.diffusion_modeloutput_blocks70out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks70skip_connectionweight",
    "model_ema.diffusion_modeloutput_blocks80in_layers2weight",
    "model_ema.diffusion_modeloutput_blocks80out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks80skip_connectionweight",
    "model_ema.diffusion_modeloutput_blocks82in_layers2weight",
    "model_ema.diffusion_modeloutput_blocks82out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks90in_layers2weight",
    "model_ema.diffusion_modeloutput_blocks90out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks90skip_connectionweight",
    "model_ema.diffusion_modeloutput_blocks100in_layers2weight",
    "model_ema.diffusion_modeloutput_blocks100out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks100skip_connectionweight",
    "model_ema.diffusion_modeloutput_blocks110in_layers2weight",
    "model_ema.diffusion_modeloutput_blocks110out_layers3weight",
    "model_ema.diffusion_modeloutput_blocks110skip_connectionweight",
]
# model_ema.diffusion_modelout2weight
# model.diffusion_model.out.2.weight
# x = inpaint_conv_keys[0]
# "model_ema" + x[5:]


# size mismatch for model.diffusion_model.input_blocks.0.0.weight: copying a param with shape torch.Size([256, 7, 3, 3]) from checkpoint, the shape in current model is torch.Size([256, 16, 3, 3]).
# size mismatch for model_ema.diffusion_modelinput_blocks00weight: copying a param with shape torch.Size([256, 7, 3, 3]) from checkpoint, the shape in current model is torch.Size([256, 16, 3, 3]).
class InpaintingInit:
    def __init__(self, weights, init3d=False,insert_out_layer=False):
        super().__init__()
        self.weights = weights
        self.init3d = init3d
        self.insert_out_layer = insert_out_layer

    def init(self, model):
        if self.init3d:
            print(f"initializing from {self.weights}")
            init = torch.load(self.weights)
            sd = init["state_dict"]
            for key in inpaint_conv_keys+ema_conv_keys:
                sd[key] = sd[key].unsqueeze(-3)
            if self.insert_out_layer:
                sd['model.diffusion_model.out.4.weight'] = sd['model.diffusion_model.out.2.weight'].clone()
                sd['model.diffusion_model.out.4.bias'] = sd['model.diffusion_model.out.2.bias'].clone()
                sd['model_ema.diffusion_modelout4weight'] = sd['model_ema.diffusion_modelout2weight'].clone()
                sd['model_ema.diffusion_modelout4bias'] = sd['model_ema.diffusion_modelout2bias'].clone()
                del sd['model.diffusion_model.out.2.weight']
                del sd['model.diffusion_model.out.2.bias']
                del sd['model_ema.diffusion_modelout2weight']
                del sd['model_ema.diffusion_modelout2bias']

            res = model.load_state_dict(sd, strict=False)
            def okay_to_miss(x):
                in_cond_stage = re.match('^cond_stage_model.first_stage_model',x)
                if self.insert_out_layer:
                    init_scratch = x in ['model.diffusion_model.out.2.weight', 'model.diffusion_model.out.2.bias', 'model_ema.diffusion_modelout2weight', 'model_ema.diffusion_modelout2bias',]
                else:
                    init_scratch = False
                return (in_cond_stage is not None) or init_scratch
            # lremove(okay_to_miss,res.missing_keys)
            assert all(lmap(okay_to_miss,res.missing_keys))
        else:
            self.init_plain(model)


    def init_plain(self, model):
        print(f"initializing from {self.weights}")
        init = torch.load(self.weights)
        init["state_dict"].keys()
        ow1 = init["state_dict"]["model.diffusion_model.input_blocks.0.0.weight"]
        ow2 = init["state_dict"]["model_ema.diffusion_modelinput_blocks00weight"]
        # change the depth of the kernel to match the new model
        new_shape = (
            ow1.shape[0],
            model.model.diffusion_model.in_channels,
            *ow1.shape[2:],
        )
        # initialize new weights to 0
        nw = torch.zeros(new_shape)
        nw[:, :7] = ow1
        init["state_dict"]["model.diffusion_model.input_blocks.0.0.weight"] = nw
        init["state_dict"]["model_ema.diffusion_modelinput_blocks00weight"] = nw
        # import pdb; pdb.set_trace()
        """
        device = 'cuda:4'
        xin = torch.zeros((1,16,64,64)).to(device)
        t = torch.tensor([100]).to(device)
        model.to(device)
        model.model.diffusion_model(xin,t)
        """
        model.load_state_dict(init["state_dict"], strict=False)


# model.diffusion_model.out.2.weight
# RuntimeError: Error(s) in loading state_dict for LatentDiffusion:
# model.diffusion_model.input_blocks.0.0.weight
# model.diffusion_model.input_blocks.1.0.in_layers.2.weight
# model.diffusion_model.input_blocks.1.0.out_layers.3.weight
# model.diffusion_model.input_blocks.2.0.in_layers.2.weight
# model.diffusion_model.input_blocks.2.0.out_layers.3.weight
# model.diffusion_model.input_blocks.3.0.in_layers.2.weight
# model.diffusion_model.input_blocks.3.0.out_layers.3.weight
# model.diffusion_model.input_blocks.4.0.in_layers.2.weight
# model.diffusion_model.input_blocks.4.0.out_layers.3.weight
# model.diffusion_model.input_blocks.4.0.skip_connection.weight
# model.diffusion_model.input_blocks.5.0.in_layers.2.weight
# model.diffusion_model.input_blocks.5.0.out_layers.3.weight
# model.diffusion_model.input_blocks.6.0.in_layers.2.weight
# model.diffusion_model.input_blocks.6.0.out_layers.3.weight
# model.diffusion_model.input_blocks.7.0.in_layers.2.weight
# model.diffusion_model.input_blocks.7.0.out_layers.3.weight
# model.diffusion_model.input_blocks.7.0.skip_connection.weight
# model.diffusion_model.input_blocks.8.0.in_layers.2.weight
# model.diffusion_model.input_blocks.8.0.out_layers.3.weight
# model.diffusion_model.input_blocks.9.0.in_layers.2.weight
# model.diffusion_model.input_blocks.9.0.out_layers.3.weight
# model.diffusion_model.input_blocks.10.0.in_layers.2.weight
# model.diffusion_model.input_blocks.10.0.out_layers.3.weight
# model.diffusion_model.input_blocks.10.0.skip_connection.weight
# model.diffusion_model.input_blocks.11.0.in_layers.2.weight
# model.diffusion_model.input_blocks.11.0.out_layers.3.weight
# model.diffusion_model.middle_block.0.in_layers.2.weight
# model.diffusion_model.middle_block.0.out_layers.3.weight
# model.diffusion_model.middle_block.2.in_layers.2.weight
# model.diffusion_model.middle_block.2.out_layers.3.weight
# model.diffusion_model.output_blocks.0.0.in_layers.2.weight
# model.diffusion_model.output_blocks.0.0.out_layers.3.weight
# model.diffusion_model.output_blocks.0.0.skip_connection.weight
# model.diffusion_model.output_blocks.1.0.in_layers.2.weight
# model.diffusion_model.output_blocks.1.0.out_layers.3.weight
# model.diffusion_model.output_blocks.1.0.skip_connection.weight
# model.diffusion_model.output_blocks.2.0.in_layers.2.weight
# model.diffusion_model.output_blocks.2.0.out_layers.3.weight
# model.diffusion_model.output_blocks.2.0.skip_connection.weight
# model.diffusion_model.output_blocks.2.2.in_layers.2.weight
# model.diffusion_model.output_blocks.2.2.out_layers.3.weight
# model.diffusion_model.output_blocks.3.0.in_layers.2.weight
# model.diffusion_model.output_blocks.3.0.out_layers.3.weight
# model.diffusion_model.output_blocks.3.0.skip_connection.weight
# model.diffusion_model.output_blocks.4.0.in_layers.2.weight
# model.diffusion_model.output_blocks.4.0.out_layers.3.weight
# model.diffusion_model.output_blocks.4.0.skip_connection.weight
# model.diffusion_model.output_blocks.5.0.in_layers.2.weight
# model.diffusion_model.output_blocks.5.0.out_layers.3.weight
# model.diffusion_model.output_blocks.5.0.skip_connection.weight
# model.diffusion_model.output_blocks.5.2.in_layers.2.weight
# model.diffusion_model.output_blocks.5.2.out_layers.3.weight
# model.diffusion_model.output_blocks.6.0.in_layers.2.weight
# model.diffusion_model.output_blocks.6.0.out_layers.3.weight
# model.diffusion_model.output_blocks.6.0.skip_connection.weight
# model.diffusion_model.output_blocks.7.0.in_layers.2.weight
# model.diffusion_model.output_blocks.7.0.out_layers.3.weight
# model.diffusion_model.output_blocks.7.0.skip_connection.weight
# model.diffusion_model.output_blocks.8.0.in_layers.2.weight
# model.diffusion_model.output_blocks.8.0.out_layers.3.weight
# model.diffusion_model.output_blocks.8.0.skip_connection.weight
# model.diffusion_model.output_blocks.8.2.in_layers.2.weight
# model.diffusion_model.output_blocks.8.2.out_layers.3.weight
# model.diffusion_model.output_blocks.9.0.in_layers.2.weight
# model.diffusion_model.output_blocks.9.0.out_layers.3.weight
# model.diffusion_model.output_blocks.9.0.skip_connection.weight
# model.diffusion_model.output_blocks.10.0.in_layers.2.weight
# model.diffusion_model.output_blocks.10.0.out_layers.3.weight
# model.diffusion_model.output_blocks.10.0.skip_connection.weight
# model.diffusion_model.output_blocks.11.0.in_layers.2.weight
# model.diffusion_model.output_blocks.11.0.out_layers.3.weight
# model.diffusion_model.output_blocks.11.0.skip_connection.weight
# model.diffusion_model.out.2.weight
# model_ema.diffusion_modelinput_blocks00weight
# model_ema.diffusion_modelinput_blocks10in_layers2weight
# model_ema.diffusion_modelinput_blocks10out_layers3weight
# model_ema.diffusion_modelinput_blocks20in_layers2weight
# model_ema.diffusion_modelinput_blocks20out_layers3weight
# model_ema.diffusion_modelinput_blocks30in_layers2weight
# model_ema.diffusion_modelinput_blocks30out_layers3weight
# model_ema.diffusion_modelinput_blocks40in_layers2weight
# model_ema.diffusion_modelinput_blocks40out_layers3weight
# model_ema.diffusion_modelinput_blocks40skip_connectionweight
# model_ema.diffusion_modelinput_blocks50in_layers2weight
# model_ema.diffusion_modelinput_blocks50out_layers3weight
# model_ema.diffusion_modelinput_blocks60in_layers2weight
# model_ema.diffusion_modelinput_blocks60out_layers3weight
# model_ema.diffusion_modelinput_blocks70in_layers2weight
# model_ema.diffusion_modelinput_blocks70out_layers3weight
# model_ema.diffusion_modelinput_blocks70skip_connectionweight
# model_ema.diffusion_modelinput_blocks80in_layers2weight
# model_ema.diffusion_modelinput_blocks80out_layers3weight
# model_ema.diffusion_modelinput_blocks90in_layers2weight
# model_ema.diffusion_modelinput_blocks90out_layers3weight
# model_ema.diffusion_modelinput_blocks100in_layers2weight
# model_ema.diffusion_modelinput_blocks100out_layers3weight
# model_ema.diffusion_modelinput_blocks100skip_connectionweight
# model_ema.diffusion_modelinput_blocks110in_layers2weight
# model_ema.diffusion_modelinput_blocks110out_layers3weight
# model_ema.diffusion_modelmiddle_block0in_layers2weight
# model_ema.diffusion_modelmiddle_block0out_layers3weight
# model_ema.diffusion_modelmiddle_block2in_layers2weight
# model_ema.diffusion_modelmiddle_block2out_layers3weight
# model_ema.diffusion_modeloutput_blocks00in_layers2weight
# model_ema.diffusion_modeloutput_blocks00out_layers3weight
# model_ema.diffusion_modeloutput_blocks00skip_connectionweight
# model_ema.diffusion_modeloutput_blocks10in_layers2weight
# model_ema.diffusion_modeloutput_blocks10out_layers3weight
# model_ema.diffusion_modeloutput_blocks10skip_connectionweight
# model_ema.diffusion_modeloutput_blocks20in_layers2weight
# model_ema.diffusion_modeloutput_blocks20out_layers3weight
# model_ema.diffusion_modeloutput_blocks20skip_connectionweight
# model_ema.diffusion_modeloutput_blocks22in_layers2weight
# model_ema.diffusion_modeloutput_blocks22out_layers3weight
# model_ema.diffusion_modeloutput_blocks30in_layers2weight
# model_ema.diffusion_modeloutput_blocks30out_layers3weight
# model_ema.diffusion_modeloutput_blocks30skip_connectionweight
# model_ema.diffusion_modeloutput_blocks40in_layers2weight
# model_ema.diffusion_modeloutput_blocks40out_layers3weight
# model_ema.diffusion_modeloutput_blocks40skip_connectionweight
# model_ema.diffusion_modeloutput_blocks50in_layers2weight
# model_ema.diffusion_modeloutput_blocks50out_layers3weight
# model_ema.diffusion_modeloutput_blocks50skip_connectionweight
# model_ema.diffusion_modeloutput_blocks52in_layers2weight
# model_ema.diffusion_modeloutput_blocks52out_layers3weight
# model_ema.diffusion_modeloutput_blocks60in_layers2weight
# model_ema.diffusion_modeloutput_blocks60out_layers3weight
# model_ema.diffusion_modeloutput_blocks60skip_connectionweight
# model_ema.diffusion_modeloutput_blocks70in_layers2weight
# model_ema.diffusion_modeloutput_blocks70out_layers3weight
# model_ema.diffusion_modeloutput_blocks70skip_connectionweight
# model_ema.diffusion_modeloutput_blocks80in_layers2weight
# model_ema.diffusion_modeloutput_blocks80out_layers3weight
# model_ema.diffusion_modeloutput_blocks80skip_connectionweight
# model_ema.diffusion_modeloutput_blocks82in_layers2weight
# model_ema.diffusion_modeloutput_blocks82out_layers3weight
# model_ema.diffusion_modeloutput_blocks90in_layers2weight
# model_ema.diffusion_modeloutput_blocks90out_layers3weight
# model_ema.diffusion_modeloutput_blocks90skip_connectionweight
# model_ema.diffusion_modeloutput_blocks100in_layers2weight
# model_ema.diffusion_modeloutput_blocks100out_layers3weight
# model_ema.diffusion_modeloutput_blocks100skip_connectionweight
# model_ema.diffusion_modeloutput_blocks110in_layers2weight
# model_ema.diffusion_modeloutput_blocks110out_layers3weight
# model_ema.diffusion_modeloutput_blocks110skip_connectionweight
# model_ema.diffusion_modelout2weight
