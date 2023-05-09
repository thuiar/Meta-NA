import torch
import torch.nn as nn

class additionFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, t, a, v):
        return t + a + v

class multipleFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, t, a, v):
        return t * a * v

class concatFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, t, a, v):
        return torch.cat((t, a, v), 1)

class tensorFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args


    def forward(self, t, a, v):
        batch_size = a.data.shape[0]
        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        add_one = torch.ones(size=[batch_size, 1], requires_grad=False).type_as(a).to(t.device)
        _audio_h = torch.cat((add_one, a), dim=1)
        _video_h = torch.cat((add_one, v), dim=1)
        _text_h = torch.cat((add_one, t), dim=1)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        
        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fusion_tensor = fusion_tensor.view(-1, (self.args.embd_size_a + 1) * (self.args.embd_size_v + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)
        return fusion_tensor


fusionMethod = {
    'add': additionFusion,
    'mul': multipleFusion,
    'concat': concatFusion,
    'tensor': tensorFusion
}

class Fusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fusion = fusionMethod[args.fusion_method](args)
        
    def forward(self, *args, **kwargs):
        return self.fusion(*args, **kwargs)