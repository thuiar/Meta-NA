import torch
import os
from models.base import BaseModel
from models.networks import seqEncoder, Fusion, FcClassifier,seqOffset
from models.networks.AlignSubNet import AlignSubNet
from transformers import BertModel
import torch.nn as nn
from models.tools import special_sigmoid

class MetaNa(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # Model Parameters
        parser.add_argument('--input_dim_a', type=int, default=5, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=768, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=20, help='visual input dim')
        parser.add_argument('--embd_size_a', default=32, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=64, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=32, type=int, help='visual model embedding size')

        # The seq_lens parameter is used in the AlignSubNet
        parser.add_argument('--seq_lens', type=list, default=[50, 925, 232], help='input batch size')

        parser.add_argument('--fusion_method', default='tensor', type=str, choices=['concat', 'add', 'mul', 'tensor'])
        parser.add_argument('--cls_input_size', default=128, type=int, help='input size of classifier')
        parser.add_argument('--cls_layers', type=str, default='64,64', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--output_dim', type=int, default=1, help='output classification. linear classification')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')

        parser.add_argument('--text_weight', type=float, default=1, help='')
        parser.add_argument('--audio_weight', type=float, default=0.5, help='')
        parser.add_argument('--vision_weight', type=float, default=0.5, help='')
        
        # Meta Parameters.
        parser.add_argument('--approx', type=bool, default=False, help='whether to use first order approximation')
        # Training Parameters.
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--task_inner_size', type=int, default=512, help='input batch size')

        parser.add_argument('--niter', type=int, default=6, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=6, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate for adam')
        parser.add_argument('--alpha_lr', type=float, default=0.0005, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight decay when training')

        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.loss_names = ['l1']
        self.model_names = ['netA', 'netL', 'netV', 'netF', 'netC','netOffsetA','netOffsetL','netOffsetV']
        self.offset_names = ['netOffsetA','netOffsetL','netOffsetV']
        self.backbone_names = ['netA', 'netL', 'netV', 'netF', 'netC']

        self.bert = BertModel.from_pretrained(opt.pretrained)
        self.bert = self.bert.to(self.device)
        self.bert.eval()
        opt.device = self.device
        self.align_subnet = AlignSubNet(opt, mode='avg_pool')
        
        self.netA = seqEncoder(opt.input_dim_a, opt.embd_size_a, dropout=opt.dropout_rate)
        self.netL = seqEncoder(opt.input_dim_l, opt.embd_size_l, dropout=opt.dropout_rate)
        self.netV = seqEncoder(opt.input_dim_v, opt.embd_size_v, dropout=opt.dropout_rate)

        self.netF = Fusion(opt)

        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

        if opt.fusion_method == 'tensor':
            opt.cls_input_size = (opt.embd_size_a+ 1) * (opt.embd_size_l + 1) * (opt.embd_size_v + 1)
        elif opt.fusion_method == 'add' or opt.fusion_method == 'mul':
            opt.cls_input_size = opt.embd_size_a
        
        self.netC = FcClassifier(opt.cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate)
        
        self.set_offset_model()
        self.set_alpha_model()

        self.fast_parameters = []
        if self.isTrain:
            self.criterion_L1 = opt.criterion_loss
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.paremeters_backbone = [{'params': getattr(self, net).parameters()} for net in self.backbone_names]
            self.backbone_optimizer = torch.optim.Adam(self.paremeters_backbone, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizers.append(self.backbone_optimizer)

            self.paremeters_offset = [{'params': getattr(self, net).parameters()} for net in self.offset_names]
            self.offset_optimizer = torch.optim.Adam(self.paremeters_offset, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizers.append(self.offset_optimizer)

            paremeters_alpha = [{'params': getattr(self, f"{net}_alpha").parameters(), 'lr':opt.alpha_lr,'betas':(opt.beta1, 0.999), 'weight_decay':opt.weight_decay} for net in self.backbone_names]
            self.optimizer_alpha = torch.optim.Adam(paremeters_alpha)

            self.output_dim = opt.output_dim

        # modify save_dir
        if not os.path.exists(self.save_dir) and opt.save_model:
            os.mkdir(self.save_dir)
        self.save_dir = os.path.join(self.save_dir, str(opt.seed))
        if not os.path.exists(self.save_dir) and opt.save_model:
            os.mkdir(self.save_dir)

    def set_offset_model(self):
        self.netOffsetA = seqOffset(self.opt.embd_size_a)
        self.netOffsetV = seqOffset(self.opt.embd_size_v)
        self.netOffsetL = seqOffset(self.opt.embd_size_l)

    def set_alpha_model(self):
        for m in self.model_names:
            lrs = [torch.ones_like(p).to(self.device) * self.opt.inner_lr for p in getattr(self, m).parameters()]
            lrs = nn.ParameterList([nn.Parameter(lr) for lr in lrs])
            setattr(self, f"{m}_alpha", lrs)

    def forward(self):
        return super().forward()

    def offset_backward(self,losses):
        losses.backward()
        for m in self.offset_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, m).parameters(), 0.8)

    def backbone_backward(self,losses):
        losses.backward()
        for m in self.backbone_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, m).parameters(), 0.8)

    def set_input(self, input):
        return super().set_input(input)

    def optimize_offset_parameters(self,losses):
        self.offset_zero_grad()
        self.zero_grad_alpha()
        self.offset_backward(losses)
        self.offset_optimizer.step()
        self.optimizer_alpha.step()

    def optimize_parameters(self,losses):
        self.backbone_zero_grad()
        self.backbone_backward(losses)
        self.backbone_optimizer.step()
            
    def inner_train(self, tr_loader):
        self.net_reset()
        for i in range(self.opt.inner_step):

            for batch_data in tr_loader:

                text_bert = batch_data['text_bert'].float().to(self.device)
                audio = batch_data['audio'].float().to(self.device)
                vision = batch_data['vision'].float().to(self.device)

                with torch.no_grad():
                    text = self.bert(
                        input_ids=text_bert[:,0,:].long(),
                        attention_mask=text_bert[:,1,:].long(), 
                        token_type_ids=text_bert[:,2,:].long())[0]
                    
                if self.opt.dataset != 'mosi':
                    text, audio, vision = self.align_subnet(text,audio,vision,batch_data['text_len'], batch_data['audio_len'], batch_data['vision_len'])
                
                text = self.netL(text)
                audio = self.netA(audio)
                vision = self.netV(vision)

                text = self.netOffsetL(text)
                audio = self.netOffsetA(audio)
                vision = self.netOffsetV(vision)

                fusion = self.netF(text, audio, vision)
                logits, _ = self.netC(fusion)

                set_loss = self.criterion_L1(logits, batch_data['label'].float().to(self.device))

                grad = torch.autograd.grad(set_loss, self.fast_parameters, create_graph=True)

                if self.opt.approx:
                    grad = [g.detach() for g in grad]

                self.fast_parameters = []

                pointer = 0
                for m in self.backbone_names:
                    for weight,alpha in zip(getattr(self, m).parameters(),getattr(self, f"{m}_alpha").parameters()):
                        # alpha is limited to 0 to 0.03
                        alpha = special_sigmoid(alpha)
                        if weight.fast is None:
                            weight.fast = weight - torch.mul(alpha, grad[pointer])# create weight.fast 
                        else:
                            weight.fast = weight.fast - torch.mul(alpha, grad[pointer]) # update weight.fast
                        pointer += 1
                        self.fast_parameters.append(weight.fast) # gradients are based on newest weights, but the graph will retain the link to old weight.fasts


    
    def inner_eval(self, ts_loader):
        logits, labels = [], []
        
        for batch_data in ts_loader:

            text_bert = batch_data['text_bert'].float().to(self.device)
            audio = batch_data['audio'].float().to(self.device)
            vision = batch_data['vision'].float().to(self.device)

            clean_text_bert = batch_data['clean_text_bert'].float().to(self.device)
            clean_audio = batch_data['clean_audio'].float().to(self.device)
            clean_vision = batch_data['clean_vision'].float().to(self.device)

            with torch.no_grad():
                text = self.bert(
                    input_ids=text_bert[:,0,:].long(),
                    attention_mask=text_bert[:,1,:].long(),
                    token_type_ids=text_bert[:,2,:].long())[0]
                
                clean_text = self.bert(
                    input_ids=clean_text_bert[:,0,:].long(),
                    attention_mask=clean_text_bert[:,1,:].long(), 
                    token_type_ids=clean_text_bert[:,2,:].long())[0]
            
            if self.opt.dataset != 'mosi':
                text, audio, vision = self.align_subnet(text,audio,vision,batch_data['text_len'], batch_data['audio_len'], batch_data['vision_len'])
                clean_text, clean_audio, clean_vision = self.align_subnet(clean_text,clean_audio,clean_vision,batch_data['text_len'], batch_data['audio_len'], batch_data['vision_len'])

            text = self.netL(text)
            audio = self.netA(audio)
            vision = self.netV(vision)
            
            clean_text = self.netL(clean_text)
            clean_audio = self.netA(clean_audio)
            clean_vision = self.netV(clean_vision)

            text = self.netOffsetL(text)
            audio = self.netOffsetA(audio)
            vision = self.netOffsetV(vision)
            
            text_loss = torch.mean(torch.absolute(text-clean_text))
            audio_loss = torch.mean(torch.absolute(audio-clean_audio))
            vision_loss = torch.mean(torch.absolute(vision-clean_vision))

            feature_losses = self.opt.text_weight*text_loss + self.opt.audio_weight*audio_loss + self.opt.vision_weight*vision_loss
            fusion = self.netF(text, audio, vision)
            logit, _ = self.netC(fusion)

            logits.append(logit)
            labels.append(batch_data['label'].float().to(self.device))

        return logits,labels,feature_losses
    
    def outer_train(self, tr_loader):
        
        self.net_reset()
        logits, labels = [], []
        for batch_data in tr_loader:

            text_bert = batch_data['text_bert'].float().to(self.device)
            audio = batch_data['audio'].float().to(self.device)
            vision = batch_data['vision'].float().to(self.device)

            with torch.no_grad():
                text = self.bert(
                    input_ids=text_bert[:,0,:].long(),
                    attention_mask=text_bert[:,1,:].long(), 
                    token_type_ids=text_bert[:,2,:].long())[0]
                
            if self.opt.dataset != 'mosi':
                text, audio, vision = self.align_subnet(text,audio,vision,batch_data['text_len'], batch_data['audio_len'], batch_data['vision_len'])
    
            text = self.netL(text)
            audio = self.netA(audio)
            vision = self.netV(vision)

            text = self.netOffsetL(text)
            audio = self.netOffsetA(audio)
            vision = self.netOffsetV(vision)

            fusion = self.netF(text, audio, vision)
            logit, _ = self.netC(fusion)

            logits.append(logit)
            labels.append(batch_data['label'].float().to(self.device))
        return logits,labels
    
    def finetune_inner_train(self, text, audio, vision, label):
        self.net_reset()
        text = self.netL(text)
        audio = self.netA(audio)
        vision = self.netV(vision)

        text = self.netOffsetL(text)
        audio = self.netOffsetA(audio)
        vision = self.netOffsetV(vision)

        fusion = self.netF(text, audio, vision)
        logits, _ = self.netC(fusion)

        set_loss = self.criterion_L1(logits, label.float().to(self.device))

        grad = torch.autograd.grad(set_loss, self.fast_parameters, create_graph=True)

        if self.opt.approx:
            grad = [g.detach() for g in grad]

        self.fast_parameters = []

        pointer = 0
        for m in self.backbone_names:
            for weight,alpha in zip(getattr(self, m).parameters(),getattr(self, f"{m}_alpha").parameters()):
                
                # alpha is limited to 0 to 0.03
                alpha = special_sigmoid(alpha)

                if weight.fast is None:
                    weight.fast = weight - torch.mul(alpha, grad[pointer])# create weight.fast 
                else:
                    weight.fast = weight.fast - torch.mul(alpha, grad[pointer]) # update weight.fast
                pointer += 1
                self.fast_parameters.append(weight.fast) # gradients are based on newest weights, but the graph will retain the link to old weight.fasts
    

    def finetune_inner_eval(self, noise_clean_data):
        logits, labels = [], []
        for batch_data in noise_clean_data:
            text_bert = batch_data['noise_text_bert'].float().to(self.device)
            audio = batch_data['noise_audio'].float().to(self.device)
            vision = batch_data['noise_vision'].float().to(self.device)

            clean_text_bert = batch_data['clean_text_bert'].float().to(self.device)
            clean_audio = batch_data['clean_audio'].float().to(self.device)
            clean_vision = batch_data['clean_vision'].float().to(self.device)

            with torch.no_grad():
                text = self.bert(
                    input_ids=text_bert[:,0,:].long(),
                    attention_mask=text_bert[:,1,:].long(),
                    token_type_ids=text_bert[:,2,:].long())[0]
                
                clean_text = self.bert(
                    input_ids=clean_text_bert[:,0,:].long(),
                    attention_mask=clean_text_bert[:,1,:].long(), 
                    token_type_ids=clean_text_bert[:,2,:].long())[0]
                
            if self.opt.dataset != 'mosi':
                text, audio, vision = self.align_subnet(text,audio,vision,batch_data['text_len'], batch_data['audio_len'], batch_data['vision_len'])
                clean_text, clean_audio, clean_vision = self.align_subnet(clean_text,clean_audio,clean_vision,batch_data['text_len'], batch_data['audio_len'], batch_data['vision_len'])

            text = self.netL(text)
            audio = self.netA(audio)
            vision = self.netV(vision)
            
            clean_text = self.netL(clean_text)
            clean_audio = self.netA(clean_audio)
            clean_vision = self.netV(clean_vision)

            text = self.netOffsetL(text)
            audio = self.netOffsetA(audio)
            vision = self.netOffsetV(vision)
            
            text_loss = torch.mean(torch.absolute(text-clean_text))
            audio_loss = torch.mean(torch.absolute(audio-clean_audio))
            vision_loss = torch.mean(torch.absolute(vision-clean_vision))

            feature_losses = self.opt.text_weight*text_loss + self.opt.audio_weight*audio_loss + self.opt.vision_weight*vision_loss
            fusion = self.netF(text, audio, vision)
            logit, _ = self.netC(fusion)

            logits.append(logit)
            labels.append(batch_data['label'].float().to(self.device))
        return logits,labels,feature_losses
    
    def finetune_outer_train(self, text, audio, vision, label):
        self.net_reset()
        logits, labels = [], []
        text = self.netL(text)
        audio = self.netA(audio)
        vision = self.netV(vision)

        text = self.netOffsetL(text)
        audio = self.netOffsetA(audio)
        vision = self.netOffsetV(vision)

        fusion = self.netF(text, audio, vision)
        logit, _ = self.netC(fusion)

        logits.append(logit)
        labels.append(label.float().to(self.device))
        return logits,labels
    
    def net_reset(self):
        self.fast_parameters = self.get_inner_loop_params()
        for m in self.backbone_names:
            for weight in getattr(self, m).parameters():  # reset fast parameters
                weight.fast = None

    def get_inner_loop_params(self):
        inner_loop_p = []
        for m in self.backbone_names:
            inner_loop_p.extend(list(getattr(self, m).parameters()))
        return inner_loop_p

    def offset_zero_grad(self):
        for m in self.offset_names:
            getattr(self, m).zero_grad()

    def backbone_zero_grad(self):
        for m in self.backbone_names:
            getattr(self, m).zero_grad()

    def zero_grad_alpha(self):
        for m in self.backbone_names:
            getattr(self, f"{m}_alpha").zero_grad()