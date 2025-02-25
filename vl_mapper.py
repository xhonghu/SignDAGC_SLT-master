import torch
import torch.nn as nn
import torch.nn.functional as F

class VLMapper(torch.nn.Module):
    def __init__(self, cfg, in_features, out_features,
        gloss_id2str=None,
        gls2embed=None) -> None:
        super().__init__()
        self.type = cfg.get('type','projection')
        if self.type == 'projection':
            self.hidden_size = in_features
            self.mapping = torch.nn.Sequential(
                torch.nn.Linear(in_features=in_features * 2 , out_features=self.hidden_size),
                # torch.nn.ReLU(),
                # torch.nn.Linear(in_features=self.hidden_size , out_features=self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=self.hidden_size, out_features=out_features)
            )
        elif self.type == 'embedding':
            self.mapping = torch.nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=False)
            assert in_features==len(gloss_id2str), (in_features, gloss_id2str)
            with torch.no_grad():
                for i,s in gloss_id2str.items():
                    if s in gls2embed:
                        self.mapping.weight[:, i] = gls2embed[s]
                    else:
                        self.mapping.weight[:, i] = 0

    
    def forward(self, visual_outputs, lengths=None):
        if self.type=='projection':
            output = self.mapping(torch.cat((visual_outputs['gloss_feature'].permute(1, 0, 2), visual_outputs['conv_feature'].permute(1, 0, 2)), dim=-1))
        elif self.type=='embedding':
            output = self.mapping(visual_outputs['sequence_logits'].softmax(-1).permute(1, 0, 2))
        else:
            raise ValueError
        return output