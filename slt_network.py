import torch
import numpy as np
from collections import OrderedDict
from Tokenizer import GlossTokenizer_S2G
from translation import TranslationNetwork
from vl_mapper import VLMapper
from slr_network import SLRModel


def modified_weights(state_dict, modified=False):
    state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
    if not modified:
        return state_dict
    modified_dict = dict()
    return modified_dict

def load_model_weights(model, weight_path):
    state_dict = torch.load(weight_path)
    weights = modified_weights(state_dict['model_state_dict'], False)
    s_dict = model.state_dict()
    for name in weights:
        if name not in s_dict:
            continue
        if s_dict[name].shape == weights[name].shape:
            s_dict[name] = weights[name]
    model.load_state_dict(s_dict, strict=True)


class SignLanguageModel(torch.nn.Module):
    def __init__(self, args,gloss_dict):
        super().__init__()
        self.args = args
        self.frozen_modules = []
        self.recognition_network = SLRModel(**self.args.model_args, gloss_dict=gloss_dict, loss_weights=self.args.loss_weights,)
        if self.args.dataset_info['pretrained_path']:
            load_model_weights(self.recognition_network, self.args.dataset_info['pretrained_path'])
        self.translation_network = TranslationNetwork(cfg=args.dataset_info['TranslationNetwork'])
        self.gloss_tokenizer = gloss_dict
        if self.args.dataset_info['VLMapper'].get('type','projection') == 'projection':
            self.vl_mapper = VLMapper(cfg=self.args.dataset_info['VLMapper'],
                                      in_features=1024,
                                      out_features=self.translation_network.input_dim,)
        else:
            in_features = len(self.gloss_tokenizer)
            self.vl_mapper = VLMapper(cfg=self.args.dataset_info['VLMapper'],
                                      in_features=in_features,
                                      out_features=self.translation_network.input_dim,
                                      gloss_id2str=self.gloss_tokenizer.id2gloss,
                                      gls2embed=getattr(self.translation_network, 'gls2embed', None))

    def forward(self, vid, vid_lgt, label, label_lgt, src_input, **kwargs):
        recognition_outputs = self.recognition_network(vid, vid_lgt, label=label, label_lgt=label_lgt)
        recognition_loss = self.recognition_network.criterion_calculation(recognition_outputs, label, label_lgt)
        mapped_feature = self.vl_mapper(visual_outputs=recognition_outputs)
        translation_inputs = {
            **src_input,
            'input_feature': mapped_feature,
            'input_lengths': recognition_outputs['feat_len']}
        translation_outputs = self.translation_network(**translation_inputs)
        model_outputs = {**translation_outputs}
        model_outputs['transformer_inputs'] = model_outputs['transformer_inputs']  # for latter use of decoding
        model_outputs['total_loss'] = recognition_loss + model_outputs['translation_loss']
        return model_outputs


    def generate_txt(self, transformer_inputs=None, generate_cfg={}, **kwargs):
            model_outputs = self.translation_network.generate(**transformer_inputs, **generate_cfg)
            return model_outputs