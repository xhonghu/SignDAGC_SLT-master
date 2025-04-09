import numpy as np
import os
import glob
import cv2
import yaml
from utils import video_augmentation
from Tokenizer import GlossTokenizer_S2G, TextTokenizer
import torch
from collections import OrderedDict
import utils
from slt_network import SignLanguageModel
from utils.metrics import bleu, rouge 

    
gpu_id = 0 # The GPU to use
### Adjustable parameters
dataset = 'CSL-Daily'  # support [phoenix2014-T, CSL-Daily]
model_weights = f'./{dataset}/best_model.pt'
select_id = 200 
###
sparser = utils.get_parser()
p = sparser.parse_args()
if p.config is not None:
    with open(p.config, 'r') as f:
        try:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            default_arg = yaml.load(f)
    key = vars(p).keys()
    for k in default_arg.keys():
        if k not in key:
            print('WRONG ARG: {}'.format(k))
            assert (k in key)
    sparser.set_defaults(**default_arg)
args = sparser.parse_args()
with open(f"./configs/{dataset}.yaml", 'r') as f:
    args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
prefix = args.dataset_info['dataset_root']
# Load data and apply transformation
gloss_tokenizer = GlossTokenizer_S2G(args.dataset_info['gloss'])
text_tokenizer = TextTokenizer(args.dataset_info['TranslationNetwork']['TextTokenizer'])
inputs_list = np.load(f"./preprocess/{dataset}/dev_info.npy", allow_pickle=True).item()
name = inputs_list[select_id]['fileid']
img_folder = os.path.join(prefix, "features/fullFrame-210x260px/" + inputs_list[select_id]['folder']) if 'phoenix' in dataset else os.path.join(prefix, "sentence/frames_512x512/" +inputs_list[select_id]['folder'])
img_list = sorted(glob.glob(img_folder))
img_list = [cv2.cvtColor(cv2.resize(cv2.imread(img_path), (256, 256), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB) for img_path in img_list]
label_list = gloss_tokenizer([inputs_list[select_id]['label']])
t = text_tokenizer([inputs_list[select_id]['text'].strip()])
src_input = {}
src_input['labels'] = t['labels']
src_input['decoder_input_ids'] = t['decoder_input_ids']
src_input['text'] = inputs_list[select_id]['text'].strip()
src_input['name'] = name
src_input['gloss_ids'] = label_list['gloss_labels']
src_input['gloss_lengths'] = label_list['gls_lengths']  
transform = video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.Resize(1.0),
                video_augmentation.ToTensor(),
            ])
vid = transform(img_list)
vid = vid.float() / 127.5 - 1
vid = vid.unsqueeze(0)
left_pad = 0
last_stride = 1
total_stride = 1
kernel_sizes = ['K5', "P2", 'K5', "P2"]
for layer_idx, ks in enumerate(kernel_sizes):
    if ks[0] == 'K':
        left_pad = left_pad * last_stride 
        left_pad += int((int(ks[1])-1)/2)
    elif ks[0] == 'P':
        last_stride = int(ks[1])
        total_stride = total_stride * last_stride
max_len = vid.size(1)
video_length = torch.LongTensor([int(np.ceil(vid.size(1) / total_stride))* total_stride + 2*left_pad ])
right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
max_len = max_len + left_pad + right_pad
vid = torch.cat(
    (
        vid[0,0][None].expand(left_pad, -1, -1, -1),
        vid[0],
        vid[0,-1][None].expand(max_len - vid.size(1) - left_pad, -1, -1, -1),
    )
    , dim=0).unsqueeze(0)
device = utils.GpuDataParallel()
device.set_device(gpu_id)
# Load model
model = SignLanguageModel(args,gloss_tokenizer)
state_dict = torch.load(model_weights)['model_state_dict']
state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
s_dict = model.state_dict()
for name1 in state_dict:
    if name1 not in s_dict:
        continue
    if s_dict[name1].shape == state_dict[name1].shape:
        s_dict[name1] = state_dict[name1]
model.load_state_dict(s_dict, strict=True)
model = model.to(device.output_device)
model.cuda()
model.eval()
# inference
vid = device.data_to_device(vid)
vid_lgt = device.data_to_device(video_length)
label = device.data_to_device(label_list['gloss_labels'])
label_lgt = device.data_to_device(label_list['gls_lengths'])
src_inputs = {}
src_inputs['labels'] = device.data_to_device(src_input['labels'])
src_inputs['decoder_input_ids'] = device.data_to_device(src_input['decoder_input_ids'])
output = model(vid, vid_lgt, label=label, label_lgt=label_lgt, src_input=src_inputs)
generate_output = model.generate_txt(
    transformer_inputs=output['transformer_inputs'],
    generate_cfg=args.dataset_info['translation'])
txt_ref = [src_input['text']]
txt_hyp = generate_output['decoded_sequences']
if args.dataset_info['level']=='char':
    txt_hyp = [txt_hyp[0].replace(",", "，").replace("?", "？")]
print('Video name is         :',name)
print('Label is              :',txt_ref)
print('The predicted value is:',txt_hyp)
bleu_dict = bleu(references=txt_ref, hypotheses=txt_hyp, level=args.dataset_info['level'])
rouge_score = rouge(references=txt_ref, hypotheses=txt_hyp, level=args.dataset_info['level'])
print('The bleu score is     :',bleu_dict)
print('The rouge score is    :',rouge_score)