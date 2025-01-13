import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
from utils import video_augmentation, clean_phoenix_2014_trans
from Tokenizer import GlossTokenizer_S2G, TextTokenizer
from utils import clean_phoenix_2014_trans,clean_phoenix_2014,clean_csl
sys.path.append("..")
global kernel_sizes 

class BaseFeeder(data.Dataset):
    def __init__(self, prefix, tokenizer, gloss_tokenizer, dataset='phoenix2014', drop_ratio=1, num_gloss=-1, mode="train", 
                 transform_mode=True,
                 datatype="lmdb", frame_interval=1, image_scale=1.0, kernel_size=1, input_size=224):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.data_type = datatype
        self.dataset = dataset
        self.input_size = input_size
        global kernel_sizes 
        kernel_sizes = kernel_size
        self.frame_interval = frame_interval # not implemented for read_features()
        self.image_scale = image_scale # not implemented for read_features()
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        #/data1/gsw/CSL-Daily/sentence/frames_512x512
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"./preprocess/{dataset}/{mode}_info.npy", allow_pickle=True).item()
        self.gloss_tokenizer = GlossTokenizer_S2G(gloss_tokenizer)
        self.text_tokenizer = TextTokenizer(tokenizer)
        print(mode, len(self))
        self.data_aug = self.transform()

    def __getitem__(self, idx):
        input_data, fi = self.read_video(idx)
        input_data = self.normalize(input_data)
        if self.dataset=='phoenix2014':
            fi['label'] = clean_phoenix_2014(fi['label'])
        if self.dataset=='phoenix2014-T':
            fi['label'] = clean_phoenix_2014_trans(fi['label'])
        if self.dataset=='CSL-Daily':
            fi['label'] = clean_csl(fi['label'])   
        gloss = fi['label']
        return input_data, gloss, fi['text'].strip(), fi['fileid'] ,fi

    def read_video(self, index):
        # load file info
        fi = self.inputs_list[index]
        if 'phoenix' in self.dataset:
            img_folder = os.path.join(self.prefix, "features/fullFrame-210x260px/" + fi['folder'])
        elif self.dataset == 'CSL-Daily':
            img_folder = os.path.join(self.prefix, "sentence/frames_512x512/" + fi['folder'])
        img_list = sorted(glob.glob(img_folder))
        img_list = img_list[int(torch.randint(0, self.frame_interval, [1]))::self.frame_interval]
        return [cv2.cvtColor(cv2.resize(cv2.imread(img_path), (256, 256), interpolation=cv2.INTER_LANCZOS4),
                             cv2.COLOR_BGR2RGB) for img_path in img_list], fi

    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def normalize(self, video):
        video = self.data_aug(video)
        video = video.float() / 127.5 - 1
        return video

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(self.input_size),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2, self.frame_interval),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(self.input_size),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
            ])

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    def collate_fn(self, batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, gloss, text, name, info = list(zip(*batch))
        gloss_input = self.gloss_tokenizer(gloss)
        t = self.text_tokenizer(text)
        src_input = {}
        src_input['labels'] = t['labels']
        src_input['decoder_input_ids'] = t['decoder_input_ids']
        src_input['text'] = list(text)
        src_input['name'] = list(name)
        src_input['gloss_ids'] = gloss_input['gloss_labels']
        src_input['gloss_lengths'] = gloss_input['gls_lengths']       
        left_pad = 0
        last_stride = 1
        total_stride = 1
        global kernel_sizes 
        for layer_2idx, ks in enumerate(kernel_sizes):
            if ks[0] == 'K':
                left_pad = left_pad * last_stride 
                left_pad += int((int(ks[1])-1)/2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride = total_stride * last_stride
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            # video_length = torch.LongTensor([np.ceil(len(vid) / total_stride) * total_stride + 2*left_pad for vid in video])
            video_length = torch.LongTensor([int(np.ceil(len(vid) / total_stride))* total_stride + 2*left_pad for vid in video])
            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = gloss_input['gls_lengths']
        label = gloss_input['gloss_labels']
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for i, length in enumerate(label_length):
                slice_i = label[i, :length]  # 从第 i 块中切出长度为 length 的切片
                padded_label.append(slice_i)
            padded_label = torch.LongTensor(torch.cat(padded_label))
            return padded_video, video_length, padded_label, label_length, src_input, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()