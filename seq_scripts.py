import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from einops import rearrange
from collections import defaultdict
from utils.metrics import bleu, rouge 
from utils.misc import *
from utils.video_augmentation import *
import gc

def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    scaler = GradScaler()
    for batch_idx, data in enumerate(tqdm(loader)):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        src_inputs = {}
        src_input = data[4]
        src_inputs['labels'] = device.data_to_device(src_input['labels'])
        src_inputs['decoder_input_ids'] = device.data_to_device(src_input['decoder_input_ids'])
        optimizer.zero_grad()
        with autocast():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt, src_input=src_inputs)
            loss = ret_dict['total_loss']
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print('loss is nan')
            print(str(data[1])+'  frames', str(data[3])+'  glosses')
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer.optimizer)
        scaler.update() 
        if len(device.gpu_list)>1:
            torch.cuda.synchronize() 
            torch.distributed.reduce(loss, dst=0)
        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0 and is_main_process():
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.8f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[2]))
        del vid
        del vid_lgt
        del label
        del label_lgt
        del ret_dict
        del loss
    optimizer.scheduler.step()
    if is_main_process():
        recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    del loss_value
    del clr
    gc.collect()
    return


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder, generate_cfg):
    model.eval()
    results=defaultdict(dict)
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader)):
            recoder.record_timer("device")
            vid = device.data_to_device(data[0])
            vid_lgt = device.data_to_device(data[1])
            label = device.data_to_device(data[2])
            label_lgt = device.data_to_device(data[3])
            src_inputs = {}
            src_input = data[4]
            src_inputs['labels'] = device.data_to_device(src_input['labels'])
            src_inputs['decoder_input_ids'] = device.data_to_device(src_input['decoder_input_ids'])
            output = model(vid, vid_lgt, label=label, label_lgt=label_lgt, src_input=src_inputs)
            generate_output = model.generate_txt(
                transformer_inputs=output['transformer_inputs'],
                generate_cfg=generate_cfg)
            for name, txt_hyp, txt_ref in zip(src_input['name'], generate_output['decoded_sequences'],
                                                src_input['text']):
                results[name]['txt_hyp'], results[name]['txt_ref'] = txt_hyp, txt_ref
            del vid
            del vid_lgt
            del label
            del label_lgt
            del output
        txt_ref = [results[n]['txt_ref'] for n in results]
        if cfg.dataset_info['level']=='char':
            # 将预测值标点符号转换成中文符号。
            txt_hyp = [results[n]['txt_hyp'].replace(",", "，").replace("?", "？") for n in results]
        else:
            txt_hyp = [results[n]['txt_hyp'] for n in results]
        if epoch==6667:
            name = [n for n in results]
            with open(f"{work_dir}/{mode}_visual.txt", "w") as file:
                for item in range(len(txt_ref)):
                    file.write('fileid  :  '+name[item] + "\n")
                    file.write('GT      :  '+txt_ref[item] + "\n")
                    file.write('Predict :  '+txt_hyp[item] +"  Bleu4:"+str(
                        bleu(references=[txt_ref[item]], hypotheses=[txt_hyp[item]], level=cfg.dataset_info['level'])[
                            'bleu4']) +'  Rouge:'+str(rouge(references=[txt_ref[item]], hypotheses=[txt_hyp[item]],
                                                              level=cfg.dataset_info['level'])) +"\n\n")      
        bleu_dict = bleu(references=txt_ref, hypotheses=txt_hyp, level=cfg.dataset_info['level'])
        rouge_score = rouge(references=txt_ref, hypotheses=txt_hyp, level=cfg.dataset_info['level'])
        for k, v in bleu_dict.items():
            print('{} {:.2f}'.format(k, v))
        print('ROUGE: {:.2f}'.format(rouge_score))
        bleu1=bleu_dict['bleu1']
        bleu2=bleu_dict['bleu2']
        bleu3=bleu_dict['bleu3']
        bleu4=bleu_dict['bleu4']
        recoder.print_log('\t {} {} done. bleu1: {:.4f}, bleu2:{:.4f}, bleu3:{:.4f}, bleu4:{:.4f}, rouge:{:.4f}'.format(epoch,
            mode, bleu1, bleu2, bleu3, bleu4, rouge_score), f"{work_dir}/{mode}.txt")
        gc.collect()
    return {"bleu1": bleu1, "bleu2":bleu2, "bleu3":bleu3, "bleu4":bleu4, "rouge":rouge_score}
 
