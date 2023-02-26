import logging
import os
import time
import torch

from config import parse_args
from data_helper import create_dataloaders
from util import *
import torch.nn as nn
import tqdm as tqdm
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import collections
import copy

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
import gc
import numpy as np

from model_hunliu_small import MultiModal
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")
def evaluation(model, val_dataloader, args):
    model.eval()
    metric = {}
    preds, labels = [], []
    props = []
    data_result_dict = {}
    val_loss = 0.
    txt_all = []
    with torch.no_grad():
        for batch in val_dataloader:
            title_input = batch['title_input'].to(args.device)
            title_mask= batch['title_mask'].to(args.device)

            targets = batch['label'].squeeze(dim=1).squeeze(dim=1).to(args.device)
            outputs = model(title_input, title_mask,targets)
            cirtion = nn.BCEWithLogitsLoss(reduction="mean")
            loss = cirtion(outputs, targets)
            outputs = torch.sigmoid(outputs)

            val_loss += loss.item()
            outtmp = outputs.cpu().numpy()
            # outtmp_result = np.int32(outtmp >= 0.5).tolist()
                
            preds.extend(outtmp)
            labels.extend(np.int32(targets.cpu().numpy()).tolist())
    avg_val_loss = val_loss / len(val_dataloader)
    
    
    preds1 = np.array(preds)
    
    pres_z = np.int32(preds1>= 0.5).tolist()
    metric_ttt = metric_f1_cx(pres_z, labels)   
    

    
    best_52 = np.array([1.00997009,1.00589623, 1.02881844, 1.03095975, 1.09090909, 1.05235602,
 1.01162791, 1.00602047, 1.19230769, 1.0591716,  1.04385965, 1.52631579,
 1.00345185, 1.01838235, 1.02061856 ,1.0057971 , 1.05434783 ,1.05235602,
 1.14705882, 1.04048583 ,1.02008032 ,1.02309469 ,1.20408163, 1.18181818,
 1.02832861, 1.35714286 ,1.03690037 ,1.08403361, 1.58823529, 1.0140056,
 1.05524862, 1.00805802 ,1.10309278 ,1.03012048, 1.0097561 , 1.01251564,
 1.04950495, 1.00865052 ,1.06451613 ,1.04854369, 1.05617978, 1.05952381,
 1.5 ,       1.04016064 ,1.08064516, 1.01605136, 1.12345679, 1.01677852,
 1.03597122, 1.01207729 ,1.06756757 ,1.12345679])
    
#     pp = np.array(preds)
#     for i in range(52):
#         pp[:,i] = np.int32(pp[:,i] >= best_52[i])
#     pres_z = np.int32(pp>= 0.5).tolist()
    
    pp = np.array(preds)*best_52
    pres_z = np.int32(pp>= 0.4).tolist()

    metric = metric_f1_cx(pres_z, labels)   
    metric['avg_val_loss'] =  round(avg_val_loss, 4)
    metric_ttt['avg_val_loss'] =  round(avg_val_loss, 4)
    print(metric_ttt['f1'], metric['f1'])
    return metric_ttt, best_52


def metric_f1_cx(preds, labels):
    metric = {}
    acc_all = []
    f1_all = []
    labels = np.array(labels)
    preds = np.array(preds)
    for idx in range(52):
        lab_tmp = labels[:, idx]
        pred_tmp = preds[:, idx]
        acc_tmp, f1_tmp = accuracy_score(y_true=lab_tmp, y_pred=pred_tmp), f1_score(y_true=lab_tmp, y_pred=pred_tmp, average='binary')
        acc_all.append(acc_tmp)
        f1_all.append(f1_tmp)

    acc = np.mean(acc_all)
    f1 = np.mean(f1_all)
    # acc, f1 = round(acc, 4), round(f1, 4)
    metric['acc'], metric['f1'] = acc, f1
    return metric
    
    

def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = MultiModal(args.bert_dir)
    # args.pre_ckpt_file = './save/model_hunliu_small_best_2021_60845.pt'
    #
    if args.pre_ckpt_file != '':
        checkpoint = torch.load(args.pre_ckpt_file)
        # model.load_state_dict(checkpoint['model_state_dict'])
        model_dict = model.state_dict()
        checkpoint = {k: v for k, v in checkpoint.items()
                      if k in model_dict.keys()}

        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)

        print('+++++++++++++加载的预训练模型是：', args.pre_ckpt_file)
        
        
#     model.load_state_dict(torch.load('./save/model_hunliu_small_best_202160186.pt', map_location='cpu'))
    
#     print('那个在加载。。。。。model_hunliu_small_best_202160186.pt')
    # for name, paramer in model.named_parameters():
    #     if paramer.requires_grad:
    #         print(name)
    
    optimizer, scheduler = build_optimizer(args, model)

    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    
    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()

    num_total_steps = len(train_dataloader) * args.max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_total_steps * 0.1,
                                                num_training_steps=num_total_steps)

    if 'fgm' in args.attack_func:
        attack_func = FGM(model)
        print('Enable FGM')
    elif 'pgd' in args.attack_func:
        attack_func = PGD(model)
        print('Enable PGD')
    elif 'awp' in args.attack_func:
        attack_func = AWP(model)
        print('Enable AWP')

    if args.ema:
        ema = EMA(model, 0.999)
        ema.register()
        print('Enable EMA--0.999')
    scaler = GradScaler()
    print('Enable scaler')
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            
            if batch['title_input'].shape[0] != args.batch_size:
                continue
            model.train()
            targets = batch['label'].squeeze(dim=1).to(args.device)
            title_input = batch['title_input'].to(args.device)
            title_mask= batch['title_mask'].to(args.device)
            helpfu =  batch['helpfu'].to(args.device)
            
            with autocast():
                loss = model(title_input, title_mask,targets, helpfu)
            loss = loss.mean()
            scaler.scale(loss).backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if 'fgm' in args.attack_func:
                attack_func.attack()
                # optimizer.zero_grad()
                with autocast():
                    loss_adv = model(title_input, title_mask,targets, helpfu)
                loss_adv = loss_adv.mean()
                scaler.scale(loss_adv).backward()
                attack_func.restore()

            elif 'pgd' in args.attack_func or 'awp' in args.attack_func:
                attack_func.backup_grad()
                awp_k = 1
                for t in range(awp_k):
                    attack_func.attack(is_first_attack=(t == 0))
                    if t != awp_k - 1:
                        optimizer.zero_grad()
                    else:
                        attack_func.restore_grad()
                    with autocast():
                        loss_adv = model(title_input, targets)
                    loss_adv = loss_adv.mean()
                    scaler.scale(loss_adv).backward()
                attack_func.restore()

                # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            if args.ema:
                ema.update()

            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if epoch >= 3:
                args.print_steps = 300
            else:
                args.print_steps = 300

            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}")

                if args.ema:
                    ema.apply_shadow()

                results,best_52 = evaluation(model, val_dataloader, args)
                acc, f1, avg_val_loss = results['acc'], results['f1'], results['avg_val_loss']
                
                logging.info(f"Epoch {epoch} step {step}: {results}")
                if args.ema:
                    ema.restore()

                # 5. save checkpoint
                mean_f1 = f1
                if mean_f1 >= best_score:
                    if args.ema:
                        ema.apply_shadow()
                    best_score = mean_f1
                    state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()

#                     torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
#                                f'{args.savedmodel_path}/model_hunliu_small_best_{args.seed}.bin')

                    torch.save(state_dict, f'{args.savedmodel_path}/model_hunliu_small_best_{args.seed}.pt')
                    np.save( f'{args.savedmodel_path}/model_hunliu_small_best_{args.seed}.npy', np.array(best_52))
        
                    # print('已保存:', best_52) 


                    if args.ema:
                        ema.restore()

def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
