import torch
import os
import time
import numpy as np
from torch.utils.data import DataLoader
from tasks.dataset import BaseDataset,NoiseCleanDataset
from models import create_model
from tasks import create_task,creatNoiseClean
from utils import setup_seed, Options, MetricsTop, get_logger
from tqdm import tqdm
from utils import *

def train_outer(e, model, tasks_generator, metrics, opt, logger):
    losses, outer_losses, epoch_start_time = 0, 0, time.time()
    qry_clean_data = {
        'audio':tasks_generator.vl_audio,
        'id':tasks_generator.vl_id,
        'label':tasks_generator.vl_label,
        'text':tasks_generator.vl_text,
        'vision':tasks_generator.vl_vision
    }
    for n in range(len(tasks_generator.datasets_cache['meta_train'])):
        task_ = tasks_generator.datasets_cache['meta_train'][n]
        logger.info(f"Epoch {e} - Task {n}: \n {str(task_)}:")

        sup_x, sup_y, qry_x, qry_y = task_.get_task_d()
        tr_data, ts_data = BaseDataset(sup_x, sup_y), BaseDataset(qry_x, qry_y, qry_clean_data)
        tr_loader = DataLoader(tr_data, batch_size=opt.inner_batch_size, shuffle=True)
        ts_loader = DataLoader(ts_data, batch_size=opt.inner_batch_size, shuffle=False)
        
        model.inner_train(tr_loader)

        preds, labels,feature_losses = model.inner_eval(ts_loader)
        preds, labels = torch.cat(preds, dim=0), torch.cat(labels, dim=0)
        losses = opt.criterion_loss(preds, labels)
        losses += feature_losses
        que_result = metrics(preds.cpu().detach().numpy(), labels.cpu().detach().numpy())
        logger.info(f"Time Taken: {time.time() - epoch_start_time} sec\n, Test out Results: {que_result}")
        model.optimize_offset_parameters(losses)

        preds, labels = model.outer_train(tr_loader)
        preds, labels = torch.cat(preds, dim=0), torch.cat(labels, dim=0)
        outer_losses += opt.criterion_loss(preds, labels)
        sup_result = metrics(preds.cpu().detach().numpy(), labels.cpu().detach().numpy())
        logger.info(f"Time Taken: {time.time() - epoch_start_time} sec\n, Train out Results: {sup_result}")
        model.optimize_parameters(outer_losses)


def finetune(model,  metrics, tr_loader, vl_loader, ts_loader, nc_data, opt, logger, result_file):
    # Loop over the Evalutation Tasks.
    best_ts_res, best_losses = None, 1e6
    best_eval_epoch = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        model.train()
        with tqdm(tr_loader) as td:
            for batch_data in td:
                text_bert = batch_data['text_bert'].float().to(model.device)
                audio = batch_data['audio'].float().to(model.device)
                vision = batch_data['vision'].float().to(model.device)
                
                with torch.no_grad():
                    text = model.bert(
                        input_ids=text_bert[:,0,:].long(),
                        attention_mask=text_bert[:,1,:].long(), 
                        token_type_ids=text_bert[:,2,:].long())[0]
                
                if opt.dataset != 'mosi':
                    text, audio, vision = model.align_subnet(text,audio,vision,batch_data['text_len'], batch_data['audio_len'], batch_data['vision_len'])
                
                model.finetune_inner_train(text, audio, vision, batch_data['label'])
                
                preds, labels,feature_losses = model.finetune_inner_eval(nc_data)
                preds, labels = torch.cat(preds, dim=0), torch.cat(labels, dim=0)
                losses = opt.criterion_loss(preds, labels)
                losses += feature_losses
                model.optimize_offset_parameters(losses)
                
                preds, labels = model.finetune_outer_train(text, audio, vision, batch_data['label'])
                preds, labels = torch.cat(preds, dim=0), torch.cat(labels, dim=0)
                outer_losses = opt.criterion_loss(preds, labels)
                model.optimize_parameters(outer_losses)

        # Evalutation Model Performances on Val Set.
        model.eval()
        vl_losses = 0
        with torch.no_grad():
            with tqdm(vl_loader) as td:
                for batch_data in td:
                    text_bert = batch_data['text_bert'].float().to(model.device)
                    audio = batch_data['audio'].float().to(model.device)
                    vision = batch_data['vision'].float().to(model.device)
                    
                    text = model.bert(
                        input_ids=text_bert[:,0,:].long(),
                        attention_mask=text_bert[:,1,:].long(), 
                        token_type_ids=text_bert[:,2,:].long())[0]
                    
                    if opt.dataset != 'mosi':
                        text, audio, vision = model.align_subnet(text,audio,vision,batch_data['text_len'], batch_data['audio_len'], batch_data['vision_len'])
                    
                    text = model.netL(text)
                    audio = model.netA(audio)
                    vision = model.netV(vision)

                    text = model.netOffsetL(text)
                    audio = model.netOffsetA(audio)
                    vision = model.netOffsetV(vision)

                    fusion = model.netF(text, audio, vision)

                    logit, _ = model.netC(fusion)
                    loss = model.criterion_L1(logit, batch_data['label'].float().to(model.device))
                    vl_losses += loss.item()
        
        vl_losses = vl_losses / len(vl_loader)

        logger.info(f'End of training epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {time.time() - epoch_start_time} sec \t Train Loss: {losses}')
        logger.info(f'Val Loss: {vl_losses}')
    
        isBetter = vl_losses <= (best_losses - 1e-6)
        if isBetter:
            # Evalutation Model Performances on Test Set.
            preds, labels = [], []
            with torch.no_grad():
                with tqdm(ts_loader) as td:
                    for batch_data in td:
                        text_bert = batch_data['text_bert'].float().to(model.device)
                        audio = batch_data['audio'].float().to(model.device)
                        vision = batch_data['vision'].float().to(model.device)
                        text = model.bert(
                            input_ids=text_bert[:,0,:].long(),
                            attention_mask=text_bert[:,1,:].long(), 
                            token_type_ids=text_bert[:,2,:].long())[0]
                        
                        if opt.dataset != 'mosi':
                            text, audio, vision = model.align_subnet(text,audio,vision,batch_data['text_len'], batch_data['audio_len'], batch_data['vision_len'])
                        
                        text = model.netL(text)
                        audio = model.netA(audio)
                        vision = model.netV(vision)
                        
                        text = model.netOffsetL(text)
                        audio = model.netOffsetA(audio)
                        vision = model.netOffsetV(vision)

                        fusion = model.netF(text, audio, vision)
                        logit, _ = model.netC(fusion)
                        labels.append(batch_data['label'].cpu().detach().numpy())
                        preds.append(logit.cpu().detach().numpy())

            # model.save_networks(e,f"{task_info['type']}{round(task_info['rate'],1)}")

            labels = np.concatenate(labels, axis=0)
            preds = np.concatenate(preds, axis=0)
            ts_results = metrics(preds, labels)
            logger.info(f'Test Results: {ts_results}')
            best_eval_epoch, best_losses, best_ts_res = epoch, vl_losses, ts_results

    logger.info(f'Best Eval Epoch: {best_eval_epoch} \t Best Eval Loss: {best_losses}')
    with open(result_file, 'a') as f:
        f.write(str(best_ts_res) + '\n')
        f.flush()

def train_meta(opt):

    opt.criterion_loss = torch.nn.L1Loss()
    # setup the Task
    task_generator = create_task(opt)  # create a task given opt.task and other options

    # prepare the environment. seed, logger, device.
    setup_seed(opt.seed)
    logger_path = os.path.join(opt.log_dir, opt.name)
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.seed))
    suffix = '_'.join([f'{opt.dataset}_{opt.model}_{opt.train_size}', opt.task])
    logger = get_logger(logger_path, suffix)            # get logger
    result_file = os.path.join(opt.result_dir, 'best_eval.txt')

    # setup the Metrics
    metrics = MetricsTop().getMetics(opt.dataset)

    # setup the base model.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    opt.task= 'eval'
    eval_task_generator = create_task(opt)  # create a task given opt.task and other options

    # Be used for finetune phase 
    clean_data = creatNoiseClean(eval_task_generator,'clean',opt.q_sz)

    for e in range(1, 1 + opt.outer_step): # each outer epoch contains n training tasks.
        train_outer(e, model, task_generator, metrics, opt, logger)
        task_generator.next(mode='meta_train')
        if e % opt.test_interval == 0:
            logger.info('*' * 40 + 'Starting Test Pharse' + '*' * 40 + '\n')
            with open(result_file, 'a') as f:
                f.write('*' * 40 + f'Test Pharse (Epoch {e})' + '*' * 40 + '\n')
            
            for task in eval_task_generator.task_cache:
                
                logger.info(task.meta_info)
                with open(result_file, 'a') as f:
                    f.write(str(task.meta_info) + '\n')
                
                # deep copy the model using load_state_dict.
                opt.gpu_ids = opt.gpu_ids_reserve
                model_t = create_model(opt)
                model_t.setup(opt)

                for m in model_t.model_names:
                    getattr(model_t, m).load_state_dict(getattr(model, m).state_dict())
                
                x_train, y_train, x_val, y_val, x_test, y_test= task.train_x, task.train_y, task.val_x, task.val_y, task.test_x, task.test_y
                                
                noise_data = creatNoiseClean(x_train,'noise',opt.q_sz)
                nc_data = NoiseCleanDataset(clean_data, noise_data)
                tr_data, vl_tata, ts_data = BaseDataset(x_train, y_train),  BaseDataset(x_val, y_val), BaseDataset(x_test, y_test)
                tr_loader = DataLoader(tr_data, batch_size=opt.task_inner_size, shuffle=True)
                vl_loader = DataLoader(vl_tata, batch_size=opt.batch_size, shuffle=False)
                ts_loader = DataLoader(ts_data, batch_size=opt.batch_size, shuffle=False)
                nc_data = DataLoader(nc_data, batch_size=opt.batch_size, shuffle=False)

                finetune(model_t, metrics, tr_loader, vl_loader, ts_loader, nc_data ,opt, logger, result_file)
                

if __name__ == "__main__":
    for seed in [1111,1112,1113]:
        opt = Options().parse(seed=seed)
        train_meta(opt)
