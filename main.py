import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
from tensorboardX import SummaryWriter
import os
import numpy as np
from math import *
import time
from util.visualizer import Visualizer
from PIL import Image

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)

# lzc: python3 main.py -p train -c config/train.json
if __name__ == "__main__":
    print('(lzc-main.py) torch.cuda.is_available()--main',torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    # 获取参数
    parser.add_argument('-c', '--config', type=str, default='config/test.json', help='JSON file for configuration')
    # 设定json文件路径
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or test(inference)', default='train')
    # 选择训练模式
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    # 解析参数
    opt = Logger.parse(args)
    # opt： OrderedDict([('name', 'DARM_train'), ('phase', 'train'), ('gpu_ids', [1]), ...
    # Convert to NoneDict, which return None for missing key. # 转换为NoneDict，如果缺少密钥，则返回None。
    opt = Logger.dict_to_nonedict(opt)
    visualizer = Visualizer(opt) # 感觉这个Visualizer的作用可能是用于可视化显示
    
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    if False: #是否输出所有配置参数
        logger.info(Logger.dict2str(opt))
    else: print('[main.py]--控制台不输出配置参数')
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    batchSize = opt['datasets']['train']['batch_size'] # batch_size=1
    print("batchSize:",batchSize)
    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = Data.create_dataset_xcad(dataset_opt, phase) #train_set.data_len=2
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
            training_iters = int(ceil(train_set.data_len / float(batchSize))) # training_iters=2
            val_set = Data.create_dataset_xcad(dataset_opt, 'val') # val_set.data_len=2
            val_loader = Data.create_dataloader(val_set, dataset_opt, 'val')
            valid_iters = int(ceil(val_set.data_len / float(batchSize))) # valid_iters=2
        elif phase == 'test':
            val_set = Data.create_dataset_xcad(dataset_opt, 'test')
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
            valid_iters = int(ceil(val_set.data_len / float(batchSize)))
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    val_dice = 0
    if opt['phase'] == 'train':
        while current_epoch < n_epoch:
            current_epoch += 1
            for istep, train_data in enumerate(train_loader):
                # print('(lzc) istep, train_data:', istep, train_data)
                '''
                istep:1, 
                train_data:  {
                    'A': tensor([[[[-0.6627, ...), 
                    'B': tensor([[[[-0.6941, ...), 
                    'F': tensor([[[[-1.0000, ...), 
                    'P': ['./data/Dataset_XCAD/train\\trainC\\003_PPA_-29_PSA_29_2.png'], 'Index': tensor([1])}
                '''
                iter_start_time = time.time()
                current_step += 1
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if (istep+1) % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    t = (time.time() - iter_start_time) / batchSize
                    visualizer.print_current_errors(current_epoch, istep+1, training_iters, logs, t, 'Train')
                    visualizer.plot_current_errors(current_epoch, (istep+1) / float(training_iters), logs)
                    visuals = diffusion.get_current_visuals()
                    visualizer.display_current_results(visuals, current_epoch, True)

                # validation
                if (current_step+1) % opt['train']['val_freq'] == 0:
                    diffusion.test(continous=False)
                    visuals = diffusion.get_current_visuals(isTrain=False)
                    visualizer.display_current_results(visuals, current_epoch, True)

            if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)

            dice_per_case_score = 0
            # print("val_loader:",val_loader)
            # print('enumerate(val_loader)',enumerate(val_loader))
            for idata, val_data in enumerate(val_loader):
                diffusion.feed_data(val_data)
                diffusion.test_segment()
                visuals = diffusion.get_current_segment()
                predseg = visuals['test_V'].squeeze().numpy()
                predseg = (predseg + 1) / 2.
                predseg = (predseg > 0.5).astype(bool)

                label = val_data['F'].cpu().squeeze().numpy()
                label = (label + 1) / 2.
                label = (label > 0.5).astype(bool)
                dice = (visualizer.calculate_score(label, predseg, "dice"))
                dice_per_case_score += dice
            dice_case = (dice_per_case_score) / valid_iters
            if dice_case >= val_dice:
                val_dice = dice_case
                diffusion.save_network(current_epoch, current_step, seg_save=True, dice=round(val_dice, 4))

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        file = open(result_path + 'test_score.txt', 'w')
        dice_per_case_score = []
        prec_per_case_score = []
        jacc_per_case_score = []
        for idata,  val_data in enumerate(val_loader):
            dataInfo = val_data['P'][0].split('\\')[-1][:-4]
            time1 = time.time()
            diffusion.feed_data(val_data)
            diffusion.test_segment()
            time2 = time.time()

            visuals = diffusion.get_current_segment()
            predseg = visuals['test_V'].squeeze().numpy()
            predseg = (predseg + 1) / 2.
            predseg = (predseg > 0.5).astype(bool)

            label = val_data['F'].cpu().squeeze().numpy()
            label = (label + 1) / 2.
            label = (label > 0.5).astype(bool)

            data = val_data['A'].cpu().squeeze().numpy()
            data = (data+1)/2.
            savePath = os.path.join(result_path, '%d_data.png' % idata)
            save_image(data * 255, savePath)
            savePath = os.path.join(result_path, '%d_pred.png' % (idata))
            save_image(predseg * 255, savePath)
            savePath = os.path.join(result_path, '%d_label.png' % (idata))
            save_image(label * 255, savePath)

            dice = (visualizer.calculate_score(label, predseg, "dice"))
            prec = (visualizer.calculate_score(label, predseg, "prec"))
            jacc = (visualizer.calculate_score(label, predseg, "jacc"))
            file.write('%04d: process image... %03s | Dice=%f | Prec=%f | Jacc=%f \n' % (idata, dataInfo, dice, prec, jacc))

            print('%04d: process image... %s' % (idata, dataInfo))
            dice_per_case_score.append(dice)
            prec_per_case_score.append(prec)
            jacc_per_case_score.append(jacc)

        print("score_dice_per_case: %3f +- %3f" % (np.mean(dice_per_case_score), np.std(dice_per_case_score)))
        print("score_prec_per_case : %3f +- %3f" % (np.mean(prec_per_case_score), np.std(prec_per_case_score)))
        print("score_jacc_per_case : %3f +- %3f" % (np.mean(jacc_per_case_score), np.std(jacc_per_case_score)))
        file.write('score_case | Dice=%f | Precision=%f | Jaccard=%f \n'
                   % (np.mean(dice_per_case_score), np.mean(prec_per_case_score), np.mean(jacc_per_case_score)))
        file.write('S_std_case | Dice=%f | Precision=%f | Jaccard=%f \n'
                   % (np.std(dice_per_case_score), np.std(prec_per_case_score), np.std(jacc_per_case_score)))
        file.close()