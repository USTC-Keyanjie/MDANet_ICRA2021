"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""

########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"

########################################
from tqdm import tqdm

from trainers.trainer import Trainer  # from CVLPyDL repo
import torch

import matplotlib.pyplot as plt
import os.path
from utils.AverageMeter import AverageMeter
from utils.saveTensorToImage import *
from utils.ErrorMetrics import *
import time
from modules.losses import *
import cv2
import logging

err_metrics = ['MAE()', 'RMSE()', 'iMAE()', 'iRMSE()']


class KittiDepthTrainer(Trainer):
    def __init__(self, net, params, optimizer, objective1, objective2, lr_scheduler, dataloaders, dataset_sizes,
                 workspace_dir, sets=['train', 'val'], use_load_checkpoint=None, exp_dir=None):

        # Call the constructor of the parent class (trainer)
        super(KittiDepthTrainer, self).__init__(net, optimizer, lr_scheduler, objective1, objective2,
                                                use_gpu=params['use_gpu'],
                                                workspace_dir=workspace_dir)

        self.lr_scheduler = lr_scheduler
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.use_load_checkpoint = use_load_checkpoint

        self.params = params
        self.save_chkpt_each = params['save_chkpt_each']
        self.sets = sets
        self.save_images = params['save_out_imgs']
        self.load_rgb = params['load_rgb'] if 'load_rgb' in params else False

        self.model_name = params['model']
        self.use_binLoss = True if params['loss1'] == 'BINLoss' else False

        self.exp_dir = exp_dir

        for s in self.sets:
            self.stats[s + '_loss'] = []

    ####### Training Function #######

    def train(self, max_epochs):
        logger = open(os.path.join(self.exp_dir, 'train.log'), mode='w')

        logger.write(f'[Train] - {self.model_name}\n\n')

        logger.write('[Experiment Parameters]\n')
        logger.write('Number of model parameters: {}\n'.format(sum([p.data.nelement() for p in self.net.parameters()])))

        for k, v in self.params.items():
            logger.write('{0:<22s} : {1:}\n'.format(k, v))

        # Load last save checkpoint
        if self.use_load_checkpoint != None:
            if isinstance(self.use_load_checkpoint, int):
                if self.use_load_checkpoint > 0:
                    logger.write('=> Loading checkpoint {} ...\n'.format(self.use_load_checkpoint))
                    if self.load_checkpoint(self.use_load_checkpoint):
                        logger.write('Checkpoint was loaded successfully!\n')
                    else:
                        logger.write('Evaluating using initial parameters\n')
                elif self.use_load_checkpoint == -1:
                    logger.write('=> Loading last checkpoint ...\n')
                    if self.load_checkpoint():
                        logger.write('Checkpoint was loaded successfully!\n')
                    else:
                        logger.write('Evaluating using initial parameters\n')
            elif isinstance(self.use_load_checkpoint, str):
                logger.write('loading checkpoint from : \n' + self.use_load_checkpoint)
                if self.load_checkpoint(self.use_load_checkpoint):
                    self.epoch = int(self.use_load_checkpoint[-10:-8]) + 1
                    logger.write('Checkpoint was loaded successfully!\n')
                else:
                    logger.write('Evaluating using initial parameters\n')

        logger.write('\n')
        logger.write('=' * 15 * 4)
        logger.write('\n')
        logger.write(f'{"Epoch":^15}{"lr":^15}{"Loss":^15}{"Time(h)":^15}\n')

        for epoch in range(self.epoch, max_epochs + 1):  # range function returns max_epochs-1

            logger.write('-' * 15 * 4)
            logger.write('\n')
            logger.flush()

            start_epoch_time = time.time()

            self.epoch = epoch

            # Decay Learning Rate
            self.lr_scheduler.step(epoch)  # LR decay
            logger.write(f'{epoch:^15}{self.optimizer.param_groups[0]["lr"]:^15.4g}')

            # Train the epoch
            loss_meter = self.train_epoch(logger)

            # Add the average loss for this epoch to stats
            for s in self.sets: self.stats[s + '_loss'].append(loss_meter[s].avg)

            # Save checkpoint
            if self.use_save_checkpoint and self.epoch % self.save_chkpt_each == 0:
                self.save_checkpoint()

            end_epoch_time = time.time()
            epoch_duration = end_epoch_time - start_epoch_time
            logger.write(f'{epoch_duration / 3600:^15.2f}\n')

        logger.write('=' * 15 * 4)
        logger.write('\n')

        # Save the final model
        torch.save(self.net, self.workspace_dir + '/final_model.pth')
        logger.close()
        return self.net

    def train_epoch(self, logger):

        loss_meter = {}
        for s in self.sets: loss_meter[s] = AverageMeter()

        for s in self.sets:
            # Iterate over data.
            for data in tqdm(self.dataloaders[s]):
                inputs_d, C, item_idxs, inputs_rgb, gt_depth = data

                inputs_d = inputs_d.cuda()
                gt_depth = gt_depth.cuda()
                inputs_rgb = inputs_rgb.cuda()

                outputs = self.net(inputs_d, inputs_rgb)

                # Calculate loss for valid pixel in the ground truth
                MSELoss11 = self.objective1(outputs[0], gt_depth)
                MSELoss12 = self.objective1(outputs[1], gt_depth)
                MSELoss14 = self.objective1(outputs[2], gt_depth)

                MAELoss11 = self.objective2(outputs[0], gt_depth)
                MAELoss12 = self.objective2(outputs[1], gt_depth)
                MAELoss14 = self.objective2(outputs[2], gt_depth)

                if self.epoch < 6:
                    loss = MSELoss14 + MSELoss12 + MSELoss11 + \
                           MAELoss14 + MAELoss12 + MAELoss11
                elif self.epoch < 11:
                    loss = 0.1 * MSELoss14 + 0.1 * MSELoss12 + MSELoss11 + \
                           0.1 * MAELoss14 + 0.1 * MAELoss12 + MAELoss11
                elif self.epoch < 41:
                    loss = MSELoss11 + MAELoss11
                else:
                    loss = MSELoss11

                # backward + optimize only if in training phase
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # statistics
                loss_meter[s].update(MSELoss11.item(), inputs_d.size(0))

            logger.write(f'{loss_meter[s].avg:^15.8f}')
            torch.cuda.empty_cache()

        return loss_meter

    ####### Evaluation Function #######
    def evaluate(self, save_output):
        logger = open(os.path.join(os.path.split(self.use_load_checkpoint)[0], '..', 'eval.log'), mode='w')

        # Load last save checkpoint

        epoch = 0
        if self.use_load_checkpoint != None:
            if self.use_load_checkpoint[-12:-8].isdigit():
                epoch = int(self.use_load_checkpoint[-12:-8])
            if isinstance(self.use_load_checkpoint, int):
                if self.use_load_checkpoint > 0:
                    self.load_checkpoint(self.use_load_checkpoint)
                elif self.use_load_checkpoint == -1:
                    self.load_checkpoint()
            elif isinstance(self.use_load_checkpoint, str):
                self.load_checkpoint(self.use_load_checkpoint)

        self.net.train(False)

        # AverageMeters for Loss
        loss_meter = {}
        for s in self.sets: loss_meter[s] = AverageMeter()

        # AverageMeters for error metrics
        err = {}
        for m in err_metrics: err[m] = AverageMeter()

        # AverageMeters for time
        times = AverageMeter()

        logger.write(f'[Eval] - {self.model_name}\n')
        logger.write("=" * 15 * (4 + len(err_metrics)))
        logger.write('\n')
        logger.write(f'{"Epoch":^15}{"Val Loss":^15}')
        for m in err_metrics:
            logger.write(f'{m:^15}')
        logger.write(f'{"Time":^15}{"Time_av":^15}\n')

        with torch.no_grad():
            for s in self.sets:
                # Iterate over data.
                Start_time = time.time()
                logger.write("-" * 15 * (4 + len(err_metrics)))
                logger.write('\n')

                for data in tqdm(self.dataloaders[s]):

                    # inference
                    # ========================
                    start_time = time.time()
                    inputs_d, C, item_idxs, inputs_rgb, labels = data
                    inputs_d = inputs_d.cuda()
                    C = C.cuda()
                    labels = labels.cuda()
                    inputs_rgb = inputs_rgb.cuda()

                    outputs = self.net(inputs_d, inputs_rgb)

                    if len(outputs) > 1:
                        outputs = outputs[0]

                    duration = time.time() - start_time
                    times.update(duration / inputs_d.size(0), inputs_d.size(0))
                    # ========================


                    if s == 'selval' or s == 'val' or s == 'test':

                        # Calculate loss for valid pixel in the ground truth
                        loss = self.objective1(outputs, labels, self.epoch)

                        # statistics
                        loss_meter[s].update(loss.item(), inputs_d.size(0))

                        # Convert data to depth in meters before error metrics
                        outputs[outputs == 0] = -1
                        if not self.load_rgb:
                            outputs[outputs == outputs[0, 0, 0, 0]] = -1
                        labels[labels == 0] = -1
                        if self.params['invert_depth']:
                            outputs = 1 / outputs
                            labels = 1 / labels
                        outputs[outputs == -1] = 0
                        labels[labels == -1] = 0
                        outputs *= self.params['data_normalize_factor'] / 256
                        labels *= self.params['data_normalize_factor'] / 256

                        # Calculate error metrics
                        for m in err_metrics:
                            if m.find('Delta') >= 0:
                                fn = globals()['Deltas']()
                                error = fn(outputs, labels)
                                err['Delta1'].update(error[0], inputs_d.size(0))
                                err['Delta2'].update(error[1], inputs_d.size(0))
                                err['Delta3'].update(error[2], inputs_d.size(0))
                                break
                            else:
                                fn = eval(m)  # globals()[m]()
                                error = fn(outputs, labels)
                                err[m].update(error.item(), inputs_d.size(0))

                    # Save output images (optional)
                    if save_output:
                        outputs = outputs.data
                        outputs *= 256
                        saveTensorToImage(
                            outputs,
                            item_idxs,
                            os.path.join(
                                os.path.split(self.use_load_checkpoint)[0],
                                '..',
                                s + '_output_epoch_' + str(epoch)))

                average_time = (time.time() - Start_time) / len(self.dataloaders[s].dataset)

                logger.write(f'{epoch:^15}')
                logger.write(f'{loss_meter[s].avg:^15.8f}')
                for m in err_metrics:
                    logger.write(f'{err[m].avg:^15.8f}')
                logger.write(f'{times.avg:^15.4f}')
                logger.write(f'{average_time:^15.4f}\n')

                logger.write("=" * 15 * (4 + len(err_metrics)))
                logger.write('\n')

                torch.cuda.empty_cache()

        logger.close()

    ####### Evaluation Function #######
    def evaluate_folder(self):
        logger = open(os.path.join(self.use_load_checkpoint, '..', 'eval_folder.log'), mode='w')

        min_err = {}
        min_epoch = 0
        for m in err_metrics: min_err[m] = float('Inf')

        # Load last save checkpoint
        if self.use_load_checkpoint is not None and isinstance(self.use_load_checkpoint, str):

            logger.write(f'[Eval] - {self.model_name}\n\n')
            logger.write("=" * 15 * (4 + len(err_metrics)))
            logger.write('\n')
            logger.write(f'{"Epoch":^15}{"Val Loss":^15}')
            for m in err_metrics:
                logger.write(f'{m:^15}')
            logger.write(f'{"Time":^15}{"Time_av":^15}\n')
            logger.flush()

            for ckpt in sorted(os.listdir(self.use_load_checkpoint))[-20:]:
                self.load_checkpoint(self.use_load_checkpoint + '/' + ckpt)

                self.net.train(False)

                # AverageMeters for Loss
                loss_meter = {}
                for s in self.sets: loss_meter[s] = AverageMeter()

                # AverageMeters for error metrics
                err = {}
                for m in err_metrics: err[m] = AverageMeter()

                # AverageMeters for time
                times = AverageMeter()

                # device = torch.device("cuda:" + str(self.config['gpu_id']) if torch.cuda.is_available() else "cpu")

                with torch.no_grad():

                    for s in self.sets:
                        epoch = int(ckpt[15:19])

                        # Iterate over data.
                        Start_time = time.time()
                        logger.write("-" * 15 * (4 + len(err_metrics)))
                        logger.write('\n')
                        logger.flush()

                        for data in tqdm(self.dataloaders[s]):
                            start_time = time.time()

                            inputs_d, C, item_idxs, inputs_rgb, gt_depth = data

                            inputs_d = inputs_d.cuda()
                            gt_depth = gt_depth.cuda()
                            inputs_rgb = inputs_rgb.cuda()

                            outputs = self.net(inputs_d, inputs_rgb)

                            if self.use_binLoss:
                                outputs = outputs[-1]
                            elif len(outputs) > 1:
                                outputs = outputs[0]

                            duration = time.time() - start_time
                            times.update(duration / inputs_d.size(0), inputs_d.size(0))

                            if s == 'selval' or s == 'val' or s == 'test':

                                # Calculate loss for valid pixel in the ground truth
                                loss = self.objective1(outputs, gt_depth, self.epoch)

                                # statistics
                                loss_meter[s].update(loss.item(), inputs_d.size(0))

                                # Convert data to depth in meters before error metrics
                                outputs[outputs == 0] = -1
                                if not self.load_rgb:
                                    outputs[outputs == outputs[0, 0, 0, 0]] = -1
                                gt_depth[gt_depth == 0] = -1
                                if self.params['invert_depth']:
                                    outputs = 1 / outputs
                                    gt_depth = 1 / gt_depth
                                outputs[outputs == -1] = 0
                                gt_depth[gt_depth == -1] = 0
                                outputs *= self.params['data_normalize_factor'] / 256
                                gt_depth *= self.params['data_normalize_factor'] / 256

                                # Calculate error metrics
                                for m in err_metrics:
                                    if m.find('Delta') >= 0:
                                        fn = globals()['Deltas']()
                                        error = fn(outputs, gt_depth)
                                        err['Delta1'].update(error[0], inputs_d.size(0))
                                        err['Delta2'].update(error[1], inputs_d.size(0))
                                        err['Delta3'].update(error[2], inputs_d.size(0))
                                        break
                                    else:
                                        fn = eval(m)  # globals()[m]()
                                        error = fn(outputs, gt_depth)
                                        err[m].update(error.item(), inputs_d.size(0))

                            # Save output images (optional)

                            if s in ['test']:
                                outputs = outputs.data

                                outputs *= 256

                                saveTensorToImage(outputs, item_idxs, os.path.join(self.workspace_dir,
                                                                                   s + '_output_' + 'epoch_' + str(
                                                                                       self.epoch)))

                        average_time = (time.time() - Start_time) / len(self.dataloaders[s].dataset)

                        logger.write(f'{epoch:^15}')
                        logger.write(f'{loss_meter[s].avg:^15.8f}')
                        for m in err_metrics:
                            logger.write(f'{err[m].avg:^15.8f}')

                        if err['RMSE()'].avg < min_err['RMSE()']:
                            min_epoch = epoch
                            for m in err_metrics:
                                min_err[m] = err[m].avg
                        logger.write(f'{times.avg:^15.4f}')
                        logger.write(f'{average_time:^15.4f}\n')

                        torch.cuda.empty_cache()
            logger.write("=" * 15 * (4 + len(err_metrics)))
            logger.write('\n')

        logger.write('\nMin Error show:\n')
        logger.write(f'Min Epoch: {min_epoch}\n')
        for m in err_metrics:
            logger.write('Min Error: [{}]: {:.8f}\n'.format(m, min_err[m]))
        logger.close()
