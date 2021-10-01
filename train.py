import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


#def interleave(x, size):
#    s = list(x.shape)
#    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


#def de_interleave(x, size):
#    s = list(x.shape)
#    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
 

def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='resnet', type=str,
                        help='dataset name')
    parser.add_argument('--total-steps', default=156200, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=781, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled ssl loss')
    parser.add_argument('--lambda-c', default=0.08, type=float,
                        help='coefficient of labeled cl loss')
    parser.add_argument('--lambda-sim', default=0.1, type=float,
                        help='coefficient of unlabeled cl loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--stop_active', type=int, default=250,
                        help="the number of labeled data after active learnging")
    parser.add_argument('--num_sample', type=int, default=16,
                        help="the number of batches per sampling")
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument('--epoch_warmup', type=int, default=10,
                        help="the epoch for cl to warmup")                    

    args = parser.parse_args()
    global best_acc

    def create_model(args):
        if args.arch == 'resnet':
            import models.resnet as models
            model = models.build_resnet(num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100


    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset, labeled_idxs_org, unlabeled_sim_dataset, labeled_sim_dataset\
        = DATASET_GETTERS[args.dataset](
        args, './data')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    unlabeled_sim_trainloader = DataLoader(
        unlabeled_sim_dataset,
        sampler=train_sampler(unlabeled_sim_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    labeled_sim_trainloader = DataLoader(
        labeled_sim_dataset,
        sampler=train_sampler(labeled_sim_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler,
          labeled_idxs_org, unlabeled_sim_trainloader, labeled_sim_trainloader)


def uncertainty_margin_al(logit_matrix, input_idx):
    '''
    Choose the sample from the current batch which has closest distance between the most possible 
    calss and second-most possible class, and get the corresponding label from the labels.

    Parameters: (N is number of total number of samples, d is the number of classes)
    logit_matrix: (N, d) tensor.      probabilities of d-class for each sample
    input_idx:    (N,  ) tensor.      global index of all samples in the batch

    Outputs:
    select_idx:   int.                index of the most uncertain sample.
    '''        
    logit_matrix_copy = logit_matrix.clone().detach()
    max_prob, max_idx = torch.max(logit_matrix_copy, axis=1)
    logit_matrix_copy[range(logit_matrix_copy.shape[0]), max_idx] = 0
    second_max_prob, _ = torch.max(logit_matrix_copy, axis=1)
    diff_prob = max_prob - second_max_prob
    min_diff, idx = torch.min(diff_prob, axis=-1)
    select_idx = input_idx[idx].item()
    return select_idx, min_diff


def info_nce_loss(args, features):

    labels = torch.cat([torch.arange(args.batch_size*args.mu) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(args.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)

    logits = logits / args.temperature
    return logits, labels



def classwise_contrastive_loss(args, features, labels):
    '''
    classwise_contrastive_loss:
            This function defines the classwise contrastive loss
    Inputs:
            features: It is the trained representations of the labeled batch
            labels: It is the labels for the labeled batch
    Outputs:
            loss: classwise contrastive loss
    '''
    # set up two local parameters (should be set to hyperparameters later)
    eps = 1e-9
    my_temp = 0.07
    #print(my_temp)
    # Obtain the similarity matrix
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.exp(torch.matmul(features, features.T)/my_temp)
    # Mask the entries on the main diagonal
    mask1 = torch.eye(similarity_matrix.shape[0]).to(args.device)
    similarity_matrix = similarity_matrix * (1-mask1)
    # Obtain the total sum: denomenator
    similarity_total_sum = torch.sum(similarity_matrix, dim=1)
    # Obtain the label for the augmented labelled set
    labels_extend = labels.repeat(2)
    # Mask out samples in *different* classes and obtain the numerator
    mask2 = (labels_extend == labels_extend.view(labels_extend.shape[0], -1)).to(args.device)
    similarity_class_sum = torch.sum((similarity_matrix * mask2), dim=1)
    # Obtain the final loss
    loss = torch.sum((-1/(similarity_matrix.shape[0]*torch.sum(mask2, dim=1))) * (torch.log(similarity_class_sum/(similarity_total_sum) + eps)))
    return loss


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler,
          labeled_idxs_org, unlabeled_sim_trainloader, labeled_sim_trainloader):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        unlabeled_sim_epoch = 0
        labeled_sim_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
        unlabeled_sim_trainloader.sampler.set_epoch(unlabeled_sim_epoch)
        labeled_sim_trainloader.sampler.set_epoch(labeled_sim_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    unlabeled_sim_iter = iter(unlabeled_sim_trainloader)
    labeled_sim_iter = iter(labeled_sim_trainloader)
    cnt = 0
    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_sim = AverageMeter()
        losses_c = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])

        diff_set = []
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x,_ = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x,_ = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), _, global_index = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _, global_index = unlabeled_iter.next()

            try:
                (inputs_u_1, inputs_u_2), _, _ = unlabeled_sim_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_sim_epoch += 1
                    unlabeled_sim_trainloader.sampler.set_epoch(unlabeled_sim_epoch)
                unlabeled_sim_iter = iter(unlabeled_sim_trainloader)
                (inputs_u_1, inputs_u_2), _, _ = unlabeled_sim_iter.next()

            try:
                (inputs_1, inputs_2), targets_c,_ = labeled_sim_iter.next()
            except:
                if args.world_size > 1:
                    labeled_sim_epoch += 1
                    labeled_sim_trainloader.sampler.set_epoch(labeled_sim_epoch)
                labeled_sim_iter = iter(labeled_sim_trainloader)
                (inputs_1, inputs_2), targets_c,_ = labeled_sim_iter.next()


            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            #inputs = interleave(
            #    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            
            inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(args.device)
            #inputs_sim = interleave(
            #    torch.cat((inputs_u_1, inputs_u_2)), 2*args.mu).to(args.device)
            inputs_sim = torch.cat((inputs_1, inputs_2, inputs_u_1, inputs_u_2)).to(args.device)
            logits, logits_sim = model(inputs, inputs_sim)
            #logits = de_interleave(logits, 2*args.mu+1)
            #logits_sim = de_interleave(logits, 2*args.mu)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            logits_c = logits_sim[:batch_size*2]
            logits_u_sim = logits_sim[batch_size*2:]
            logits_u_sim, labels_u_sim = info_nce_loss(args, logits_u_sim)
            del logits,logits_sim
            
            targets_c = targets_c.to(args.device)
            targets_x = targets_x.to(args.device)

            Lc = classwise_contrastive_loss(args, logits_c, targets_c)
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            criterion_clr = torch.nn.CrossEntropyLoss().to(args.device)
            Lsim = criterion_clr(logits_u_sim, labels_u_sim)

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            if epoch >= args.epoch_warmup+5:
                if labeled_idxs_org.shape[0] <args.stop_active:
                    get_index, get_diff = uncertainty_margin_al(logits_u_w, global_index)
                    diff_set.append((get_diff, get_index))
                    if(cnt == args.num_sample-1):
                        cnt = 0
                        min_diff_v = get_diff
                        min_diff_i = get_index
                        for diff_i, get_ind in diff_set:
                            if min_diff_v > diff_i:
                                min_diff_v = diff_i
                                min_diff_i = get_ind
                        diff_set = []
                        if min_diff_i not in labeled_idxs_org:
                            labeled_idxs_org = np.append(labeled_idxs_org, min_diff_i)

                        labeled_dataset_up, _, _, _,_, labeled_sim_dataset_up = DATASET_GETTERS[args.dataset](args, './data',labeled_idxs_org)
                        train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
                        labeled_trainloader = DataLoader(
                          labeled_dataset_up,
                          sampler=train_sampler(labeled_dataset_up),
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          drop_last=True)
                        labeled_iter = iter(labeled_trainloader)
                        labeled_sim_trainloader = DataLoader(
                          labeled_sim_dataset_up,
                          sampler=train_sampler(labeled_sim_dataset_up),
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          drop_last=True)
                        labeled_sim_iter = iter(labeled_sim_trainloader)

                    else:
                        cnt += 1

            if epoch >= args.epoch_warmup:
                loss = Lx + args.lambda_u * Lu + Lc*args.lambda_u
            else:
                loss = args.lambda_sim*Lsim

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            losses_sim.update(Lsim.item())
            losses_c.update(Lc.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Loss_sim: {loss_sim:.4f}. Loss_c: {loss_c:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    loss_sim=losses_sim.avg,
                    loss_c=losses_c.avg,
                    mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.train_loss_sim', losses_sim.avg, epoch)
            args.writer.add_scalar('train/5.train_loss_c', losses_c.avg, epoch)
            args.writer.add_scalar('train/6.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs,_ = model(inputs, inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
