""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from architect import Architect
from visualize import plot
from torch.autograd import Variable


config = SearchConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)

    net_crit = nn.CrossEntropyLoss().to(device)
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                net_crit, device_ids=config.gpus, n_nodes=2)
    model = model.to(device)

#     print('=====================================')

#     print(list(model.alphas()))
#     print('=====================================')

    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_top1 = 0.
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        # training
        train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, epoch, cur_step)

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        plot(genotype.normal, plot_path + "-normal", caption)
        plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()
        
        
        ###here calculate hessian of loss function
        #loss = model.loss(trn_X, trn_y) # L_val(w`)
        #v_alphas = tuple(model.alphas())
        #v_weights = tuple(model.weights())
        #v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        #dalpha = v_grads[:len(v_alphas)]
        #dw = v_grads[len(v_alphas):]

        #hessian = architect.compute_hessian(dw, trn_X, trn_y)
        #print(len(hessian))
#         print(hessian)
        #print([x.shape for x in hessian])
        if step % 100 == 0 or step == len(train_loader)-1:
            name_add = config.name#'_FAST_CIFAR_100_srch_2'
            
            print('***************************************************************')
            print('LAPLACE CALC...')
            print('ALPHA SHAPE')

            vec_par = architect.net.alphas()
            vec_par = [x.view(-1, 1) for x in vec_par]
            print([x.shape for x in vec_par])
            vec_par = torch.cat(vec_par, dim=0)
            print(f'SIZE OF ALPHAS: {vec_par.shape}')
            a = architect.compute_Hw(trn_X, trn_y)
            #a = np.linalg.det(a.cpu().detach().numpy())
            #a = torch.potrf(a).diag().prod()
            a = a.slogdet()
            
            det_1, det_2 = a[0].cpu().detach().numpy(), a[1].cpu().detach().numpy()
            print(f'EPOCH - {epoch} ITER - {step} | sign: {det_1}, log_value: {det_2}', file=open(f"__out_determinant_{name_add}.txt", "a"))
            
            
            a = a[0]*a[1]
            
            a = a.cpu().detach().numpy()
            loss = model.loss(trn_X, trn_y)
            print(f'LOGLOSS: {loss}')
            print(f'LOGDET {a}')
            #temp1 = torch.mm(vec_par.t(), a)
            #temp1 = torch.log(torch.mm(vec_par.t(), vec_par)).cpu().detach().numpy() + a
            #print(f'TEMP 1 {temp1.shape}')
            #temp2 = torch.mm(temp1, vec_par)
            #print(f'TEMP 2 {temp2.shape}')
            #print(f'RESULT: {temp2.cpu().detach().numpy()[0,0]}')
            #print(f'RESULT: {temp1}')
            print(f'EPOCH - {epoch} ITER - {step} | LOGDET: {vec_par.shape}', file=open(f"__out_d_{name_add}.txt", "a"))
            print(f'EPOCH - {epoch} ITER - {step} | LOGDET: {a}', file=open(f"__out_LOGDET_{name_add}.txt", "a"))
            print(f'EPOCH - {epoch} ITER - {step} | LOGLOSS: {loss}', file=open(f"__out_LOGLOSS_{name_add}.txt", "a"))            
            print("LAPLACE DONE!")

        #print(len(list(model.alphas())))
        #print([x.shape for x in model.alphas()])
 
#         logits = model(trn_X)
#         tr_loss = model.criterion(logits, trn_y)
#         #tr_loss = model.loss(trn_X, trn_y) # L_train(w)
#         grad_L_train_w = torch.autograd.grad(tr_loss, model.alphas(), create_graph = True)
        
#         print(len(grad_L_train_w))
#         print(grad_L_train_w)
        
        
#         print('STEP 1')
        
        
        
        #v_alphas = tuple(model.alphas())
#         #dw = torch.autograd.grad(loss, v_alphas)
#         hessian_vector = torch.autograd.grad(grad_L_train_w, model.alphas(),
#                                              grad_outputs = grad_L_train_w, retain_graph = False)
        
#         print(f'ALPHA SHAPES: {len(list(model.alphas()))}')
#         #print(f'ALPHA SHAPES: {model.alphas().shape}')

        
#         print(hessian_vector)
        ###
        

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    main()
