import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from loss import clip_loss


def do_train(cfg, model, center_criterion, train_loader, val_loader, optimizer, optimizer_center, scheduler, loss_fn, num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")
    _LOCAL_PROCESS_GROUP = None

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # 损失计算器
    loss_meter = AverageMeter()
    # 准确率计算器
    acc_meter = AverageMeter()

    # 创建评估器实例
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    # 创建梯度缩放器
    # scaler = amp.GradScaler()
    scaler = torch.amp.GradScaler('cuda')

    # train
    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        model.module.train_with_single()
    else:
        model.train_with_single()
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            # with amp.autocast(enabled=True):
            with torch.amp.autocast('cuda', enabled=True):
                # 模型前向传播，获取分数、特征以及预测掩码 (OSSDet 改进：解包 pred_masks)
                score, feat, pred_masks = model(img, target, cam_label=target_cam)
                # 计算损失 (OSSDet 改进：传入 pred_masks 计算 Act Loss)
                loss = loss_fn(score, feat, target, target_cam, pred_masks=pred_masks)

            # 缩放损失并反向传播
            scaler.scale(loss).backward()

            # 执行主优化器步骤
            scaler.step(optimizer)
            scaler.update()

            # 检查是否使用中心损失
            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                # 调整中心损失参数梯度
                for param in center_criterion.parameters():
                    param.grad.data *= 1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
                scaler.step(optimizer_center)
                scaler.update()
            
            # 检查score是否为列表（多分支输出）
            if isinstance(score, list):
                # 多分支时使用第一个分支计算准确率
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                # 单分支时直接计算准确率
                acc = (score.max(1)[1] == target).float().mean()

            # 更新损失统计
            loss_meter.update(loss.item(), img.shape[0])
            # 更新准确率统计
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                    logger.info(
                        "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                            epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]
                        )
                    )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info(
                "Epoch {} done. Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}, Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0], time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        # 检查点保存逻辑
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)))

        # 检查评估周期
        if epoch % eval_period == 0:
            # 分布式和单卡评估逻辑，计算mAP和CMC指标
            if cfg.MODEL.DIST_TRAIN:
                # 在分布式训练中，只在rank 0进程执行评估
                if dist.get_rank() == 0:
                    # 将模型切换到评估模式，关闭dropout、batch normalization的train模式
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            
                            # 提取图像特征
                            feat = model(img, cam_label=camids)
                            evaluator.update((feat, vid, camid))
                    # 计算累积的评估指标
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    # 输出Rank-1, Rank-3, Rank-5的准确率
                    # 遍历三个重要的rank位置
                    for r in [1, 3, 5]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                # 单卡训练
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        
                        feat = model(img, cam_label=camids)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 3, 5]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for inference".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            
            feat = model(img, cam_label=camids)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 3, 5]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]