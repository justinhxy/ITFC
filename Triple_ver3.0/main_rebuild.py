import argparse
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import KFold
from Dataset import MriPetDataset
import torch.utils.data
from model_object import models
from Config import parse_args
from observer import Runtime_Observer
from Net.api import *
import numpy as np


# def prepare_to_train(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_index,
#                      seed, device, fold, data_parallel):
#     global experiment_settings
#     assert torch.cuda.is_available(), "Please ensure codes are executed on cuda."
#     try:
#         experiment_settings = models[model_index]
#     except KeyError:
#         print('model not in model_object!')
#     torch.cuda.empty_cache()
#
#     '''
#     Dataset init, You can refer to the dataset format defined in data/dataset.py to define your private dataset
#
#     '''
#     dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file)
#     # dataset = Liver_dataset(summery_path=experiment_settings['Data'], mode=experiment_settings['Dataset_mode'])
#     torch.manual_seed(seed)
#     # train_ratio = 0.7 train_dataset, test_dataset = random_split(dataset, [int(train_ratio * len(dataset)),
#     # len(dataset) - int(train_ratio * len(dataset))]) trainDataLoader = torch.utils.data.DataLoader(train_dataset,
#     # batch_size=experiment_settings['Batch'], shuffle=True, num_workers=4, drop_last=False) testDataLoader =
#     # torch.utils.data.DataLoader(test_dataset, batch_size=experiment_settings['Batch'], shuffle=False, num_workers=4)
#     '''
#     The seed in Kfold should be same!
#     '''
#     kf = KFold(n_splits=5, shuffle=True, random_state=seed)
#     train_index, test_index = [[t1, t2] for t1, t2 in kf.split(dataset)][fold]
#     train_dataset = torch.utils.data.Subset(dataset, train_index)
#     test_dataset = torch.utils.data.Subset(dataset, test_index)
#
#     trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                                                   num_workers=4, drop_last=True)
#     testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
#                                                  num_workers=4)
#
#     '''
#     Training logs and monitors
#     '''
#     target_dir = Path('./logs/')
#     target_dir.mkdir(exist_ok=True)
#     target_dir = target_dir.joinpath('classification')
#     target_dir.mkdir(exist_ok=True)
#     current_time = str(datetime.now().strftime('%Y-%m-%d_%H-%M'))
#     target_dir = target_dir.joinpath(experiment_settings['Name'] + current_time)
#     target_dir.mkdir(exist_ok=True)
#     checkpoints_dir = target_dir.joinpath('checkpoints')
#     checkpoints_dir.mkdir(exist_ok=True)
#     log_dir = target_dir.joinpath('logs')
#     log_dir.mkdir(exist_ok=True)
#
#     observer = Runtime_Observer(log_dir=log_dir, device=device, name=experiment_settings['Name'], seed=seed)
#     observer.log(f'[DEBUG]Observer init successfully, program start @{current_time}\n')
#     '''
#     Model load
#     '''
#     _model = experiment_settings['Model']
#     print(f"The name of model will run {_model}")
#     model = _model()
#     # 如果有多个GPU可用，使用DataParallel来并行化模型
#     if torch.cuda.device_count() > 1 and data_parallel == 1:
#         observer.log("Using" + str(torch.cuda.device_count()) + "GPUs for training.\n")
#         model = torch.nn.DataParallel(model)
#     observer.log(f'Use model : {str(experiment_settings)}\n')
#     num_params = 0
#     for p in model.parameters():
#         if p.requires_grad:
#             num_params += p.numel()
#     observer.log("\n===============================================\n")
#     observer.log("model parameters: " + str(num_params))
#     observer.log("\n===============================================\n")
#
#     '''
#     Hyper parameter settings
#     '''
#     optimizer = experiment_settings['Optimizer'](model.parameters(), experiment_settings['Lr'])
#     # if experiment_settings['w1'] is not None:
#     if 'w1' in experiment_settings:
#         criterion = experiment_settings['Loss'](w1=experiment_settings['w1'], w2=experiment_settings['w2'])
#     else:
#         criterion = experiment_settings['Loss']()
#
#     print("prepare completed! launch training!\U0001F680")
#
#     # launch
#     _run = experiment_settings['Run']
#     _run(observer, experiment_settings['Epoch'], trainDataLoader, testDataLoader, model, device,
#          optimizer, criterion)

def prepare_to_train(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_index,
                     seed, device, data_parallel):
    global experiment_settings
    assert torch.cuda.is_available(), "Please ensure codes are executed on cuda."
    try:
        experiment_settings = models[model_index]
    except KeyError:
        print('model not in model_object!')
    torch.cuda.empty_cache()

    # 初始化数据集
    dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file)
    torch.manual_seed(seed)

    # K折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    # 存储每个fold的评估指标
    metrics = {
        'accuracy': [],
        'auc': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'Specificity': []
    }

    for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}/{5}')

        # 分割数据集
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        test_dataset = torch.utils.data.Subset(dataset, test_index)

        trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                      num_workers=4, drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                     num_workers=4)

        # 训练日志和监控
        target_dir = Path('./logs/')
        target_dir.mkdir(exist_ok=True)
        target_dir = target_dir.joinpath('classification')
        target_dir.mkdir(exist_ok=True)
        current_time = str(datetime.now().strftime('%Y-%m-%d_%H-%M'))
        target_dir = target_dir.joinpath(experiment_settings['Name'] + f'_{current_time}')
        target_dir.mkdir(exist_ok=True)
        fold_target_dir = target_dir.joinpath(f'fold_{fold + 1}')
        fold_target_dir.mkdir(exist_ok=True)
        checkpoints_dir = fold_target_dir.joinpath('checkpoints')
        checkpoints_dir.mkdir(exist_ok=True)
        log_dir = fold_target_dir.joinpath('logs')
        log_dir.mkdir(exist_ok=True)

        observer = Runtime_Observer(log_dir=log_dir, device=device, name=experiment_settings['Name'], seed=seed)
        observer.log(f'[DEBUG] Observer init successfully, program start @{current_time}\n')

        # 模型加载
        _model = experiment_settings['Model']
        print(f"The name of model will run {_model}")
        model = _model()

        # 使用 DataParallel 进行多GPU训练
        if torch.cuda.device_count() > 1 and data_parallel == 1:
            observer.log("Using " + str(torch.cuda.device_count()) + " GPUs for training.\n")
            model = torch.nn.DataParallel(model)

        observer.log(f'Use model : {str(experiment_settings)}\n')
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        observer.log("\n===============================================\n")
        observer.log("model parameters: " + str(num_params))
        observer.log("\n===============================================\n")

        # 超参数设置
        optimizer = experiment_settings['Optimizer'](model.parameters(), experiment_settings['Lr'])
        if 'w1' in experiment_settings:
            criterion = experiment_settings['Loss'](w1=experiment_settings['w1'], w2=experiment_settings['w2'])
        else:
            criterion = experiment_settings['Loss']()

        print("Prepare completed for fold {}! Launch training!\U0001F680".format(fold + 1))

        # 启动训练
        _run = experiment_settings['Run']
        _run(observer, experiment_settings['Epoch'], trainDataLoader, testDataLoader, model, device,
             optimizer, criterion)

        # 收集评估指标
        metrics['accuracy'].append(observer.best_dicts['acc'])
        metrics['auc'].append(observer.best_dicts['auc'])
        metrics['f1'].append(observer.best_dicts['f1'])
        metrics['precision'].append(observer.best_dicts['p'])
        metrics['recall'].append(observer.best_dicts['recall'])
        metrics['Specificity'].append(observer.best_dicts['spe'])

    for key in metrics:
        mean_value = np.mean(metrics[key])
        std_value = np.std(metrics[key])
        print(f'{key.capitalize()} - Mean: {mean_value:.4f}, Std: {std_value:.4f}')

    print("Cross-validation training completed for all folds.")

if __name__ == "__main__":

    args = parse_args()
    print(args)
    # prepare_to_train(model_index=args.model, mri_dir=args.mri_dir, pet_dir=args.pet_dir, cli_dir=args.cli_dir, csv_file=args.csv_file, batch_size=args.batch_size, seed=args.seed , device=args.device, fold=args.fold, data_parallel=args.data_parallel)
    prepare_to_train(model_index=args.model, mri_dir=args.mri_dir, pet_dir=args.pet_dir, cli_dir=args.cli_dir,csv_file=args.csv_file, batch_size=args.batch_size, seed=args.seed, device=args.device, data_parallel=args.data_parallel)