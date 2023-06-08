import warnings
warnings.filterwarnings("ignore")
import os
import uuid
import numpy as np
import pandas as pd
import datetime
import torch
import pandas as pd
import torch.nn as nn
import sys

import sys
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path,'..'))

from utils.metrics_utils import *
from utils.deepctr_torch_utils import get_input_data,get_data_loader
from utils.deepctr_torch_utils import get_pebg_data_loader
from utils.competition_utils import load_wide_feature,load_save_vectors

from deepctr_torch.inputs import build_input_features

# data
def load_data_loader(args,add_config=True,remove_deep_x=False):
    # wide
    fname= args.fname if "fname" in args else "df_feature_num_label-3.csv"
    data_dir = "data/dataset/"
    sparse_features, dense_features, lbe_dict, mms, train_data, dev_data, test_data = load_wide_feature(data_dir, fname, config=args)
    data_list = [train_data, dev_data, test_data]
    config = {"dense_features": dense_features, 'emb_dim': 1,
            "sparse_features": sparse_features, 'sparse_emb_dim': args.sparse_emb_dim}
    dnn_feature_columns, linear_feature_columns, train_model_input, dev_model_input, test_model_input = get_input_data(
        data_list, lbe_dict, config)
    feature_index = build_input_features(
                linear_feature_columns + dnn_feature_columns)
    if add_config:
        args.dnn_feature_columns = dnn_feature_columns
        args.feature_index = feature_index
    # deep 
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_save_vectors(
        'data/features/lstm_only_text/{}'.format(args.emb_name),remove_x=remove_deep_x)

    # print(test_dataset[0][1])
    # print(test_dataset[0][1].shape)
    mode_pebg = False if "pebg" not in args else args.pebg
    print(f"mode_pebg={mode_pebg}")
    if mode_pebg:
        train_dataset, train_loader = get_pebg_data_loader(x=train_model_input, x_deep=x_train, y=y_train, shuffle=True,feature_index=feature_index, args=args,lbe_dict=lbe_dict,dense_features=dense_features,sparse_features=sparse_features)
        dev_dataset, dev_loader = get_pebg_data_loader(x=dev_model_input, x_deep=x_dev, y=y_dev, shuffle=False,
                                            feature_index=feature_index, args=args,lbe_dict=lbe_dict,dense_features=dense_features,sparse_features=sparse_features)
        test_dataset, test_loader = get_pebg_data_loader(x=test_model_input, x_deep=x_test, y=y_test, shuffle=False,
                                                feature_index = feature_index, args=args,lbe_dict=lbe_dict,dense_features=dense_features,sparse_features=sparse_features)
    else:
        train_dataset, train_loader = get_data_loader(train_model_input, x_train, y_train, shuffle=True,
                                                feature_index=feature_index, args=args)
        dev_dataset, dev_loader = get_data_loader(dev_model_input, x_dev, y_dev, shuffle=False,
                                            feature_index=feature_index, args=args)
        test_dataset, test_loader = get_data_loader(test_model_input, x_test, y_test, shuffle=False,
                                                feature_index = feature_index, args=args)
       
    deep_feature_dim = test_dataset[0][1].shape[1]
    if add_config:
        args.deep_feature_dim = deep_feature_dim
    return train_dataset, train_loader,dev_dataset, dev_loader,test_dataset, test_loader


# model
def eval_model(model, dataloader, device):
    y_pred = []
    y_true = []
    y_prob = []
    loss_list = []
    wide_loss_list = []
    deep_loss_list = []
    with torch.no_grad():
        model.train(False)
        for idx, data in enumerate(dataloader):
            x_wide, x_deep, labels = [x.to(device) for x in data]
            wide_out, deep_out, wdl_out = model(x_wide, x_deep)
            loss_func = nn.CrossEntropyLoss(reduction='mean')

            batch_wide_loss = loss_func(wide_out, labels)
            batch_deep_loss = loss_func(deep_out, labels)
            batch_wdl_loss = loss_func(wdl_out, labels)

            wide_loss_list.append(float(batch_wide_loss.cpu()))
            deep_loss_list.append(float(batch_deep_loss.cpu()))
            loss_list.append(float(batch_wdl_loss.cpu()))

            sm = nn.Softmax(dim=1)
            pred_prob = wdl_out.cpu()
            pred_prob = sm(pred_prob)
            predict = torch.argmax(pred_prob, axis=1)
            labels = labels.cpu()
            y_pred = y_pred+predict.tolist()
            y_true = y_true+labels.tolist()
            y_prob = y_prob+pred_prob.tolist()
    loss = np.mean(loss_list)
    wide_loss = np.mean(wide_loss_list)
    deep_loss = np.mean(deep_loss_list)
    return y_pred, y_true, y_prob, loss, wide_loss, deep_loss


def get_eval_report(model, dataloader, device, data_set):
    y_pred, y_true, y_prob, loss, wide_loss, deep_loss = eval_model(
        model, dataloader, device)
    model_report = get_model_result_adv(
        y_true, y_pred, data_set=data_set)
    model_report['{}_target_labels'.format(data_set)] = ",".join(
        [str(x) for x in y_true])
    model_report['{}_predicted_labels'.format(data_set)] = ",".join(
        [str(x) for x in y_pred])
    model_report['{}_loss'.format(data_set)] = loss
    model_report['{}_wide_loss'.format(data_set)] = wide_loss
    model_report['{}_deep_loss'.format(data_set)] = deep_loss
    return model_report

from .loss import Loss
from .adt_utils import *
def train_model(data,model,args,device):
    train_dataset, train_loader,dev_dataset, dev_loader,test_dataset, test_loader = data
    # set wandb
    import wandb
    wandb_config={}
    wandb.init(
        # project='aied2022',
        config=wandb_config,
        tags=args.tags if "tags" in args else []
    )
    # wandb.config.update(args)
    
    wandb.watch(model, log_freq=100,log_graph=True)# see model
    
    print("Start Training...")
    # nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # print("=========="*8 + "%s" % nowtime)
    history_list = []
    model = model.to(device)
    opt = torch.optim.Adam(lr=args.lr, params=model.parameters())

    best_dev_score = 1000
    final_report = {}
    patience_num = 0
    gradient_clipping = args.gradient_clipping
    
    loss_func  = Loss(args.loss_type,epsilon=args.epsilon,gamma=args.gamma).get_loss
    
    
    if args.adt_type=='fgm':
        fgm = FGM(model)
        print("Training use FGM ~~")
    elif args.adt_type == 'pgd':
        pgd = PGD(model)
        print("Training use PGD ~~")
    elif args.adt_type == 'freeat':
        freeat = FreeAT(model)
        print("Training use FreeAT ~~")
    elif args.adt_type == 'freelb':
        freelb = FreeLB(model)
        print("Training use FreeLB ~~")
    else:
        print("Training use none adt ~~")
    
    

    for epoch in range(args.epochs):
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*4 + "EPOCH = %s" %
            epoch+"=========="*4 + "%s" % nowtime)
        # weight = [1.0, 1.0, 1.0, 1.0]
        # print('balanced weight: ',weight)
        # train
        train_loss_list = []
        history_eopch = {"epoch": epoch, "best_until_now": 0}
        for data in train_loader:
            model.train(True)
            opt.zero_grad()
            x_wide, x_deep, labels = [x.to(device) for x in data]
            _, _, wdl_out = model(x_wide, x_deep)
            # weight = torch.tensor(weight).to(device)
            
            loss = loss_func(wdl_out, labels)
            train_loss_list.append(float(loss.cpu()))
            loss.backward()
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                
            if args.adt_type=='fgm':#使用fgm对抗训练方式
                #对抗训练
                emb_name = "bilstm.weight"
                fgm.attack(args.adt_epsilon, emb_name=emb_name) # 在embedding上添加对抗扰动
                _, _, wdl_out = model(x_wide, x_deep)
                loss_adv = loss_func(wdl_out, labels)
                loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore(emb_name=emb_name) # 恢复embedding参数    
                
                
            opt.step()

        train_loss = np.mean(train_loss_list)  # train loss
        history_eopch.update(
            {"train_loss": train_loss})

        # eval dev
        dev_report = get_eval_report(model, dev_loader, device, data_set='dev')
        history_eopch.update(dev_report)
        # eval test
        test_report = get_eval_report(model, test_loader, device, data_set='test')
        history_eopch.update(test_report)

        # save best model
        refit = args.refit
        if best_dev_score >= dev_report['dev_loss']:
            best_dev_score = dev_report['dev_loss']
            final_report.update(dev_report)
            final_report.update(test_report)
            patience_num = 0
            if args.save_model and not args.save_dir is None:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                if 'save_path' not in final_report:
                    save_path = os.path.join(
                        args.save_dir, "{}_best.model".format(str(uuid.uuid4())))
                    final_report['save_path'] = save_path
                save_path = final_report['save_path']
                torch.save(model.state_dict(), save_path)
            history_eopch['best_until_now'] = 1
        else:
            patience_num += 1
        if patience_num > args.patience:
            print("early stop")
            break
        info = (epoch,
                train_loss,
                dev_report['dev_loss'],
                refit.replace("dev_", ""),
                dev_report[refit],
                refit.replace("dev_", ""),
                test_report[refit.replace("dev", "test")])
        print(("\nEPOCH = %d,Train loss = %.3f Val loss=%.3f Val %s is %.3f,test %s is %.3f")
            % info)
        history_list.append(history_eopch)
        # wandb.log(history_eopch)
    dfhistory = pd.DataFrame(history_list)
    wandb.log(final_report)
    print('Finished Training...')
    return dfhistory,final_report
