from argparse import ArgumentParser
parser = ArgumentParser()


parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--mode', type=str, default="static")
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--seed', type=int, default=3435)
parser.add_argument('--sentence_num_hidden', type=int, default=32)
parser.add_argument('--word_num_hidden', type=int, default=32)
parser.add_argument('--num_labels', type=int, default=3)
parser.add_argument('--emb_name', type=str, default="word2vec")
parser.add_argument('--words_dim', type=int, default=200)
parser.add_argument('--gradient_clipping', type=float, default=1.0)
parser.add_argument('--sparse_emb_dim', type=int, default=4)
parser.add_argument('--refit',default='dev_rmse')
parser.add_argument('--save_model', type=int, default=1)
parser.add_argument('--save_dir',default='model/wd_han')

args = parser.parse_args()
print("args is ",args)



import wandb
wandb_config={}
wandb.init(
    project='aied2022',
    config=wandb_config,
    tags=["HAN"]
)

wandb.config.update(args)


import os
import sys
import torch
torch.set_num_threads(2) 
import pickle
import uuid
import json
import itertools
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils.metrics_utils import *
from utils.competition_utils import load_data, set_seed
from torch.utils.data import DataLoader,Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)


# ### deep 
def load_text_data(data_dir):
    with open(os.path.join(data_dir, 'dataset.pkl'), 'rb') as f:
        x_train, y_train, x_dev, y_dev, x_test, y_test = [np.array(x) for x in pickle.load(f)]
    vocab_map = json.load(open(os.path.join(data_dir, "vocab.json")))
    return x_train, y_train, x_dev, y_dev, x_test, y_test, vocab_map

x_train, y_train, x_dev, y_dev, x_test, y_test, vocab_map = load_text_data('data/features/han/')


words_num = len(vocab_map)
print("words_num is ",words_num)
args.words_num = words_num

emb_dict_path = 'data/word2vec/{}.pkl'.format(args.emb_name)
emb_dict = pickle.load(open(emb_dict_path,'rb'))

# init emb
embeddings = np.zeros((words_num, args.words_dim))
embeddings.shape

ignore_tokens = []
for word, word_index in vocab_map.items():
    if word in emb_dict:
        word_emb = emb_dict[word]
    else:
        word_emb = np.random.random(args.words_dim)
        ignore_tokens.append(word)
    embeddings[word_index] = word_emb
embeddings = torch.Tensor(embeddings)
print("total ignore_tokens is {}".format(len(ignore_tokens)))


config = deepcopy(args)
config.embeddings = embeddings


class Han_Dataset(Dataset):
    def __init__(self, vocab_map, data, y_label, max_words=100, max_sentences=400, x_wide=None):
        self.raw_text = data
        index_text = []
        for para in data:
            para_index = []
            for sentence in para[:max_sentences]:
                para_index.append(
                    [vocab_map.get(x, vocab_map['<unk>']) for x in sentence[:max_words]])
            index_text.append(para_index)
        self.text = index_text
        self.y_label = y_label

    def __getitem__(self, index):  # 根据索引返回数据和对应的标签
        return self.text[index], self.y_label[index]

    def __len__(self):  # 返回文件数据的数目
        return len(self.text)


def collate_fn(batch):
    batch_text = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    max_sen_num = max([len(x) for x in batch_text])
    max_word_num = max([len(x) for x in itertools.chain(*batch_text)])
    # max_sen_num,max_word_num
    x = np.ones((len(batch_text), max_sen_num, max_word_num), dtype=np.int32)
    for para_i, para in enumerate(batch_text):
        for sentence_i, sentence in enumerate(para):
            x[para_i, sentence_i, :len(sentence)] = sentence

    x = torch.Tensor(x).type(torch.IntTensor)
    labels = torch.Tensor(labels).type(torch.LongTensor)
    return x, labels


def get_data_loader(wide_input, deep_input, y_label, shuffle=True, vocab_map=vocab_map,num_workers=1):
    dataset = Han_Dataset(vocab_map, deep_input, y_label)
    dataloader = DataLoader(
        dataset=dataset, shuffle=shuffle, batch_size=args.batch_size, collate_fn=collate_fn,num_workers=num_workers)
    return dataset, dataloader




train_dataset,train_loader = get_data_loader({},x_train,y_train,shuffle=True,num_workers=2)
dev_dataset,dev_loader = get_data_loader({},x_dev,y_dev,shuffle=False,num_workers=2)
test_dataset,test_loader = get_data_loader({},x_test,y_test,shuffle=False,num_workers=2)


# ## 模型训练

from models.han_layers import HAN
model = HAN(config)
model = model.to(device)
opt = torch.optim.Adam(lr=args.lr, params=model.parameters())

def eval_model(model,dataloader):
    y_pred = []
    y_true = []
    y_prob = []
    with torch.no_grad():
        model.train(False)
        for idx, data in enumerate(dataloader):
            x_deep, labels = [x.to(device) for x in data]
            out = model(x_deep)
            loss = nn.CrossEntropyLoss(reduction='mean')
            output = loss(out, labels)
            sm = nn.Softmax(dim=1)
            pred_prob = out.cpu()
            pred_prob = sm(pred_prob)
            predict = torch.argmax(pred_prob, axis=1)
            labels = labels.cpu()
            y_pred = y_pred+predict.tolist()
            y_true = y_true+labels.tolist()
            y_prob = y_prob+pred_prob.tolist()
    return y_pred,y_true,y_prob,float(output.cpu())


best_dev_score = 1000
final_report = {}
patience_num = 0
gradient_clipping = args.gradient_clipping

for e in tqdm(range(args.epochs)):
  
    # print('balanced weight: ',weight)
    # train
    train_loss_list = []
    for data in train_loader:
        model.train(True)
        opt.zero_grad()
        x_deep, labels = [x.to(device) for x in data]
        out = model(x_deep)
        loss = nn.CrossEntropyLoss(reduction='mean')
        output = loss(out, labels)
        train_loss_list.append(float(output.detach().cpu()))
        output.backward()

        if gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        opt.step()
    train_loss = np.mean(train_loss_list)# train loss
    
    # eval dev
    y_dev_pred, y_dev_true, y_dev_prob, dev_loss = eval_model(
        model, dev_loader)
    dev_report = get_model_result_adv(y_dev_true, y_dev_pred, data_set='dev')

    # eval test
    y_test_pred, y_test_true, y_test_prob, test_loss = eval_model(
        model, test_loader)
    test_report = get_model_result_adv(
        y_test_true, y_test_pred, data_set='test')

    
    # save best model
    refit = args.refit
    #if best_dev_score >= dev_report[refit]:
    #best_dev_score = dev_report[refit]
    if best_dev_score >= dev_loss:
        best_dev_score = dev_loss
#         
        final_report.update(dev_report)
        final_report.update(test_report)
        final_report['dev_target_labels'] = ",".join(
            [str(x) for x in y_dev_true])
        final_report['dev_predicted_labels'] = ",".join(
            [str(x) for x in y_dev_pred])
        final_report['test_target_labels'] = ",".join(
            [str(x) for x in y_test_true])
        final_report['test_predicted_labels'] = ",".join(
            [str(x) for x in y_test_pred])
        
        patience_num = 0
        if args.save_model and not args.save_dir is None:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            if 'save_path' not in final_report:
                save_path = os.path.join(args.save_dir, "{}_best.model".format(str(uuid.uuid4())))
                final_report['save_path'] = save_path
            save_path = final_report['save_path']
            torch.save(model, save_path)
            
    else:
        patience_num += 1
    if patience_num > args.patience:
        print("early stop")
        break
    print("train loss is {},val loss is {} ,val {} is {},test {} is {}".format(
        round(train_loss,4), 
        round(dev_loss,4), 
        refit.replace("dev_",""), 
        dev_report[refit],
        refit.replace("dev_",""), 
        test_report[refit.replace("dev","test")]))
wandb.log(final_report)
print("finish train")