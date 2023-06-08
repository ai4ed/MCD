from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--lr', type=float, default=2.5e-05)
parser.add_argument('--num_labels', type=int, default=3)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--max_len', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=421)
parser.add_argument('--gradient_clipping', type=float, default=1.0)
parser.add_argument('--num_hidden_layers', type=int, default=1)
parser.add_argument('--num_attention_heads', type=int, default=1)
parser.add_argument('--refit',default='dev_rmse')
parser.add_argument('--save_model', type=int, default=1)
parser.add_argument('--save_dir',default='/share/tabchen/aied2022_process_assess/model/hbm')
parser.add_argument('--emb_name', type=str, default="edu_roberta")


# CUDA_VISIBLE_DEVICES=3 python hbm_tabchen.py --batch_size 128 --epochs 2


import torch
import os
import sys
sys.path.append('/share/tabchen/izhikang/projects/')
from utils.metrics_utils import get_model_result_adv
from utils.metrics_utils import *
import json
from run_hbm import *

args = parser.parse_args()
print("args is ",args)



import wandb
wandb_config={}
wandb.init(
    project='aied2022',
    config=wandb_config,
    tags=["only_text","HBM","{}_BERT".format(args.emb_name)]
)
wandb.config.update(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()  




gradient_clipping = args.gradient_clipping
batch_size=args.batch_size
seed = args.seed
max_len = args.max_len
lr = args.lr
epochs = args.epochs


## initialize training set 
def load_data(data_dir='data/v6.0/lstm_only_text/'):
    print("Start load data")
    # load data
    X_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_dev = np.load(os.path.join(data_dir, 'x_dev.npy'))
    y_dev = np.load(os.path.join(data_dir, 'y_dev.npy'))
    X_test = np.load(os.path.join(data_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    # # demo
    print("Finish load data")
    #return X_train, y_train, X_dev, y_dev, X_test, y_test

    tensor_train_x = torch.from_numpy(X_train).type(torch.FloatTensor)
    tensor_train_y = torch.from_numpy(y_train).type(torch.LongTensor)

    tensor_dev_x = torch.from_numpy(X_dev).type(torch.FloatTensor)
    tensor_dev_y = torch.from_numpy(y_dev).type(torch.LongTensor)

    tensor_test_x = torch.from_numpy(X_test).type(torch.FloatTensor)
    tensor_test_y = torch.from_numpy(y_test).type(torch.LongTensor)
    hidden_size = X_train.shape[2]
    return tensor_train_x, tensor_train_y, tensor_dev_x, tensor_dev_y, tensor_test_x, tensor_test_y,hidden_size


data_dir = "/share/tabchen/aied2022_process_assess/data/v6.0/lstm_only_text_{}/".format(args.emb_name)
tensor_train_x, tensor_train_y, tensor_dev_x, tensor_dev_y, tensor_test_x, tensor_test_y,hidden_size = load_data(data_dir)

training_set = torch.utils.data.TensorDataset(tensor_train_x,tensor_train_y) # create your datset
dev_set = torch.utils.data.TensorDataset(tensor_dev_x,tensor_dev_y)
test_set = torch.utils.data.TensorDataset(tensor_test_x,tensor_test_y)

trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=4)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

def eval_model(model,dataloader):
    y_pred = []
    y_true = []
    y_prob = []
    with torch.no_grad():
        model.train(False)
        for idx, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = Variable(inputs.to(device)), labels.to(device)
            out = model(inputs)
            sm = nn.Softmax(dim=1)
            pred_prob = out[0].cpu()
            pred_prob = sm(pred_prob)
            predict = torch.argmax(pred_prob, axis=1)
            labels = labels.cpu()
            y_pred = y_pred+predict.tolist()
            y_true = y_true+labels.tolist()
            y_prob = y_prob+pred_prob.tolist()
            # y_edev_prob_pos = np.array(y_edev_prob)[:,1]
            del inputs, labels, out
    return y_pred,y_true,y_prob


config = BertConfig(seq_length=max_len,
                    hidden_size=hidden_size,
                    num_labels=args.num_labels,
                    num_hidden_layers=args.num_hidden_layers,
                    num_attention_heads=args.num_attention_heads)

model = HTransformer(config=config)
model.apply(init_weights)
# model.cuda(args['cuda_num'])
model = model.to(device)
# model.to('cuda')
opt = torch.optim.Adam(lr=lr, params=model.parameters())
# losses = []

best_dev_score = 1000
final_report = {}
patience_num = 0

for e in tqdm(range(epochs)):
    print('\n epoch ',e)
    weight = [1.0,1.0,1.0,1.0]
    # print('balanced weight: ',weight)
    # train
    for i, data in enumerate(tqdm(trainloader)):
        model.train(True)
        opt.zero_grad()
        inputs, labels = data
        if inputs.size(1) > config.seq_length:
            inputs = inputs[:, :config.seq_length, :]
        inputs, labels = Variable(inputs.to(device)), labels.to(device)
        out = model(inputs)
       
        weight = torch.tensor(weight).to(device)
        # weight = torch.tensor(weight).to('cuda')
        loss = nn.CrossEntropyLoss(weight,reduction='mean')
        output = loss(out[0], labels)
        train_loss_tol = float(output.cpu())
        output.backward()

        if gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        opt.step()

    refit = args.refit

    y_dev_pred,y_dev_true,y_dev_prob = eval_model(model,valloader)
    dev_report = get_model_result_adv(y_dev_true,y_dev_pred,data_set='dev')

    y_test_pred,y_test_true,y_test_prob = eval_model(model,testloader)
    test_report = get_model_result_adv(y_test_true,y_test_pred,data_set='test')
    
    if best_dev_score >= dev_report[refit]:
        best_dev_score = dev_report[refit]
        final_report.update(dev_report)
        final_report.update(test_report)
        final_report['dev_target_labels'] = ",".join([str(x) for x in y_dev_true])
        final_report['dev_predicted_labels'] = ",".join([str(x) for x in y_dev_pred])
        final_report['test_target_labels'] = ",".join([str(x) for x in y_test_true])
        final_report['test_predicted_labels'] = ",".join([str(x) for x in y_test_pred])
        patience_num = 0
        if args.save_model and not args.save_dir is None:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_path = os.path.join(args.save_dir,"best.model")
            torch.save(model, save_path)
    else:
        patience_num+=1
    if patience_num>args.patience:
        break
wandb.log(final_report)
print("finish train")
    
        
