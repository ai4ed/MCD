# Multi-turn Classroom Dialogue Dataset (MCD)

We publish the Multi-turn Classroom Dialogue Dataset (MCD) and codes for assessing student performance from One-on-one conversations. The dataset contains the raw teacher's and student's conversations in each example question. And also contain the data after preprocessing, which can be straight used to train models.

![](imgs/pipeline.jpg)

## Dataset Details

The datasets can download from [this url](https://drive.google.com/file/d/1dUHYLKoE09Y8D5I0v4x77m9zIJECaRhI/view?usp=share_link), after downloading the data put the data in `data` folder. The folder structure is following:

`data`

- `dataset`
- `features`
- `word2vec`
- `README.md`

We descripte each folder are follows.

### dataset

This folder is our raw data.

- `item_dict.json` : the raw conversation data
  - key : id
  - value: one conversation, see details in `item`
- `item` : json list, each list is one sentence.
  - text : automatic speech recognition (ASR) transcriptions
  - who : speaker, teacher or student
  - begin_time : sentence begin time
  - end_time : sentence end time
- `df_feature_num_label-3.csv` : the extracted features from item
- `train_dev_test.json` : the splited data
  - key : {train,dev,test}
  - value : id list

You can use the following function to read the file.

```python
def load_data(data_dir,fname="df_feature_num_label-3.csv"):
    df = pd.read_csv(os.path.join(data_dir, fname))
    # read the split ids
    split_ids = json.load(
        open(os.path.join(data_dir, 'train_dev_test.json'), 'r'))
    # split_data
    df_train = df[df['new_id'].isin(split_ids['train'])]
    df_dev = df[df['new_id'].isin(split_ids['dev'])]
    df_test = df[df['new_id'].isin(split_ids['test'])]
    print("train num {}\ndev num {}\ntest num {}".format(
        len(df_train), len(df_dev), len(df_test)))
    return df_train, df_dev, df_test
```

### features

This folder contains the preprocessed data.

`wide_features`: the vectors for wide model

- `raw`: handcrafted 25 wide features
- `word2vec`: average of word embeddings from a pre-trained directional skip-gram model.
- `edu_roberta_max`: the sentence embeddings from EduRoBERTa model are obtained by taking the max pooling operation of all the word embeddings from EduRoBERTa.
- `edu_roberta_cls`: the sentence embedding is obtained from the embedding of the [CLS] token in each sentence from EduRoBERTa

`lstm_only_text`: the vectors for deep model

- `edu_roberta_cls`: the sentence embedding is obtained from the embedding of the [CLS] token in each sentence, which keeps all timestamps.

You can use the following function to read the file.

```python
def load_save_vectors(data_dir,remove_x=False):
    print("Start load data form {}".format(data_dir))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_dev = np.load(os.path.join(data_dir, 'y_dev.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    if remove_x:#for wide models
        x_train = np.zeros((y_train.size,1,1))
        x_dev = np.zeros((y_dev.size,1,1))
        x_test = np.zeros((y_test.size,1,1))
    else:
        x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
        x_dev = np.load(os.path.join(data_dir, 'x_dev.npy'))
        x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
    print("Finish load data")
    return x_train, y_train, x_dev, y_dev, x_test, y_test
```

### word2vec

- `word2vec.pkl` : the word2vec file.

## Codes

You can get the results in our paper by running the following command. All codes are in `codes` folder.

### Requirements

Use the following command to create a new conda enviroment.

`conda env create -f conda_env.yml`

### Training

**GBDT**

See `GBDT.ipynb`

**HAN**

```shell
python han.py --emb_name=word2vec --lr=0.001 --mode=static --seed=2022 --word_num_hidden=32
```

**LSTM**

```shell
python lstm_gate.py --area_attention=1 --deep_dnn_dropput=0.1 --deep_dnn_num=1 --deep_output_pooling_mode=avg --lr=0.0001 --lstm_layer_num=2 --lstm_output_dim=256 --max_area_width=3 --seed=3407 --use_attention=0
```

**LSTM+MAA**

```shell
python lstm_gate.py --area_attention=1 --deep_dnn_dropput=0.1 --deep_dnn_num=1 --deep_output_pooling_mode=avg --lr=0.0001 --lstm_layer_num=2 --lstm_output_dim=256 --max_area_width=3 --seed=3407 --use_attention=1
```

**LSTM+GA**

```shell
python lstm_gate.py --area_attention=1 --deep_dnn_dropput=0.1 --deep_dnn_num=1 --deep_output_pooling_mode=se_att --lr=0.0001 --lstm_layer_num=2 --lstm_output_dim=256 --max_area_width=3 --seed=3407 --use_attention=0
```

**BERT**

```shell
python bert.py --area_attention=0 --emb_name=edu_roberta_cls --lr=0.0001 --num_attention_heads=8 --num_hidden_layers=1 --output_dense_num=2 --seed=421
```

**LLaMA-7B**

```shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=21009 \
    train_llama.py \
    --model_name_or_path {model_name_or_path} \
    --data_path {data_path} \
    --bf16 True \
    --output_dir {output_dir} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 38 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True
```

**cLLaMA-7B**

```shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=21009 \
    train_llama.py \
    --model_name_or_path {cLLaMA_model_name_or_path} \
    --data_path {data_path} \
    --bf16 True \
    --output_dir {output_dir} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 38 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True
```

**Ours**

```shell
python lstm_gate.py --area_attention=1 --deep_dnn_dropput=0.1 --deep_dnn_num=1 --deep_output_pooling_mode=se_att --lr=0.0001 --lstm_layer_num=2 --lstm_output_dim=256 --max_area_width=3 --seed=3407 --use_attention=1
```
