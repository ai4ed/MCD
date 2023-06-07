# Multi-turn Classroom Dialogue Dataset (MCD)

We publish the Multi-turn Classroom Dialogue Dataset (MCD) for assessing student performance from One-on-one conversations. The dataset contains the raw teacher's and student's conversations in each example question. And also contain the data after preprocessing, which can be straight used to train models.

The datasets can download from [this url](https://drive.google.com/file/d/1dUHYLKoE09Y8D5I0v4x77m9zIJECaRhI/view?usp=share_link), after downloading the data put the data in `data` folder. The folder structure is following:

`data`
- `dataset`
- `features`
- `word2vec`
- `README.md`

We descripte each folder are follows.

## dataset
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

## features
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

## word2vec
- `word2vec.pkl` : the word2vec file. 
