import json
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

def json_parser(data_path):
    with open(data_path, 'r') as f:
        data_dict = json.load(f)
    id = []
    sentences = []
    polarities = []
    terms = []
    for value in data_dict.values():
        id.append(value['id'])
        sentences.append(value['sentence'])
        polarities.append(value['polarity'])
        terms.append(value['term'])
    df = pd.DataFrame({'id':id,'sentence':sentences,'term':terms,'polarity':polarities})
    return df

def build_vocab(sentences,tokenizer,max_size,min_freq):
    vocab_dict = {}
    raw_data = sentences['sentence'].to_list()
    for sent in tqdm(raw_data):
        sent.strip()
        for word in tokenizer(sent):
            vocab_dict[word] = vocab_dict.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dict.items() if _[1] >= min_freq], key=lambda x:x[1],reverse=True)[:max_size]
    vocab_dict = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dict.update({'UNK':len(vocab_dict),'PAD':len(vocab_dict) + 1})
    return vocab_dict


class ABSADataset(Dataset):
    def __init__(self,data,tokenizer):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass




