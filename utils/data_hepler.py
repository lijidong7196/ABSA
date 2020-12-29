import json
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import pickle
import numpy as np
import os
EMBED_PATH = '../glove.840B.300d/glove.6B.300d.txt'
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

# load the embedding matrix
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

def load_embeddings(embed_dir=EMBED_PATH):
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(embed_dir)))
    return embedding_index

def build_embedding_matrix(word_index, embeddings_index, max_features, lower = True, verbose = True):
    embedding_matrix = np.zeros((max_features, 300))
    for word, i in tqdm(word_index.items(),disable = not verbose):
        if lower:
            word = word.lower()
        if i >= max_features: continue
        try:
            embedding_vector = embeddings_index[word]
        except:
            embedding_vector = embeddings_index["unknown"]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def build_matrix(word_index, embeddings_index):
    embedding_matrix = np.zeros((len(word_index) + 1,300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embeddings_index[word]
        except:
            embedding_matrix[i] = embeddings_index["unknown"]
    return embedding_matrix

def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset(Dataset):
    def __init__(self,filename,tokenizer):
        file = open(filename,'r',encoding='utf8',newline='\n',errors='ignore')
        lines = file.readlines()
        self.data = []
        self.asp2idx = {}
        self.tokenizer = tokenizer
        for i in range(0,len(lines),3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition('$T$')]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            #build text indices
            # build vocabulary
            full_text = text_left + ' ' + aspect + ' ' + text_right
            self.tokenizer.fit_on_text(full_text)
            text_indices = self.tokenizer.text_to_sequence(full_text)
            polarity = int(polarity) + 1
            data = {'text':text_indices,
                    'aspect':aspect,
                    'polarity':polarity}
            if aspect not in self.asp2idx:
                self.asp2idx[aspect] = len(aspect)

            self.data.append(data)
        for item in self.data:
            item['aspect'] = self.asp2idx[item['aspect']]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    tokenizer = Tokenizer(30)
    train = ABSADataset('../dataset/data/14lap/Laptops_Train.xml.seg',tokenizer)

    embedding_index = load_embeddings()
    dataset = DataLoader(train,batch_size=25, shuffle=True)



