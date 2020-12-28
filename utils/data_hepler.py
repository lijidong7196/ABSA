import json
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import pickle
import numpy as np
import os

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

def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix

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
        for i in range(0,len(lines),3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition('$T$')]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            #build text indices
            # build vocabulary
            full_text = text_left + ' ' + aspect + ' ' + text_right
            tokenizer.fit_on_text(full_text)
            text_indices = tokenizer.text_to_sequence(full_text)
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
    dataset = DataLoader(train,batch_size=25, shuffle=True)




