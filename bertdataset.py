import gluonnlp as nlp
import numpy as np

from torch.utils.data import Dataset

class BERTDataset(Dataset):
    def __init__(self, dataset, bert_tokenizer, max_len=128, pad=True, pair=False):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[0]]) for i in dataset]
        self.labels = [np.int32(i[1]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))