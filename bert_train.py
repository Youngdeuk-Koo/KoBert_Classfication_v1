import pandas as pd
import gluonnlp as nlp
import torch

from bertdataset import BERTDataset
from bertclassifier import BERTClassifier
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import nn
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ClassficationTrain():

    def __init__(self, PATH):
        df = pd.read_excel(PATH)
        self.data = [(text, label) for text, label in zip(df['TEXT'], df['LABEL'])]
        self.bertmodel, vocab = get_pytorch_kobert_model()
        self.tok = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)



    def data_split(self, size=0.2, seed=0):
        dataset_train, dataset_test = train_test_split(self.data, test_size=size, random_state=seed)

        return dataset_train, dataset_test


    def data_loader(self, batch_size=64, num_workers=5):

        dataset_train, dataset_test = self.data_split()

        train_dataloader = torch.utils.data.DataLoader(
            BERTDataset(
                dataset_train, 
                self.tok, 
                max_len=128, 
                pad=True, 
                pair=False
                ), 
                batch_size=batch_size, 
                num_workers=num_workers
                )

        test_dataloader = torch.utils.data.DataLoader(
            BERTDataset(
                dataset_test, 
                self.tok, 
                max_len=128, 
                pad=True, 
                pair=False
                ), 
                batch_size=batch_size, 
                num_workers=num_workers
                )
        
        return train_dataloader, test_dataloader



    def model_schedule_status(self, rate=0.5, learning_rate=5e-5, num_epochs=5, warmup_ratio=0.1):

        self.train_dataloader, self.test_dataloader = self.data_loader()
        
        model = BERTClassifier(self.bertmodel,  dr_rate=rate).to(self.device)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        t_total = len(self.train_dataloader) * num_epochs
        warmup_step = int(t_total * warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

        return model, optimizer, loss_fn, scheduler


    def calc_accuracy(self, x, y):
        _, max_indices = torch.max(x, 1)
        train_acc = (max_indices == y).sum().data.cpu().numpy()/max_indices.size()[0]

        return train_acc


    def train_start(self, num_epochs=5, max_grad_norm=1, log_interval=200):
        
        model, optimizer, loss_fn, scheduler = self.model_schedule_status()    

        for e in range(num_epochs):
            train_acc = 0.0
            test_acc = 0.0
            model.train()
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(self.train_dataloader)):
                optimizer.zero_grad()
                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                valid_length= valid_length
                train_label = label.long().to(self.device)
                train_out = model(token_ids, valid_length, segment_ids)
                train_loss = loss_fn(train_out, train_label)
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  
                train_acc += self.calc_accuracy(train_out, train_label)
                if batch_id % log_interval == 0:
                    print("epoch :{} / train acc :{} / train loss :{}".format(e+1, train_acc / (batch_id+1), train_loss.data.cpu().numpy()))


            model.eval()
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(self.test_dataloader)):
                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                valid_length= valid_length
                test_label = label.long().to(self.device)
                test_out = model(token_ids, valid_length, segment_ids)
                test_loss = loss_fn(test_out, test_label)
                test_loss.backward()
                test_acc += self.calc_accuracy(test_out, test_label)
            print("epoch :{} / test acc :{} / test loss :{}".format(e+1, test_acc / (batch_id+1), test_loss))


        torch.save(model.state_dict(), 'model_save/model_' + f'{test_acc:.2f}.pt' )


# aaa = ClassficationTrain('data/classfication_data.xlsx')
# print(aaa.data)