from bert_train import ClassficationTrain

classfi = ClassficationTrain('data/classfication_data.xlsx')

classfi.train_start()

if __name__ == '__main__':
    print('train_start')