from NER3.data_utils import prepare_data, minibatches
from NER3.ner_model import NERModel
from NER3.config import Config

# create instance of config
config = Config()
#
# # build model
model = NERModel(config)
model.build()

# create datasets
train_sents, train_labels, length_sentences = prepare_data(config.filename_train,config.vocab_words,config.vocab_chars, config.use_chars, colum=[0,3])
dev_sents, dev_labels, length_sentences_dev = prepare_data(config.filename_dev, config.vocab_words,config.vocab_chars, config.use_chars,colum=[0,3])

# for i,(words, labels)  in enumerate(minibatches(dev_sents,dev_labels,config.batch_size)):
#     if i <2:
#
#         c, sents = zip(*words)
#         print(type(sents))
#         for sent in sents:
#             print([config.id2Word.get(word) for word in sent])
#         print('new batch')
    # pass
# train model
model.train(train_sents, train_labels, dev_sents,dev_labels)

# a = [0, 1, 2, 0, 1, 2,3]
# b =(get_chunks(a,config.vocab_tags))
# c = [0, 1, 1, 0, 0, 2,1]
# d =get_chunks(c,config.vocab_tags)
# print(b)
# print(d)
# re = score(a,c,config.vocab_tags)
# print(re)