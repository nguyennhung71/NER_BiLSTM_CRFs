import numpy as np
import tensorflow as tf
import os
import time
from gensim.models import KeyedVectors, Word2Vec
from sklearn.metrics import precision_recall_fscore_support
# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0],ls[1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for words in dataset:
        for char in words:
            vocab_char.update(char)

    return vocab_char


def get_word2vec_vocab(path):

    """Load word embedding Word2vec from file

    Args:
        filename: path to the word2vec vectors

    Returns:
        vocab: length of vocabulary
        word2id, id2word: dictionary
        word_ebeddings_matrix: contain the vector embedding of each word
    """
    start = time.time()
    print('loading model word2vec...')
    w2v_model = KeyedVectors.load_word2vec_format(path, binary=True, encoding='utf-8')
    word_Embeddings_matrix = w2v_model.wv.syn0
    vocab = w2v_model.index2word
    word2id = {}
    id2word = {}
    for i,word in enumerate(vocab):
        word2id.update({word:i})
    id2word = dict(zip(word2id.values(),word2id.keys()))
    print('Finish in {:.2f} sec.'.format(time.time()-start))
    vocab = len(vocab)
    return vocab, word2id, id2word, word_Embeddings_matrix


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w",encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            f.write("{}\t".format(word))
        f.write('_\t')
        f.write('unk')
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        # 'E:\\LAP_TRINH\\Machine_Learning\\NER\\data\\chars.txt'
        with open(filename, encoding='utf-8-sig') as f:
            data = f.read().split()
            for id,w in enumerate(data):
                d[w] = id

    except IOError:
        raise MyIOError(filename)
    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)



def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(sents, labels, minibatch_size):
    """
    Args:
        sents, labels: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    cut_batches= []
    num_batch = len(sents)//minibatch_size
    cut_sents_batches = [sents[i*num_batch: (i+1)*num_batch] for i in range(minibatch_size)]
    cut_labels_batches = [labels[i*num_batch: (i+1)*num_batch] for i in range(minibatch_size)]
    x_batch, y_batch = [], []
    for j in range(num_batch):
        for i in range(minibatch_size):
            if type(cut_sents_batches[i][j][0]) == tuple:
                cut_sents_batches[i][j] = zip(*cut_sents_batches[i][j])
            x_batch.append(cut_sents_batches[i][j])
            y_batch.append(cut_labels_batches[i][j])
        yield x_batch,y_batch
        x_batch, y_batch = [], []


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def get_processing_word(vocab_words=None, vocab_chars=None,
                    chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                char_ids.append(vocab_chars.get(char,len(vocab_chars)-1))

        # 1. get id of word
        if vocab_words is not None:
            if any(char.isdigit() for char in word):
                word_id = vocab_words.get('NUM')
            else:
                word_id = vocab_words.get(word.lower(),0)

        # 2. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word_id
        else:
            return word_id

    return f


def prepare_data(data_path, vocab_word,vocab_chars, use_chars=True, colum=[0,1], Vocab_char= None):
    '''

    :param data_path:
    :param vocab_word:
    :param colum:
    :param Vocab_char:
    :return:
    '''
    data2 = []
    NE2id = {'O': 0, 'B-ORG': 1, 'B-PER': 2, 'B-LOC': 3, 'I-ORG': 4, 'I-PER': 5, 'I-LOC': 6}
    print(vocab_chars)
    convert = get_processing_word(vocab_word,
            vocab_chars, chars=use_chars)
    with open(data_path, encoding='utf-8') as file:
        line = file.read().split('\n')
        data2 = [tuple(l.split('\t')) for l in line]
    train_data = data2
    sent = []
    label = []
    data_sents = []
    data_labels = []
    length_sentences = []
    count =0
    for word_tag in train_data:
        if (word_tag[0] == ''):
            continue
        if word_tag[0] == '.':
            sent.append(convert(word_tag[colum[0]]))
            label.append(NE2id.get(word_tag[colum[1]],0))
            x =np.array(sent)
            data_sents.append(sent)
            data_labels.append(label)
            length_sentences.append(len(sent))
            sent = []
            label = []
            continue

        words = word_tag[colum[0]].split()
        word = words[0]
        if len(words) > 1:
            for word_ in words[1:]:
                word = word+ '_' + word_
        sent.append(convert(word))
        label.append(NE2id.get(word_tag[colum[1]], 0))

    print('len_sentence: ',data_sents[0])
    print('label_sentence: ', data_labels[0])
    print('total: ', len(data_sents))
    # print(data_labels)
    return data_sents, data_labels,length_sentences

