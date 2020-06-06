from NER.data_utils import prepare_data
from NER.ner_model import NERModel
from config import Config
from pyvi import ViTokenizer

def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned



def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")
        words_raw = ViTokenizer.tokenize(sentence)
        words_raw = words_raw.strip().split(" ")

        if words_raw == ["exit"]:
            break
        print('Tokenize: ',words_raw)
        preds = model.predict(words_raw)
        print(preds)
        # to_print = align_data({"input": words_raw, "output": preds})

        # for key, seq in to_print.items():
        #     model.logger.info(seq)


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    dev_sents, dev_labels, length_sentences_dev = prepare_data(config.filename_test, config.vocab_words, colum=[0, 3])

    # evaluate and interact
    model.evaluate(dev_sents, dev_labels)
    interactive_shell(model)


if __name__ == "__main__":
    main()
