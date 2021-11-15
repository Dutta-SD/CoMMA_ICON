from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import config
import pandas as pd
import numpy as np


class KerasPreProcessor:
    """
    Add text col,
    """

    def __init__(self, text_col, label_cols: dict = None):
        config.set_seed()
        self._text_col = text_col
        self._label_cols: dict = label_cols

    def get_padded_seqs(self):
        self._tokenizer = Tokenizer(num_words=config.VOCAB_SIZE)
        self._tokenizer.fit_on_texts(self._text_col)
        seqs = self._tokenizer.texts_to_sequences(self._text_col)
        X = pad_sequences(seqs, maxlen=config.MAX_SEQ_LEN)
        return X

    def get_targets(self) -> np.array:
        # targets = pd.DataFrame(self._label_cols).values
        # # print(targets.head(10))
        # return np.expand_dims(np.expand_dims(targets, -1), -1)
        return self._label_cols
