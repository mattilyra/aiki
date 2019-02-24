import re
import unicodedata
from pathlib import Path
import logging

import torch
from torch.utils.data import Dataset, DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


SOS_token = 0
EOS_token = 1


class LanguagePairDataset(Dataset):
    def __init__(self, source, source_lang, dest_lang, max_length=-1,
                 reverse=False):
        self.lang = {'source': source_lang, 'dest': dest_lang}
        p = Path(source).expanduser()
        assert p.exists()
        self.source = p
        self.max_length = max_length
        self._init()
        self._load_data(self.source, reverse=reverse)

    def _init(self):
        self.word2index = {'source': {}, 'dest': {}}
        self.word2count = {'source': {}, 'dest': {}}
        self.index2word = {'source': {SOS_token: "SOS", EOS_token: "EOS"},
                           'dest': {SOS_token: "SOS", EOS_token: "EOS"}}
        self.n_words = {'source': 2, 'dest': 2}  # Count SOS and EOS
        self._items = []

    def _load_data(self, fname, reverse=False):
        logger.info(f'Loading {self.__class__} from {self.source}')
        with open(fname, 'r') as fh:
            max_sent_len = 0
            for row in fh:
                dest_sent, source_sent = row.strip().split('\t')
                src_words = normalizeString(source_sent).strip().split()
                dst_words = normalizeString(dest_sent).strip().split()
                max_sent_len = max(max_sent_len, len(src_words), len(dst_words))
                if (self.max_length < 1 or
                        (len(src_words) < self.max_length and
                         len(dst_words) < self.max_length)):
                    if reverse:
                        self.add_sentence(dst_words, key='source')
                        self.add_sentence(src_words, key='dest')
                        self._items.append((dst_words, src_words))
                    else:
                        self.add_sentence(src_words, key='source')
                        self.add_sentence(dst_words, key='dest')
                        self._items.append((src_words, dst_words))
            if self.max_length < 0:
                self.max_length = max_sent_len

    def add_sentence(self, words, key):
        for word in words:
            self.add_word(word, key)

    def add_word(self, word, key):
        if word not in self.word2index[key]:
            self.word2index[key][word] = self.n_words[key]
            self.word2count[key][word] = 1
            self.index2word[key][self.n_words[key]] = word
            self.n_words[key] += 1
        else:
            self.word2count[key][word] += 1

    def get_tensors(self, idx):
        src, dst = self[idx]
        src = [self.word2index['source'][word] for word in src] + [EOS_token]
        dst = [self.word2index['dest'][word] for word in dst] + [EOS_token]
        src = torch.tensor(src, dtype=torch.long, device=DEVICE).view(-1, 1)
        dst = torch.tensor(dst, dtype=torch.long, device=DEVICE).view(-1, 1)
        return src, dst

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]
