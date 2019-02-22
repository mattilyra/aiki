from io import open
import sys
import unicodedata
import string
import re
import random
import argparse
from pathlib import Path
import time
import math
import logging

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from data import LanguagePairDataset, SOS_token, EOS_token


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 10

teacher_forcing_ratio = 0.0

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(lineno)d %(message)s',
                    level=logging.INFO)
# logget.getLogger().setLevel(logging.INFO)

def _source_dir(dir_path):
    p = Path(dir_path).expanduser().absolute()
    assert p.exists(), f'Source directory {str(p)} does not exists'
    return p

parser = argparse.ArgumentParser()
parser.add_argument('--source-lang', required=True, type=str)
parser.add_argument('--dest-lang', required=True, type=str)
parser.add_argument('--data-directory', default='./data/', type=_source_dir)
parser.add_argument('--n-iter', type=int)
parser.add_argument('--n-hidden-enc', type=int, default=64)
parser.add_argument('--n-hidden-dec', type=int, default=64)
parser.add_argument('--max-sentence-length', type=int, default=12)
parser.add_argument('--print-every', type=int, default=1000)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    outputs = []
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


def train(input_tensor, target_tensor,
          encoder, decoder,
          encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=DEVICE)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden)#, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)#, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(dataset, encoder, decoder, n_iters,
               print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [dataset.get_tensors(i) for i in random.sample(range(len(dataset)), k=n_iters)]
    criterion = nn.NLLLoss()

    for i_itr in range(0, n_iters):
        training_pair = training_pairs[i_itr]
        x, y = training_pair

        loss = train(x, y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if i_itr > 0 and i_itr % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f'{timeSince(start, i_itr / n_iters)} ({i_itr} {i_itr / n_iters:.0%})'
                  f' {print_loss_avg:.4f}')

        #if iter % plot_every == 0:
        #    plot_loss_avg = plot_loss_total / plot_every
        #    plot_losses.append(plot_loss_avg)
        #    plot_loss_total = 0

    # showPlot(plot_losses)


def decode(encoder, decoder, input_tensor, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=DEVICE)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden) #, encoder_outputs)
            # decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(topi.item())

            decoder_input = topi.squeeze().detach()

        return decoded_words# , decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, dataset, n=10):
    for i in range(n):
        idx = random.choice(range(len(dataset)))
        pair = dataset[idx]
        src, dst = dataset.get_tensors(idx)
        print('>', pair[0])
        print('=', pair[1])
        output_idx = decode(encoder, decoder, src, max_length=dataset.max_length)
        output_words = [dataset.index2word['dest'].get(wi, 'UNK') for wi in output_idx]
        output_sentence = ' '.join(output_words + ['<EOS>'])
        print('<', output_sentence)
        print('')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.n_hidden_enc != args.n_hidden_dec:
        raise NotImplementedError('Different hidden sizes is not currently supported.')
    dataset = LanguagePairDataset(Path(args.data_directory) / f'{args.source_lang}.txt',
                                  args.source_lang, args.dest_lang,
                                  max_length=args.max_sentence_length)
    print(dataset.max_length)
    logger.info(f'Loaded dataset with {len(dataset)} items.')
    logger.info(f'Random sentence pair: "{random.choice(dataset)}"')

    encoder = EncoderRNN(input_size=dataset.n_words['source'],
                         hidden_size=args.n_hidden_enc).to(DEVICE)
    decoder = DecoderRNN(hidden_size=args.n_hidden_dec,
                         output_size=dataset.n_words['dest']).to(DEVICE)
    trainIters(dataset, encoder, decoder, args.n_iter, print_every=args.print_every)

    evaluateRandomly(encoder, decoder, dataset)
