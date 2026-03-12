#!/usr/bin/env python
import lzma
import os
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Dict, Optional
import random

from tqdm import tqdm

from ngram import KNCharLM


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self, ngram_lm: Optional[KNCharLM] = None):
        self.ngram_lm = ngram_lm or KNCharLM()

    @classmethod
    def load_training_data(cls, fnames, subset=None, save_cache=False):
        return {'fnames': fnames, 'subset': subset, 'save_cache': save_cache}

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data: Dict[str, object], work_dir):
        fnames = data['fnames']
        subset = data['subset']
        save_cache = data.get('save_cache', False)
        for fname in fnames:
            cache_enabled = fname.endswith('.xz') and save_cache
            cache_name = os.path.basename(fname[:-3])
            if cache_enabled and subset is not None:
                cache_name = f'{cache_name}.subset{subset}'
            cache_target = os.path.join(work_dir, cache_name) if cache_enabled else None
            cache_tmp = f'{cache_target}.tmp' if cache_enabled else None

            if fname.endswith('.xz'):
                stream = lzma.open(fname, 'rt', encoding='utf-8', errors='replace')
            else:
                stream = open(fname, 'rt')

            cache_file = None
            if cache_enabled:
                cache_file = open(cache_tmp, 'wt', encoding='utf-8')

            try:
                iterator = tqdm(stream, total=subset, desc='Training ngram ({})'.format(os.path.basename(fname)))
                for idx, line in enumerate(iterator):
                    if subset is not None and idx >= subset:
                        break
                    text = line.rstrip('\n').lower()
                    self.ngram_lm.update_from_text(text)
                    if cache_file is not None:
                        cache_file.write(f'{text}\n')
            finally:
                stream.close()
                if cache_file is not None:
                    cache_file.close()

            if cache_enabled:
                os.replace(cache_tmp, cache_target)
                print('Cached decompressed training text to {}'.format(cache_target))

    def run_pred(self, data):
        preds = []
        for inp in data:
            top_guesses = self.ngram_lm.topk_next(inp, k=3)
            preds.append(''.join(top_guesses))
        return preds

    @staticmethod
    def checkpoint_path(work_dir: str, experiment_name: str) -> str:
        suffix = f'.{experiment_name}' if experiment_name else ''
        return os.path.join(work_dir, f'model.checkpoint{suffix}')

    def save(self, work_dir, experiment_name=''):
        checkpoint_path = self.checkpoint_path(work_dir, experiment_name)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.ngram_lm, f)
        print('Saved model checkpoint to {}'.format(os.path.abspath(checkpoint_path)))

    @classmethod
    def load(cls, work_dir, experiment_name=''):
        checkpoint_path = cls.checkpoint_path(work_dir, experiment_name)
        with open(checkpoint_path, 'rb') as f:
            ngram_lm = pickle.load(f)
        return MyModel(ngram_lm=ngram_lm)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--train_data', nargs='+', help='path(s) to training data', default=['data/en.txt.subset1000000'])
    parser.add_argument('--train_subset', type=int, default=None, help='train on first N lines only')
    parser.add_argument('--save_data_cache', action='store_true', help='save decompressed training cache into work_dir')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    parser.add_argument('--experiment_name', default='', help='optional experiment name for checkpoint suffix')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data from {}'.format(', '.join(args.train_data)))
        train_data = MyModel.load_training_data(
            args.train_data,
            subset=args.train_subset,
            save_cache=args.save_data_cache,
        )
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir, experiment_name=args.experiment_name)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir, experiment_name=args.experiment_name)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
