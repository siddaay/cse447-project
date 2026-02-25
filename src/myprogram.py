#!/usr/bin/env python
import os
import pickle
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict, Counter


class MyModel:

    def __init__(self):
        self.ngram_models = None
        self.n = 6
        self.fallback = [' ', 'e', 't']  # universal fallback chars

    @classmethod
    def load_training_data(cls):
        # Training is done offline in Colab; we load the saved model instead
        return []

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # strip trailing newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # Training is done offline; weights are saved to work_dir manually
        print('Training is done offline. Load pre-trained model with load().')

    def predict_top3(self, context):
        """Predict top 3 next chars using stupid backoff from n down to 1."""
        for k in range(min(self.n, len(context) + 1), 0, -1):
            ctx = context[-(k - 1):] if k > 1 else ''
            if ctx in self.ngram_models[k]:
                top3 = [c for c, _ in self.ngram_models[k][ctx].most_common(3)]
                if top3:
                    # Pad to 3 if fewer candidates exist
                    for fb in self.fallback:
                        if len(top3) >= 3:
                            break
                        if fb not in top3:
                            top3.append(fb)
                    return top3
        return self.fallback[:]

    def run_pred(self, data):
        """
        data: list of context strings (one per line from input.txt)
        returns: list of 3-char prediction strings
        """
        preds = []
        for context in data:
            try:
                top3 = self.predict_top3(context)
                # Join into a single 3-char string (as the grader expects)
                preds.append(''.join(top3[:3]))
            except Exception as e:
                print(f'Error on context {repr(context)}: {e}', file=sys.stderr)
                preds.append(''.join(self.fallback))
        return preds

    def save(self, work_dir):
        # Model is saved from Colab as ngram_models.pkl — nothing to save here
        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
        with open(checkpoint_path, 'wt') as f:
            f.write('ngram_model_v1')
        print(f'Checkpoint marker saved to {checkpoint_path}')

    @classmethod
    def load(cls, work_dir):
        model = cls()
        pkl_path = os.path.join(work_dir, 'ngram_models.pkl')
        print(f'Loading n-gram model from {pkl_path} ...', file=sys.stderr)
        with open(pkl_path, 'rb') as f:
            model.ngram_models = pickle.load(f)
        print('Model loaded.', file=sys.stderr)
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        model = MyModel()
        train_data = MyModel.load_training_data()
        model.run_train(train_data, args.work_dir)
        model.save(args.work_dir)

    elif args.mode == 'test':
        model = MyModel.load(args.work_dir)
        test_data = MyModel.load_test_data(args.test_data)
        print(f'Predicting on {len(test_data)} inputs...', file=sys.stderr)
        pred = model.run_pred(test_data)
        assert len(pred) == len(test_data), \
            'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        MyModel.write_pred(pred, args.test_output)
        print(f'Wrote predictions to {args.test_output}', file=sys.stderr)

    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))