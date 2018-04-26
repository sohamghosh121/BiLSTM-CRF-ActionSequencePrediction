from CRFActionSequencePrediction import ActionRecognitionCRF, ActionRecognitionLSTM

import argparse 
import random
import glob
import os
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

# get train/val/test splits


# In[ ]:



import numpy as np
import json
import sys
import datetime
import edit_distance



random.seed(42)
torch.manual_seed(42)


def pred_seq_to_ann(seq):
    action_seq = []
    time_ann = []
    last = None
    s__e = [0, None]
    for i, p in enumerate(seq):
        if last is None:
            action_seq.append(p)
            last = p
        elif p != last:
            action_seq.append(p)
            last = p
            s__e[1] = i - 1
            time_ann.append(s__e)
            s__e = [i, None]
        else:
            continue
    s__e[1] = i
    time_ann.append(s__e)
    action_seq = [str(a + 1) for a in action_seq]
    annotations = list(zip(time_ann, action_seq))
    return annotations

def action_accuracy(seq, gt_seq):
    correct_actions = 0
    for g in gt_seq:
        for s in seq:
            if s[0][0] >= g[0][0] or s[0][1] <= g[0][1]:
#                 print(s[0], g[0], (min(s[0][1],g[0][1]) - max(s[0][0],g[0][0])) , float(g[0][1] - g[0][0]))
                action_length = float(g[0][1] - g[0][0]) + 1
                overlap = (min(s[0][1],g[0][1]) - max(s[0][0],g[0][0]) + 1) / action_length
                if overlap >= 0.5:
                    correct_actions += 1
            elif s[0][0] > g[0][1]:
                break
    acc = float(correct_actions)/(len(gt_seq) + 0.000000000000001)
    return acc, (correct_actions, len(gt_seq))



def action_edit_distance(seq, gt_seq):
    seq_ = [s[1] for s in seq]
    gt_seq_ = [s[1] for s in gt_seq]
    sm = edit_distance.SequenceMatcher(a=gt_seq_, b=seq)
    return sm.distance()



class BiLstmCRFTrainer:
    def __init__(self, args):
        self.EXP_FOLDER = datetime.datetime.strftime(datetime.datetime.now(),"%d_%m__%H_%M_%S")
        print("************** SAVING TO %s **************" %  self.EXP_FOLDER)
        os.system('mkdir -p exps')
        os.mkdir('exps/' + self.EXP_FOLDER)

        readme = """
            features: %s
            adaptive: %s or downsample freq: %d
            has LSTM: %s
            has CRF: %s
            activation: %s
            mean-pool features: %s
            diagonal-biasing: %s
        """ % (
            args.feature_name, 
            args.adaptive, 
            args.downsample_frequency, 
            args.use_lstm, 
            args.use_crf, 
            args.activation, 
            args.meanpool_features,
            args.diagonal_bias)

        print(readme)

        open('exps/' + self.EXP_FOLDER + '/README', 'w').write(readme)
        self.args = args
        self.train_logs = open('exps/' + self.EXP_FOLDER + '/LOGS', 'w')
        self.val_logs = open('exps/' + self.EXP_FOLDER + '/LOGS_VAL', 'w')

        self.tacos_to_ix = json.load(open('/home/ubuntu/soham/data/vg/TACoS/tacos_to_ix.json'))
        self.chosen_actions = json.load(open('/home/ubuntu/soham/data/vg/TACoS/chosen_actions.json'))
        self.chosen_actions_ix = set([self.tacos_to_ix['action_to_ix'][act] - 1 for act in self.chosen_actions])

        self.train = True
        
        self.epoch = 0

        if args.use_crf:
            self.model = ActionRecognitionCRF(args.feature_size, args.hidden_sz, self.chosen_actions_ix, args)
        else:
            self.model = ActionRecognitionLSTM(args.feature_size, args.hidden_sz, self.chosen_actions_ix, args)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-4)

    def write_to_log(self, msg):
        if self.train:
            l = self.train_logs
        else:
            l = self.val_logs
        l.write(msg)
        l.flush()

    def save_preds(self, v_name, pred_seq, gt_seq, metrics):
        mode = 'train' if self.train else 'val'
        preds_file = open('exps/%s/%s-preds-%i' % (self.EXP_FOLDER, mode, self.epoch + 1), 'a')
        preds_file.write(v_name + ':   \n')
        preds_file.write('* actual: \n' + ' '.join(map(str, gt_seq)) + '\n')
        preds_file.write('* predicted: \n' + ' '.join(map(str, pred_seq)) + '\n')
        preds_file.write('* acc: ' + str(metrics['accuracy']) + '\n')
        preds_file.write('* edit_distance: ' + str(metrics['edit_distance']) + '\n')
        preds_file.write('* action_overlap: ' + str(metrics['action_accuracy']) + '\n')
        preds_file.write('\n------------------------------------------------\n\n')
        preds_file.close()


    def load_video(self, v_name):
        self.write_to_log('Training on %s... \n' % (v_name))
        annotation_data = json.load(open('TACoS/annotations/' + v_name + '.json'))
        action = np.array(annotation_data['actions']) - 1
        img_features = np.load('features/%s-%s-features.npy' % (v_name, args.feature_name))
        filtered_ix = [a in self.chosen_actions_ix and a != -1 for a in action]

        if action.shape[0] != img_features.shape[0]:
            print('\tframe-annotation mismatch\n')
            return

        action = action[filtered_ix]
        img_features = img_features[filtered_ix, :]

        # downsample until there are N frames
        if args.adaptive:
            DOWNSAMPLE_FREQ = int(img_features.shape[0] / args.adaptive_downsampled_length)
        else:
            DOWNSAMPLE_FREQ = int(img_features.shape[0] / args.adaptive_downsampled_length)
        start_ix = np.random.randint(0, DOWNSAMPLE_FREQ)
        action = action[start_ix::DOWNSAMPLE_FREQ]
        img_features = img_features[start_ix::DOWNSAMPLE_FREQ, :]

        if self.args.meanpool_features:
            pass

        inputs = Variable(torch.Tensor(img_features))
        targets = Variable(torch.cuda.LongTensor(action))
        return inputs, action

    def train_epoch(self, v_names):
        self.metrics = []
        self.losses = []
        for v_name in v_names:
            inputs, targets = self.load_video(v_name)
            self.do_train(inputs, targets)
            self.get_predictions(v_name, inputs, targets)
            # self.write_to_log(str(e) + '\n')
            # print(e)
        self.summarize_metrics(self.metrics)
        self.epoch += 1
        print('[TRAIN] Mean Loss', np.mean(self.losses))
        # save model here
        os.system('mkdir -p ' + self.EXP_FOLDER + '/models')
        model_fname = self.EXP_FOLDER + '/models/model-' + str(self.epoch + 1)
        torch.save({
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict()
            }, model_fname)

    def val_epoch(self, v_names):
        self.train = False
        self.metrics = []
        self.path_scores = []
        for v_name in v_names:
            try:
                inputs, targets = self.load_video(v_name)
                self.get_predictions(v_name, inputs, targets)
            except Exception as e:
                self.write_to_log(str(e) + ' \n')
                print(e)
        self.summarize_metrics(self.metrics)
        self.train = True

    def do_train(self, inputs, targets):
        self.model.zero_grad()
        L = self.model.calculate_loss(inputs, targets)
        L.backward()
        self.losses.append(L.data.cpu().numpy())
        self.optimizer.step()


    def get_predictions(self, v_name, inputs, targets):
        _, pred_seq = self.model(inputs)
        metrics = self.calculate_metrics(pred_seq, targets)
        self.metrics.append(metrics)
        self.save_preds(v_name, pred_seq, targets, metrics)

    def calculate_metrics(self, pred_seq, gt_seq):
        metrics = {}
        metrics['accuracy'] = np.mean(np.array(pred_seq) == np.array(gt_seq))
        gt_seq_ann = pred_seq_to_ann(gt_seq)
        pred_seq_ann = pred_seq_to_ann(pred_seq)
        metrics['edit_distance'] = action_edit_distance(pred_seq_ann, gt_seq_ann)
        metrics['action_accuracy'] = action_accuracy(pred_seq_ann, gt_seq_ann)
        return metrics

    def summarize_metrics(self, metrics):
        frame_level_accuracy = np.mean([m['accuracy'] for m in metrics])
        edit_d = np.mean([m['edit_distance'] for m in metrics])
        action_accuracy = np.mean([m['action_accuracy'][0] for m in metrics])
        
        mode = 'train' if self.train else 'val'
        preds_file = open('exps/%s/%s-preds-%i' % (self.EXP_FOLDER, mode, self.epoch + 1), 'a')
        preds_file.write('\n\nAVERAGE:   \n')
        preds_file.write('* acc: ' + str(frame_level_accuracy) + '\n')
        preds_file.write('* edit_distance: ' + str(edit_d) + '\n')
        preds_file.write('* action_overlap: ' + str(action_accuracy) + '\n')
        preds_file.write('\n------------------------------------------------\n\n')
        preds_file.close()


train_videos = [r.strip() for r in open('experimentalSetup/sequencesTrainAttr.txt').readlines()]
val_videos = [r.strip() for r in open('experimentalSetup/sequencesVal.txt').readlines()]
test_videos = [r.strip() for r in open('experimentalSetup/sequencesTest.txt').readlines()]

videos = glob.glob('videos/*.avi')
video_names = ['-'.join(os.path.split(v)[-1].split('-')[:2]) for v in videos]

video_names = [v for v in video_names if os.path.exists('/home/ubuntu/soham/data/vg/TACoS/annotations/' + v + '.json')]

train_video_names = random.sample(video_names, int(0.8 * len(video_names)))
val_video_names = [v for v in video_names if v not in train_video_names]


"""
    Experiments to run:
        - Imagenet VGG features/Imagenet ResNet features
        - with/without LSTM
        - adaptive length 300/downsample by 5 (another experiment take videos 100 < l < 1000)
            - do mean pooling?
        - diagonal biasing
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-name', dest='feature_name', type=str, default='vgg')
    parser.add_argument('--feature-size', dest='feature_size', type=int, default=4096)
    parser.add_argument('--load-model', dest='load_model', type=str, default=None)
    parser.add_argument('--adaptive-downsampling', dest='adaptive', type=bool, default=True)

    parser.add_argument('--use-lstm', dest='use_lstm', type=int, default=1)
    parser.add_argument('--use-crf', dest='use_crf', type=int, default=1)

    parser.add_argument('--hidden-sz', dest='hidden_sz', type=int, default=256)
    parser.add_argument('--activation', dest='activation', type=str, default='none')

    parser.add_argument('--downsample-frequency', dest='downsample_frequency', type=int, default=10)
    parser.add_argument('--max-seq-length', dest='adaptive', type=int, default=1)
    
    parser.add_argument('--downsampled-length', dest='adaptive_downsampled_length', type=int, default=300)
    parser.add_argument('--meanpool-features', dest='meanpool_features', type=bool, default=False)

    # if not adaptive length, consider sequences only in this length
    parser.add_argument('--min-length', dest='min_length', type=int, default=None)
    parser.add_argument('--max-length', dest='max_length', type=int, default=None)

    parser.add_argument('--diagonal-bias', dest='diagonal_bias', type=float, default=0.0)

    parser.add_argument('--nepochs', dest='nepochs', type=int, default=30)
    parser.add_argument('--val_freq', dest='val_freq', type=int, default=5)


    args = parser.parse_args()

    trainer = BiLstmCRFTrainer(args)

    for epoch in range(0, args.nepochs):
        trainer.train = True
        trainer.train_epoch(train_video_names)
        if (epoch + 1) % args.val_freq == 0:
            trainer.val_epoch(val_video_names)
