from CRFActionSequencePrediction import ActionRecognitionCRF


import random
import glob
import os
import torch

# get train/val/test splits

random.seed(42)
torch.manual_seed(42)


train_videos = [r.strip() for r in open('experimentalSetup/sequencesTrainAttr.txt').readlines()]
val_videos = [r.strip() for r in open('experimentalSetup/sequencesVal.txt').readlines()]
test_videos = [r.strip() for r in open('experimentalSetup/sequencesTest.txt').readlines()]

videos = glob.glob('videos/*.avi')
video_names = ['-'.join(os.path.split(v)[-1].split('-')[:2]) for v in videos]

video_names = [v for v in video_names if os.path.exists('/home/ubuntu/soham/data/vg/TACoS/annotations/' + v + '.json')]

train_video_names = random.sample(video_names, int(0.8 * len(video_names)))
val_video_names = [v for v in video_names if v not in train_video_names]


# In[ ]:



import numpy as np
import json
import sys
import datetime



def main(args):
    EXP_FOLDER = datetime.datetime.strftime(datetime.datetime.now(),"%d_%m__%H_%M_%S")
    os.system('mkdir -p final-exps')
    os.mkdir('final-exps/' + EXP_FOLDER)

    readme = """
    features: 
    downsampling: 
    has LSTM:
    activation: 
    mean-pool features:
    """

    DOWNSAMPLE_FREQ = 5

    open('exps/' + EXP_FOLDER + '/README', 'w').write(readme)
    logs = open('exps/' + EXP_FOLDER + '/LOGS', 'w')

    tacos_to_ix = json.load(open('/home/ubuntu/soham/data/vg/TACoS/tacos_to_ix.json'))

    chosen_actions = json.load(open('/home/ubuntu/soham/data/vg/TACoS/chosen_actions.json'))
    chosen_actions_ix = set([tacos_to_ix['action_to_ix'][act] - 1 for act in chosen_actions])

    def write_to_log(msg, l):
        sys.stdout.write(msg)
        l.write(msg)
        sys.stdout.flush()
        l.flush()

    # model = ActionRecognitionCRF(2048, 128, tacos_to_ix['action_to_ix'], tacos_to_ix['attr_to_ix'], tacos_to_ix['obj_to_ix'])
    model = ActionRecognitionCRF(args.feature_size, 256, chosen_actions_ix, args.activation, args.diagonal_biasing)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model))

    print("SAVING TO %s" %  EXP_FOLDER)

    for epoch in range(0,10000):
        nlls = []
        i = 0
        write_to_log('\n\n-----------------------------------------------------\n', logs)
        predicted_sequences = []
        train_accuracies = []
        preds_file = open('exps/%s/train-preds-%i' % (EXP_FOLDER, epoch + 1), 'w')
        for v_name in train_video_names:
    #         try:
                write_to_log('Training on %s... ' % (v_name), logs)
                i += 1
                annotation_data = json.load(open('TACoS/annotations/' + v_name + '.json'))
                action = np.array(annotation_data['actions']) - 1

                img_features = np.load('features/%s-%s-features.npy' % (v_name, args.feature_name))

                filtered_ix = [a in chosen_actions_ix and a != -1 for a in action]

                if action.shape[0] != img_features.shape[0]:
                    write_to_log('frame-annotation mismatch\n', logs)
                    continue


                action = action[filtered_ix]
                img_features = img_features[filtered_ix, :]


                # if action.shape[0] >= 1100 or action.shape[0] < 100:
                #     write_to_log('too few/many frames\n', logs)
                #     continue


                # downsample until there are N frames
                DOWNSAMPLE_FREQ = int(img_features.shape[0] / args.downsampled_length)
                start_ix = np.random.randint(0, DOWNSAMPLE_FREQ)
                action = action[start_ix::DOWNSAMPLE_FREQ]
                img_features = img_features[start_ix::DOWNSAMPLE_FREQ, :]


                # img_features_mean = np.zeros(img_features[::DOWNSAMPLE_FREQ,:].shape)

    #             for i in range(action.shape[0]):
    #                 window = img_features[DOWNSAMPLE_FREQ*i:DOWNSAMPLE_FREQ*(i+1),:]
    #                 if window.shape[0] > 0:
    #                     img_features_mean[i, :] = np.mean(window, axis=0)
    #                 else:
    #                     img_features_mean = img_features_[:i, :]
    #                     action_ = action_[:i]
    #                     break


                inputs = Variable(torch.Tensor(img_features))
                targets = Variable(torch.cuda.LongTensor(action))

                write_to_log(" %s labels %s frames" % (str(targets.size()), str(inputs.size())), logs)

                model.zero_grad()
                neg_log_likelihood = model.neg_log_likelihood(inputs, action)
                neg_log_likelihood.backward()
                nlls.append(neg_log_likelihood[0])
                optimizer.step()

                _, pred_seq = model(inputs)
                pred_accuracy = np.mean(pred_seq == action)
                train_accuracies.append(pred_accuracy)
                predicted_sequences.append(pred_seq)

                preds_file.write(v_name + ':   \n')
                preds_file.write('actual: \n' + ' '.join(map(str, action)) + '\n')
                preds_file.write('predicted: \n' + ' '.join(map(str, pred_seq)) + '\n')
                preds_file.write('acc: ' + str(pred_accuracy) + '\n\n------------------------------------------------\n\n')
                preds_file.flush()

                torch.cuda.empty_cache()

                write_to_log(' %d/%d (epoch:%d)  NLL = %.3f    Acc = %.3f\n' % (i , len(train_video_names), epoch, neg_log_likelihood[0], pred_accuracy), logs)
    #         except Exception as e:
    #             write_to_log(' %d/%d (epoch:%d)  failed --> %s\n' % (i, len(train_video_names), epoch, e.__str__()), logs)
    #             pass
        write_to_log('Mean NLL %.3f\n' % np.mean(nlls)[0], logs)
        write_to_log('Mean Accuracy %.3f\n' % np.mean(train_accuracies), logs)
        write_to_log('--------------------------------------------\n', logs)
        torch.save(model.state_dict(), 'exps/' + EXP_FOLDER + '/model-' + str(epoch))


        if (epoch + 1) % 5 == 0:
            accuracies = []
            val_nlls = []
            preds_file = open('exps/%s/val-preds-%i' % (EXP_FOLDER, epoch + 1), 'w')
            val_preds_file = open('exps/%s/val-preds-%i' % (EXP_FOLDER, epoch + 1), 'w')
            for v_name in val_video_names:
                j = 0
                try:

                    write_to_log('Testing on %s... ' % (v_name), logs)
                    j += 1
                    annotation_data = json.load(open('TACoS/annotations/' + v_name + '.json'))
                    action = np.array(annotation_data['actions']) - 1
                    img_features = np.load('features/' + v_name + '-vgg-features.npy')
                    filtered_ix = [a in chosen_actions_ix and a != -1 for a in action]

                    if action.shape[0] != img_features.shape[0]:
                        write_to_log('frame-annotation mismatch\n', logs)
                        continue

                    action = action[filtered_ix]
                    img_features = img_features[filtered_ix, :]

                    # if action.shape[0] >= 1100 or action.shape[0] < 100:
                    #     write_to_log('too few/many frames\n', logs)
                    #     continue

                    action = action[::DOWNSAMPLE_FREQ]
                    img_features = img_features[::DOWNSAMPLE_FREQ, :]
    #                 for i in range(action_.shape[0]):
    #                     window = img_features[DOWNSAMPLE_FREQ*i:DOWNSAMPLE_FREQ*(i+1),:]
    #                     if window.shape[0] > 0:
    #                         img_features_[i, :] = np.mean(window, axis=0)
    #                     else:
    #                         img_features_ = img_features_[:i, :]
    #                         action_ = action_[:i]
    #                         break

    #                 img_features = img_features_mean

                    # opt_features = np.load('frames/' + v_name + '/opt-flow-features.npy')
                    inputs = Variable(torch.Tensor(img_features))
                    targets = Variable(torch.cuda.LongTensor(action))


                    nll, pred_seq = model(inputs)
                    pred_accuracy = np.mean(pred_seq == action)
                    accuracies.append(pred_accuracy)
                    val_nlls.append(nll[0])

                    val_preds_file.write(v_name + ':   \n')
                    val_preds_file.write('actual: \n' + ' '.join(map(str, action)) + '\n')
                    val_preds_file.write('predicted: \n' + ' '.join(map(str, pred_seq)) + '\n')
                    val_preds_file.write('acc: ' + str(pred_accuracy) + '\n\n------------------------------------------------\n\n')
                    val_preds_file.flush()

                    torch.cuda.empty_cache()
                    write_to_log(' %d/%d Frame-Level accuracy: %.4f\n' % (j, len(val_video_names), pred_accuracy), logs)
                except Exception as e:
                    write_to_log(' %d/%d (epoch:%d)  failed --> %s\n' % (j, len(val_video_names), epoch, e.__str__()), logs)

            write_to_log("\nFINAL ACCURACY: %.4f\n\n" % np.mean(accuracies), logs)
            write_to_log("\nFINAL NLL: %.4f\n\n" % np.mean(val_nlls), logs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-name', dest='feature_name', type=str, default='vgg')
    parser.add_argument('--feature-size', dest='feature_size', type=int, default=4096)
    parser.add_argument('--load-model', dest='load_model', type=str, default=None)
    parser.add_argument('--adaptive-downsampling', dest='adaptive', type=bool, default=True)
    parser.add_argument('--use-lstm', dest='use_lstm', type=bool, default=True)
    parser.add_argument('--activation', dest='activation', type=str, default='none')

    parser.add_argument('--downsample-frequency', dest='downsample_frequency', type=int, default=10)
    parser.add_argument('--max-seq-length', dest='adaptive', type=bool, default=True)
    
    parser.add_argument('--downsampled-length', dest='downsampled_length', type=int, default=300)
    parser.add_argument('--meanpool-features', dest='meanpool_features', type=bool, default=False)

    # if not adaptive length, consider sequences only in this length
    parser.add_argument('--min-length', dest='min_length', type=int, default=None)
    parser.add_argument('--max-length', dest='max_length', type=int, default=None)

    parser.add_argument('--diagonal-biasing', dest='diagonal_bias', type=float, default=0.0)
    args = parser.parse_args()
    main(args)
