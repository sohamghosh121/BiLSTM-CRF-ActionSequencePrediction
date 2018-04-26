
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import time
from torch.autograd import Variable


# Helpers

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score +         torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))



# adapted from http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

torch.manual_seed(1)

class ActionRecognitionCRF(nn.Module):

    def __init__(self, input_dim, hidden_dim, predicate_set, args):
        super(ActionRecognitionCRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.predicate_set = predicate_set

        # need to remove 0 index waale
        self.predicate_size = len(predicate_set)  + 2 # add start-stop

        self.use_lstm = args.use_lstm
        self.use_crf = args.use_crf

        if args.use_lstm:
            self.action_lstm = nn.LSTM(input_dim, hidden_dim // 2,
                                num_layers = 1, bidirectional=True) #
            self.hidden2predicate = nn.Linear(hidden_dim, self.predicate_size)
        else:
            self.hidden2predicate = nn.Linear(input_dim, self.predicate_size)

        # self.hidden2predicate = nn.Linear(input_dim, self.predicate_size) # without LSTM
        # THIS BELOW WHEN USING LSTM

        

        if args.activation == 'relu':
            self.hidden2predicate_score = nn.ReLU()
        elif args.activation == 'sigmoid':
            self.hidden2predicate_score = nn.Sigmoid()
        else:
            self.hidden2predicate_score = None
        

        # action layer (detect actions first)
        if args.diagonal_bias != 0.0:
            self.action_transitions = nn.Parameter(
                torch.randn(self.predicate_size, self.predicate_size)
                + args.diagonal_bias * torch.eye(self.predicate_size)) # initialise diagonals to a high value
        else:
            self.action_transitions = nn.Parameter(
                torch.randn(self.predicate_size, self.predicate_size)) # initialise diagonals to a high value

        self.START_TAG_IX = self.predicate_size - 2
        self.STOP_TAG_IX = self.predicate_size - 1

        self.action_transitions.data[self.START_TAG_IX, :] = -10000
        self.action_transitions.data[:, self.STOP_TAG_IX] = -10000

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.predicate_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.START_TAG_IX] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the frames
        feat_ix = 0
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.predicate_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.predicate_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.action_transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
            feat_ix += 1

        terminal_var = forward_var + self.action_transitions[self.STOP_TAG_IX]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, video):
        if self.use_lstm:
            video = video.contiguous()
            video = video.view(video.size()[0], 1, -1)
            lstm_out, _ = self.action_lstm(video)
            feats = self.hidden2predicate(lstm_out.view(video.size()[0], -1))
        else:
            feats = self.hidden2predicate(video)
        return feats

    def _score_conversation(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        last_tag = self.START_TAG_IX # exact point of change doesn't matter
        for i, feat in enumerate(feats):
            score = score + self.action_transitions[tags[i], last_tag] + feat[tags[i]]
        score = score + self.action_transitions[self.STOP_TAG_IX, tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.predicate_size).fill_(-10000.)
        init_vvars[0][self.START_TAG_IX] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.predicate_size):
                next_tag_var = forward_var + self.action_transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.action_transitions[self.STOP_TAG_IX]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.START_TAG_IX  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def calculate_loss(self, feats, tags):
        feats = self._get_lstm_features(feats)
        if self.hidden2predicate_score is not None:
            feats = self.hidden2predicate_score(feats)
        forward_score = self._forward_alg(feats) # change to bilstm-crf later
        gold_score = self._score_conversation(feats, tags)
        return forward_score - gold_score

    def forward(self, video):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        feats = self._get_lstm_features(video)
        if self.hidden2predicate_score is not None:
            feats = self.hidden2predicate_score(feats)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq


class ActionRecognitionLSTM(nn.Module):
    """
        No CRF version
        - if use_lstm is False, this is just doing frame classification
    """
    def __init__(self, input_dim, hidden_dim, predicate_set, args):
        super(ActionRecognitionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.predicate_set = predicate_set
        self.predicate_size = len(predicate_set)  + 2 # add start-stop
        self.use_lstm = args.use_lstm
        if args.use_lstm:
            self.action_lstm = nn.LSTM(input_dim, hidden_dim // 2,
                                num_layers = 1, bidirectional=True) #
            self.hidden2predicate = nn.Linear(hidden_dim, self.predicate_size)
        else:
            self.hidden2predicate = nn.Linear(input_dim, self.predicate_size)
        self.loss = nn.CrossEntropyLoss()


    def _get_lstm_features(self, video):
        if self.use_lstm:
            video = video.contiguous()
            video = video.view(video.size()[0], 1, -1)
            lstm_out, _ = self.action_lstm(video)
            feats = self.hidden2predicate(lstm_out.view(video.size()[0], -1))
        else:
            feats = self.hidden2predicate(video)
        return feats

    def calculate_loss(self, feats, tags):
        feats = self._get_lstm_features(feats) # SEQ_LEN x ACTION_SPACE
        tags_var = Variable(torch.cuda.LongTensor(tags))
        loss = self.loss(feats, tags_var)
        return loss

    def forward(self, video):
        # Get the emission scores from the BiLSTM
        feats = self._get_lstm_features(video)
        _, idx = torch.max(feats, 1)
        tag_seq = idx.data.cpu().numpy()
        return None, tag_seq

