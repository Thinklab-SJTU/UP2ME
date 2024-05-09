import numpy as np
import torch
import json

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {2: args.learning_rate * 0.5 ** 1, 4: args.learning_rate * 0.5 ** 2,
                     6: args.learning_rate * 0.5 ** 3, 8: args.learning_rate * 0.5 ** 4,
                     10: args.learning_rate * 0.5 ** 5}
    elif args.lradj == 'type2':
        lr_adjust = {5: args.learning_rate * 0.5 ** 1, 10: args.learning_rate * 0.5 ** 2,
                     15: args.learning_rate * 0.5 ** 3, 20: args.learning_rate * 0.5 ** 4,
                     25: args.learning_rate * 0.5 ** 5}
    else:
        lr_adjust = {}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = None

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.best_model = model
        self.val_loss_min = val_loss

def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args


def string_split(str_for_split):
    str_no_space = str_for_split.replace(' ', '')
    str_split = str_no_space.split(',')
    value_list = [eval(x) for x in str_split]

    return value_list


def adjustment(gt, pred):
    # adjust detection result
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def segment_adjust(gt, pred):
    '''
    as long as one point in a segment is labeled as anomaly, the whole segment is labeled as anomaly
    delaited can be found in "Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications (WWW18)"
    gt: long continuous ground truth labels, [ts_len]
    pred: long continuous predicted labels, [ts_len]
    '''
    adjusted_flag = np.zeros_like(pred)

    for i in range(len(gt)):
        if adjusted_flag[i]:
            continue  # this point has been adjusted
        else:
            if gt[i] == 1 and pred[i] == 1:
                # detect an anomaly point, adjust the whole segment
                for j in range(i, len(gt)):  # adjust the right side
                    if gt[j] == 0:
                        break
                    else:
                        pred[j] = 1
                        adjusted_flag[j] = 1
                for j in range(i, 0, -1):  # adjust the left side
                    if gt[j] == 0:
                        break
                    else:
                        pred[j] = 1
                        adjusted_flag[j] = 1
            else:
                continue  # gt=1, pred=0; gt=0, pred=0; gt=0, pred=1, do nothing

    return pred


def segment_adjust_flip(gt, old_pred, flip_idx):
    '''
    flip one prediction from 0 to 1 then re-adjust the segment
    used for compute the roc curve
    '''

    new_pred = old_pred
    delta_true_positive = 0
    delta_false_positive = 0
    if new_pred[flip_idx] == 1: #has already been adjusted by previous flip
        return gt, new_pred, delta_true_positive, delta_false_positive
    else:
        new_pred[flip_idx] = 1
        if (gt[flip_idx] == 1): delta_true_positive += 1
        else: delta_false_positive += 1

    if gt[flip_idx] == 1 and new_pred[flip_idx] == 1:
        #detect an anomaly point, adjust the whole segment
        for j in range(flip_idx + 1, len(gt)): #adjust the right side
            if gt[j] == 0:
                break
            else:
                new_pred[j] = 1
                delta_true_positive += 1

        for j in range(flip_idx - 1, -1, -1): #adjust the left side
            if gt[j] == 0:
                break
            else:
                new_pred[j] = 1
                delta_true_positive += 1

    return gt, new_pred, delta_true_positive, delta_false_positive


def adjusted_precision_recall_curve(gt, anomaly_score):
    precisions = []; recalls = []
    bound_idx = np.argsort(anomaly_score)[::-1]
    sorted_error = anomaly_score[bound_idx] #from large to small

    pred = np.zeros_like(anomaly_score) #start from all zero
    
    true_positive_num = 0
    false_positive_num = 0
    positive_num = np.sum(gt == 1)
    flip_idx = bound_idx[0]
    gt, pred, delta_tp, delta_fp = segment_adjust_flip(gt, pred, flip_idx)
    true_positive_num += delta_tp
    false_positive_num += delta_fp

    precision = 1.0 * true_positive_num / (true_positive_num + false_positive_num); precisions.append(precision)
    recall = 1.0 * true_positive_num / positive_num; recalls.append(recall)

    for flip_idx in bound_idx[1:]:
        gt, pred, delta_tp, delta_fp = segment_adjust_flip(gt, pred, flip_idx)

        true_positive_num += delta_tp
        false_positive_num += delta_fp
        precision = 1.0 * true_positive_num / (true_positive_num + false_positive_num)
        recall = 1.0 * true_positive_num / positive_num
        
        precisions.append(precision); recalls.append(recall)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    AP = ((recalls[1:] - recalls[:-1]) * precisions[1:]).sum()

    return precisions, recalls, AP
