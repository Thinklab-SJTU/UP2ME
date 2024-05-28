import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / (true + 1e-8)))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / (true + 1e-8)))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe

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
    used for compute the precision-recall curve
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
    precisions = []; recalls = []; precisions.append(0.); recalls.append(0.)
    
    bound_idx = np.argsort(anomaly_score)[::-1]

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