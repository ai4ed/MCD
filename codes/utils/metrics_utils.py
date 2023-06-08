from sklearn.metrics import *
import pandas as pd
import numpy as np

def get_model_metrics(y_true, y_pred, show=False,tradeoff=0.5):
    """Compute metrics to evaluate the model of a classification.
    Parameters
    ----------
        y_true: 1d array-like Ground truth (correct) labels.
        y_pred: Predicted labels, as returned by a classifier.
        show: Print result. Default value is False.
    Returns
    -------
        report:Value of the metrics. 
    Examples:
        {
            'Accuracy': 1,
            'Precision': 1,
            'Recall': 1,
            'F_meansure': 1,
            'AUC_Value': 1
        }
    """
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = -1
    y_pred = y_pred > tradeoff
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    if show:
        for name, value in zip(('Accuracy', 'Precision', 'Recall', 'F_meansure', 'AUC_Value'),
                               (accuracy, precision, recall, f1, auc)):
            print('{} : {:.4f}'.format(name, value))
    report = {'Accuracy': round(accuracy, 4),
              'Precision': round(precision, 4),
              'Recall': round(recall, 4),
              'F_meansure': round(f1, 4),
              'AUC_Value': round(auc, 4),
              }
    return report

def print_metris(report):
    columns = ['Accuracy','Precision','Recall','F_meansure','AUC_Value']
    print('\t'.join([str(report[x]) for x in columns]))

def get_multi_class_report(y_true, y_pred):
    report = classification_report(y_true, y_pred,output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    return df_report

def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape, axis=0)
    return output_errors

def abs_1_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true-y_pred)<=1)

def get_model_result(y_true, y_pred):
    '''掌握专用的评估方法'''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = round(mean_absolute_error(y_true, y_pred), 4)
    mape = round(mean_absolute_percentage_error(y_true+1, y_pred+1), 4)
    acc = round(accuracy_score(y_true, y_pred), 4)
    model_result = {"acc": acc,
                    "mae": mae,
                    "mape": mape,
                    "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)),4),
                    "abs_1_score":round(abs_1_score(y_true, y_pred), 4)
                   }
    return model_result

def get_model_result_adv(y_test,y_pred,data_set='train'):
    '''掌握专用的评估方法,加强版'''
    model_result = {}
    for k,v in get_model_result(y_test,y_pred).items():
        model_result['{}_{}'.format(data_set,k)] = v
    for k,v in classification_report(y_test,y_pred,output_dict=True)['macro avg'].items():
        model_result['{}_macro_{}'.format(data_set,k)] = v
    for k,v in classification_report(y_test,y_pred,output_dict=True)['weighted avg'].items():
        model_result['{}_micro_{}'.format(data_set,k)] = v
    return model_result