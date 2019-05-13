from numpy import random
import os
import keras
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import save_model, load_model
from PIL import Image
from sklearn.metrics import accuracy_score


def aggregate_bagging(result_dic, method='mean'):
    '''
    A function that aggregates predictions of bagged models
    :param result_dic: A dictionary containing all model predictions
    :param method: the method of aggergation
    :return: one final, aggregated predicion
    '''
    # create an empty array that will store the final aggregated predictions.
    final_pred = np.empty((result_dic['0'].shape[0], result_dic['0'].shape[1]))

    if method == 'weighted_absolute':
        # get absolute weights
        val_acc_list = [float(np.load(file)) for file in os.listdir() if file[:4] == 'val_']
        for i, pred in enumerate(result_dic.values()):
            # add model predictions to aggregated predictions
            final_pred += val_acc_list[i] * pred
        # average aggregated predictions
        final_pred /= len(result_dic.items())
        # one hot encode
        final_pred = keras.utils.to_categorical(np.argmax(final_pred, axis=1), num_classes=34, dtype=int)

    if method == 'weighted_ranked':
        # get ranked weights
        val_acc_list = [float(np.load(file)) for file in os.listdir() if file[:4] == 'val_']
        val_ranking = sorted(val_acc_list)
        weights = [(val_ranking.index(acc)+1) / len(val_acc_list) for i, acc in enumerate(val_acc_list)]
        for i in range(len(result_dic.items())):
            # add model predictions to aggregated predictions
            final_pred += weights[i] * result_dic['{}'.format(i)]
        # average aggregated predictions
        final_pred /= len(result_dic.items())
        # one hot encode
        final_pred = keras.utils.to_categorical(np.argmax(final_pred, axis=1), num_classes=34, dtype=int)

    elif method == 'mean':
        for i in range(len(result_dic.items())):
            # add model predictions to aggregated predictions
            final_pred += result_dic['{}'.format(i)]
        # average aggregated predicitons
        final_pred /= len(result_dic.items())
        # one hot encode
        final_pred = keras.utils.to_categorical(np.argmax(final_pred, axis=1), num_classes=34, dtype=int)

    elif method == 'majority':
        count = 0
        final_pred2 = np.empty((result_dic['0'].shape[0], result_dic['0'].shape[1]))
        for i in range(len(result_dic.items())):
            final_pred2 += keras.utils.to_categorical(np.argmax(result_dic['{}'.format(i)], axis=1), num_classes=34, dtype=int)
        # get rid of highest & lowest predicting models
        for i, row in enumerate(final_pred2):
            max_indexes = np.where(row == np.amax(row))
            if len(max_indexes[0]) > 1:
                count += 1
                index = np.random.choice(max_indexes[0])
            else:
                index = max_indexes[0][0]
            final_pred[i] = keras.utils.to_categorical(index, num_classes=34, dtype=int)
        final_pred = np.asarray(final_pred, dtype=int)

    elif method == 'trimmed_mean':
        for i in range(final_pred.shape[0]):
            row_preds = [(np.min(result_dic['{}'.format(j)][i]), np.max(result_dic['{}'.format(j)][i])) for j in range(len(result_dic.items()))]
            min_model = row_preds.index(min(row_preds, key=lambda x: x[0]))
            max_model = row_preds.index(max(row_preds, key=lambda x: x[1]))
            if min_model == max_model:
                if random.randint(0, 1):
                    max_model = row_preds.index(max(row_preds[:max_model] + row_preds[max_model+1:], key=lambda x: x[1]))
                else:
                    min_model = row_preds.index(min(row_preds[:min_model] + row_preds[min_model+1:], key=lambda x: x[0]))
            for j, pred in enumerate(result_dic.values()):
                if j == min_model or j == max_model:
                    continue
                final_pred += pred
        final_pred = keras.utils.to_categorical(np.argmax(final_pred, axis=1), num_classes=34, dtype=int)
    if method == 'majority':
        print('Amount of ties = ', count)
    return final_pred


def drop_x_models(result_dic, n_drop, y_test):
    '''
    A function that drops the lowest performing model from a collection of models
    :param result_dic: a dictionary containing model predictions
    :param n_drop: number of models that should be dropped
    :param y_test: the labels for y_test
    :return: result_dic - n_drop amount of models
    '''
    os.chdir('D:/thijs/downloads/THESIS - models - BAGGING/')

    acc_scores = [accuracy_score(y_test,
                                 keras.utils.to_categorical(np.argmax(result_dic['{}'.format(i)], axis=1),
                                                            num_classes=34, dtype=int)) for i in range(0, 9)]
    acc_scores_backup = [accuracy_score(y_test,
                                 keras.utils.to_categorical(np.argmax(result_dic['{}'.format(i)], axis=1),
                                                            num_classes=34, dtype=int)) for i in range(0, 9)]
    # get accuracy scores of models to keep
    for i in range(n_drop-1):
        min_index = acc_scores.index(min(acc_scores))
        acc_scores = acc_scores[:min_index] + acc_scores[min_index+1:]

    # create new collection of models and return it
    result_dic2 = dict()
    for i, acc in enumerate(acc_scores):
        ind = acc_scores_backup.index(acc)
        result_dic2['{}'.format(i)] = result_dic['{}'.format(ind)]

    return result_dic2


def get_results(method='Bagging', aggregation='majority', output='accuracy', drop=0):
    '''
    A function that returns MCA or predictions of ensembles
    :param method: The ensemble method
    :param aggregation: method for aggregating (only applicable for bagged models
    :param output: what the functoin should return
    :param drop: nr of models (with lowest val_acc) to drop
    :return: MCA if output == 'accuracy', y_pred if output == 'pred'
    '''

    data_dir = input('Please enter the directory where "partition", "labels" and "y_test" are stored')
    # D:/Rijksmuseum/
    os.chdir(data_dir)
    with open('./partition', 'rb') as pickle_in:
        partition = pickle.load(pickle_in)
    with open('./labels', 'rb') as pickle_in:
        labels = pickle.load(pickle_in)
    y_test = np.load('y_test.npy')
    train_IDs = partition['train']

    if method == 'Bagging':
        bagged_dir = input('Please enter the directory where your bagged models are stored')
        #'D:/thijs/downloads/THESIS - models - BAGGING/'
        os.chdir(bagged_dir)
        result_arrays = [file for file in os.listdir() if file[:4] == 'test']
        models = [file for file in os.listdir() if file[:4] == 'Dist']

        result_dic = dict()
        for i, result in enumerate(result_arrays):
            this_result = np.load(result)
            result_dic['{}'.format(i)] = this_result

        if drop != 0:
            result_dic = drop_x_models(result_dic, drop, y_test)

        if aggregation == 'weighted_rank':
            y_pred = aggregate_bagging(result_dic, method='weighted_ranked')
        elif aggregation == 'weighted_absolute':
            y_pred = aggregate_bagging(result_dic, method='weighted_absolute')
        elif aggregation == 'mean':
            y_pred = aggregate_bagging(result_dic, method='mean')
        elif aggregation == 'trimmed_mean':
            y_pred = aggregate_bagging(result_dic, method='trimmed_mean')
        elif aggregation == 'majority':
            y_pred = aggregate_bagging(result_dic, method='majority')
        else:
            print('not a valid method')
            return

    elif method == 'CONFADABOOST':
        import math
        confadaboost_dir = input('Please enter the directory of models trained with CONFADABOOST')
        # confadaboost_dir = 'D:/thijs/downloads/THESIS - models - CONFADABOOST/'
        n_models = int(input('Please enter the number of models'))
        # calculate weights
        os.chdir(confadaboost_dir)

        Dt_dic = dict()

        final_pred = np.empty((y_test.shape[0], y_test.shape[1]), dtype=float)

        for i in range(n_models):
            with open('Dt_dic-{}'.format(i), 'rb') as pickle_in:
                Dt_dic['{}'.format(i)] = pickle.load(pickle_in)

            probs = np.load('pred_prob-{}.npy'.format(i))
            preds = keras.utils.to_categorical(np.argmax(np.load('pred_prob-{}.npy'.format(i)), axis=1), num_classes=34,
                                               dtype=int)
            # create binary array: 0 correctly classified, 1 incorrectly classified
            error_array = np.asarray([0 if np.array_equal(pred, y_test[j]) else 1 for j, pred in enumerate(preds)])
            # calculate error using error_array, and the probabilities
            error = np.sum(np.asarray(
                [(Dt_dic['{}'.format(i)][train_IDs[j]] * max(probs[j])) if el == 1 else 0 for j, el in
                 enumerate(error_array)]))
            # calculate beta
            Beta = 0.5 * math.log((1 - error) / error)
            # weight predictions
            final_pred += 1 / Beta * probs
        # final pred
        y_pred = keras.utils.to_categorical(np.argmax(final_pred, axis=1), num_classes=34, dtype=int)

    elif method == 'Disturblabel':
        disturb_dir = input('Please enter the directory containing models trained with DisturbLabel')
        # disturb_dir = 'D:/thijs/downloads/THESIS - models - DISTURBLABEL/'
        alpha = input('Please enter the value of alpha')
        dropout = input('Please enter the value of Dropout')

        os.chdir(disturb_dir)
        preds = [file for file in os.listdir() if file[:4] == 'pred']
        for pred in preds:
            if alpha in pred and dropout in pred:
                y_pred = keras.utils.to_categorical(np.argmax(np.load(pred), axis=1), num_classes=34, dtype=int)

    elif method == 'SAMME':
        import math
        samme_dir = input('Please enter the directory of models trained with SAMME')
        # samme_dir = 'D:/thijs/downloads/THESIS - models - SAMME/'
        n_models = int(input('Please input the number of models'))
        os.chdir(samme_dir)
        Dt_dic = dict()
        final_pred = np.empty((y_test.shape[0], y_test.shape[1]), dtype=float)

        for i in range(n_models):
            with open('Dt_dic-{}'.format(i), 'rb') as pickle_in:
                Dt_dic['{}'.format(i)] = pickle.load(pickle_in)
            probs = np.load('pred_prob-SAMME-{}.npy'.format(i))
            preds = keras.utils.to_categorical(np.argmax(np.load('pred_prob-SAMME-{}.npy'.format(i)), axis=1), num_classes=34,
                                               dtype=int)
            error_arr = np.asarray([0 if np.array_equal(pred, y_test[j]) else 1 for j, pred in enumerate(preds)])
            error_w = np.sum(np.asarray([Dt_dic['{}'.format(i)][train_IDs[j]] if el == 1 else 0 for j, el in enumerate(error_arr)]))
            error_t = np.sum(np.asarray([Dt_dic['{}'.format(i)][ID] for ID in train_IDs]))
            error = error_w / error_t
            Beta = math.log( (1-error) / error) + math.log(33)
            final_pred += Beta * probs

        y_pred = keras.utils.to_categorical(np.argmax(final_pred, axis=1), num_classes=34, dtype=int)

    if output == 'acc':
        return accuracy_score(y_test, y_pred)
    elif output == 'pred':
        return y_pred


def confusion_matrix(y_pred, output='plot'):
    '''
    Function that takes in predictions and outputs confusion matrix (plot or raw matrix)
    :param y_pred: predictions
    :param output: what to output
    :return: plot if output == 'plot', matrix if output == 'matrix'
    '''

    data_dir = input('Please enter the directory where "partition", "labels" and "y_test" are stored')
    # D:/Rijksmuseum/
    os.chdir(data_dir)

    y_test = np.load('y_test.npy')

    with open('artists', 'rb') as pickle_in:
        artists = pickle.load(pickle_in)

    artist_list = [item[1][:item[1].find(',') + 3] if len(item[1].split(',')) == 2 else item[1] for item in
                   artists.items()]

    import seaborn as sn
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # calculate confusion matrix
    c_m = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    if output == 'matrix_absolute':
        return c_m
    c_m = c_m.astype('float') / c_m.sum(axis=1)[:, np.newaxis]
    # plot confusion matrix
    if output == 'plot':
        import matplotlib
        import matplotlib.colors as mc
        # set right colors
        flatui = ["#00008f", "#004fff", "#2fffcf", "#ffffcf", "#ff2f00", "#7f0000"]
        sn.set_palette(flatui)
        def NonLinCdict(steps, hexcol_array):
            cdict = {'red': (), 'green': (), 'blue': ()}
            for s, hexcol in zip(steps, hexcol_array):
                rgb = matplotlib.colors.hex2color(hexcol)
                cdict['red'] = cdict['red'] + ((s, rgb[0], rgb[0]),)
                cdict['green'] = cdict['green'] + ((s, rgb[1], rgb[1]),)
                cdict['blue'] = cdict['blue'] + ((s, rgb[2], rgb[2]),)
            return cdict

        hc = ["#00008f", "#004fff", "#2fffcf", '#8fff6f', "#ffff00", "#ff2f00", "#7f0000"]
        th = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]

        cdict = NonLinCdict(th, hc)
        cm = mc.LinearSegmentedColormap('test', cdict)

        ax = sn.heatmap(c_m, annot=False, xticklabels=artist_list, yticklabels=artist_list, cmap=cm)

        plt.show()
        return ax
    elif output == 'matrix':
        return c_m
    else:
        print('not a valid output type')
        return


# ----------------------------------------------------------------------------------------------------------------------
# ACCURACIES
# ----------------------------------------------------------------------------------------------------------------------
print('acc', get_results(method='Bagging', aggregation='majority', output='acc'))
print('acc', get_results(method='Bagging', aggregation='mean', output='acc'))
print('acc', get_results(method='Bagging', aggregation='trimmed_mean', output='acc'))
print('acc', get_results(method='Bagging', aggregation='weighted_ranked', output='acc'))
print('acc', get_results(method='Bagging', aggregation='weighted_absolute', output='acc'))
print('acc SAMME', get_results(method='SAMME', output='acc'))
print('acc CONFADABOOST', get_results(method='CONFADABOOST', output='acc'))
print('acc DisturbLabel', get_results(method='Disturblabel', output='acc'))
print('acc DisturbLabel', get_results(method='Disturblabel', output='acc'))
# ----------------------------------------------------------------------------------------------------------------------
# CONFUSION MATRICES
# ----------------------------------------------------------------------------------------------------------------------
confusion_matrix(get_results(method='Bagging', aggregation='majority', output='pred'), output='plot')
confusion_matrix(get_results(method='Bagging', aggregation='mean', output='pred'), output='plot')
confusion_matrix(get_results(method='Bagging', aggregation='trimmed_mean', output='pred'), output='plot')
confusion_matrix(get_results(method='Bagging', aggregation='weighted_ranked', output='pred'), output='plot')
confusion_matrix(get_results(method='Bagging', aggregation='weighted_absolute', output='pred'), output='plot')
confusion_matrix(get_results(method='SAMME', output='pred'), output='plot')
confusion_matrix(get_results(method='CONFADABOOST', output='pred'), output='plot')
confusion_matrix(get_results(method='Disturblabel', output='pred'), output='plot')
confusion_matrix(get_results(method='Disturblabel', output='pred'), output='plot')

# ----------------------------------------------------------------------------------------------------------------------
# BOOSTING CLOSER LOOK: TABLE 3
# --------------------------------------------------------------------------------------------------------------------
# get results and confusion matrices
results_disturb = get_results(method='Disturblabel', output='pred')
c_m_disturb = confusion_matrix(results_disturb, output='matrix_absolute')

results_samme = get_results(method='SAMME', output='pred')
c_m_samme = confusion_matrix(results_samme, output='matrix_absolute')

# create a SAMME confusion matrix for Jan en Caspar Luyken
matrix_samme = np.empty((2,2), dtype=int)
matrix_samme[0,0] = 1
matrix_samme[0,0] = c_m_samme[21,21]
matrix_samme[1,1] = c_m_samme[22, 22]
matrix_samme[0,1] = c_m_samme[21, 22]
matrix_samme[1,0] = c_m_samme[22, 21]
# create a Disturblabel confusion matrix for Jan en Caspar Luyken
matrix_disturb = np.empty((2,2), dtype=int)
matrix_disturb[0,0] = c_m_disturb[21,21]
matrix_disturb[1,1] = c_m_disturb[22, 22]
matrix_disturb[0,1] = c_m_disturb[21, 22]
matrix_disturb[1,0] = c_m_disturb[22, 21]
# calculate precision & Recall
samme_recall = matrix_samme[0,0] / (matrix_samme[0,0] + matrix_samme[0,1])
samme_precision = matrix_samme[0,0] / (matrix_samme[0,0] + matrix_samme[1,0])
disturb_recall = matrix_disturb[0,0] / (matrix_disturb[0,0] + matrix_disturb[0,1])
disturb_precision = matrix_disturb[0,0] / (matrix_disturb[0,0] + matrix_disturb[1,0])
# calculate F1
f1_samme = 2 * (samme_recall * samme_precision) / (samme_recall + samme_precision)
f1_disturb = 2 * (disturb_recall * disturb_precision) / (disturb_recall + disturb_precision)
# print results
print(f1_samme)
print(f1_disturb)
print()
print(samme_precision)
print(samme_recall)
print()
print(disturb_precision)
print(disturb_recall)

# ----------------------------------------------------------------------------------------------------------------------
# PLOT FIGURE 2: EFFECT OF ENSEMBLE COMPOSITION ON MCA FOR WEIGHTED AGGREGATION METHODS
# ----------------------------------------------------------------------------------------------------------------------
rank_acc = []
absolute_acc = []

# get results for different ensemble sizes
for i in range(10):
    acc = get_results(method='weighted_rank', output='acc', drop=i)
    rank_acc.append(acc)
    acc = get_results(method='weighted_absolute', output='acc', drop=i)
    absolute_acc.append(acc)
# change dir so we can....
data_dir = input('Please enter the directory where "partition", "labels" and "y_test" are stored')
# D:/Rijksmuseum/
os.chdir(data_dir)
# ... load y_test
y_test = np.load('y_test.npy')

# plot the results
x = [i for i in range(10, 0, -1)]
plt.plot(x, rank_acc, color='blue', linewidth=2, label='ranked weights')
plt.plot(x, absolute_acc, color='red', linewidth=2, label='absolute weights')
plt.legend()
plt.xlabel('ensemble size')
plt.ylabel('MCA')
plt.show()

# --------------------------------------------------------------------------------------------------------------------
# ABSOLUTE & RANKED WEIGHT STANDARD DEVIATION
# --------------------------------------------------------------------------------------------------------------------
w1 = [0.7, 0.2, 0.6, 0.1, 0.5, 1.0, 0.3, 0.9, 0.4, 0.8]
w2 = [0.7635467980295566, 0.7323481116584565, 0.7606732348111659, 0.7241379310344828, 0.7504105090311987, 0.7807881773399015, 0.7348111658456487, 0.7713464696223317, 0.7471264367816092, 0.7655993431855501]
w1_std = np.std(w1)
w2_std = np.std(w2)
# --------------------------------------------------------------------------------------------------------------------
# RUN MAJ ACC 1000 TIMES TO CONTROL FOR STOCHASTICITY IN AGGREGATION PROCESS
# --------------------------------------------------------------------------------------------------------------------
maj_acc = []
for i in range(1000):
    maj_acc.append(get_results(method='Bagging', aggregation='majority', output='acc'))

maj_min = np.min(maj_acc)
maj_max = np.max(maj_acc)
maj_mean = np.mean(maj_acc)
maj_std = np.std(maj_acc)
# # --------------------------------------------------------------------------------------------------------------------
