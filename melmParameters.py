"""
Code developed by:

Prof. Dr. Sidney Marlon Lopes de Lima
Federal University of Pernambuco
Department of Electronics and Systems
"""

from math import *
from random import seed as rnd_seed
from time import process_time
from time import perf_counter as _timer

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import math
import time
import struct
import sys, string
import argparse
import numpy as np
import pandas as pd

# === extras para o relatório ===
from jinja2 import Template
import os
import matplotlib.pyplot as plt
import seaborn as sns

#========================================================================
class melm():
    def main(self, TrainingData_File, TestingData_File, AllData_File,
             Elm_Type, NumberofHiddenNeurons, ActivationFunction,
             nSeed, kfold, virusNorm, sep, verbose):

        # -------------------------
        # parsing de listas p/ busca
        # -------------------------
        acts = [s.strip() for s in str(ActivationFunction or 'linear').split(',') if s.strip()]
        nh_list = [int(v.strip()) for v in str(NumberofHiddenNeurons).split(',') if str(v).strip()]

        if verbose: print('Ativações:', acts)
        if verbose: print('nHidden:', nh_list)

        if nSeed is None:
            nSeed = 1
        else:
            if verbose: print('seed', nSeed)
            nSeed = int(nSeed)
        rnd_seed(nSeed)
        np.random.seed(nSeed)

        Elm_Type = int(Elm_Type)
        self.REGRESSION = 0
        self.CLASSIFIER = 1

        # ============================================================
        # Caminho 1: arquivos separados (treino/teste) => sem busca/KFold
        # ============================================================
        if AllData_File is None:
            sep_character = sep if sep else " "
            if verbose: print('Loading training dataset')
            train_data = pd.read_csv(TrainingData_File, sep=sep_character, decimal=".", header=None)
            train_data = eliminateNaN_All_data(train_data)
            if verbose: print('Loading testing dataset')
            test_data = pd.read_csv(TestingData_File, sep=sep_character, decimal=".", header=None)
            test_data = eliminateNaN_All_data(test_data)

            # usa a primeira combinação apenas (compatível com comportamento antigo)
            af = acts[0]
            nh = nh_list[0]

            TrainingAccuracy, TestingAccuracy, TrainingTime, TestingTime, _ = mElmLearning(
                train_data, test_data, 1, Elm_Type, nh, af, nSeed, False, virusNorm, verbose
            )

            print('>>> Execução única (sem KFold/busca):')
            if Elm_Type == 0:
                print(f"Train RMSE: {TrainingAccuracy}")
                print(f"Test  RMSE: {TestingAccuracy}")
            else:
                print(f"Train Acc: {TrainingAccuracy*100:.2f}%")
                print(f"Test  Acc: {TestingAccuracy*100:.2f}%")
            print(f"Train Time (s): {TrainingTime:.2f}")
            print(f"Test  Time (s): {TestingTime:.2f}")
            return

        # ============================================================
        # Caminho 2: AllData_File + KFold => busca sobre (acts x nh_list)
        # ============================================================
        if verbose: print('Loading all dataset')
        all_data, samples_index = mElmStruct(AllData_File, Elm_Type, sep, verbose)

        if not kfold:
            raise ValueError("Para a busca e relatório HTML, use -kfold <int> com -tall.")

        if verbose: print('K-fold search over ActivationFunction x nHidden')
        kf = KFold(n_splits=int(kfold), shuffle=True, random_state=nSeed)

        # agrega resultados por combinação
        combo_results = []  # cada item: dict com métricas médias/DP, MC média e tempos ±DP
        for af in acts:
            for nh in nh_list:
                if verbose: print(f'\n=== Avaliando combinação: act={af}, n_hidden={nh} ===')

                acc_train, acc_test = [], []
                t_train, t_test = [], []
                cms = []
                label_count = None

                # KFold sobre índices reais (garante embaralhamento correto)
                for tr_idx, te_idx in kf.split(samples_index):
                    tr_ids = samples_index[tr_idx]
                    te_ids = samples_index[te_idx]

                    train_data = all_data[tr_ids, :]
                    test_data  = all_data[te_ids, :]

                    TA, TeA, TT, Tt, cm = mElmLearning(
                        train_data, test_data, len(acc_train),
                        Elm_Type, nh, af, nSeed, kfold, virusNorm, verbose
                    )
                    acc_train.append(TA)
                    acc_test.append(TeA)
                    t_train.append(TT)
                    t_test.append(Tt)

                    if cm is not None:
                        cms.append(cm.astype(float))
                        if label_count is None:
                            label_count = cm.shape[0]

                # médias/DP
                mean_tr = np.mean(acc_train)
                std_tr  = np.std(acc_train)
                mean_te = np.mean(acc_test)
                std_te  = np.std(acc_test)
                mean_tt = float(np.mean(t_train))
                std_tt  = float(np.std(t_train))
                mean_et = float(np.mean(t_test))
                std_et  = float(np.std(t_test))

                avg_cm = np.mean(cms, axis=0) if cms else None

                # para classificação, reportar em %
                if Elm_Type == self.CLASSIFIER:
                    mean_tr *= 100.0
                    std_tr  *= 100.0
                    mean_te *= 100.0
                    std_te  *= 100.0

                combo_results.append(dict(
                    act=af, n_hidden=int(nh),
                    mean_acc_tr=mean_tr, std_acc_tr=std_tr,
                    mean_acc_te=mean_te, std_acc_te=std_te,
                    mean_tt=mean_tt, std_tt=std_tt,
                    mean_et=mean_et, std_et=std_et,
                    cm=avg_cm
                ))

                if verbose:
                    print(f"→ Train: {mean_tr:.2f} ± {std_tr:.2f} | Test: {mean_te:.2f} ± {std_te:.2f}")
                    print(f"   Train time (s): {mean_tt:.2f} ± {std_tt:.2f} | Test time (s): {mean_et:.2f} ± {std_et:.2f}")

        # escolher melhor/pior por acurácia de teste
        best = max(combo_results, key=lambda r: r['mean_acc_te'])
        worst = min(combo_results, key=lambda r: r['mean_acc_te'])

        # montar dicionário para HTML (compatível com template do SVM, porém com rótulos ELM)
        global_results = {
            "max": {
                "kernel_name": best['act'],  # usamos o campo do template para mostrar a ativação
                "accuracy_train": best['mean_acc_tr'], "std_train": best['std_acc_tr'],
                "accuracy_test":  best['mean_acc_te'], "std_test":  best['std_acc_te'],
                "kernel_id": 0, "cost": best['n_hidden'], "gamma": 0.0,
                "confusion_matrix": best['cm'],
                "train_time_mean": best['mean_tt'], "train_time_std": best['std_tt'],
                "test_time_mean":  best['mean_et'], "test_time_std":  best['std_et'],
            },
            "min": {
                "kernel_name": worst['act'],
                "accuracy_train": worst['mean_acc_tr'], "std_train": worst['std_acc_tr'],
                "accuracy_test":  worst['mean_acc_te'], "std_test":  worst['std_acc_te'],
                "kernel_id": 0, "cost": worst['n_hidden'], "gamma": 0.0,
                "confusion_matrix": worst['cm'],
                "train_time_mean": worst['mean_tt'], "train_time_std": worst['std_tt'],
                "test_time_mean":  worst['mean_et'], "test_time_std":  worst['std_et'],
            }
        }

        print('\n==========================================')
        print('RESULTADOS GLOBAIS (ELM)')
        print('==========================================')
        print(f"Melhor: act={best['act']}, n_hidden={best['n_hidden']}")
        print(f"  Train: {best['mean_acc_tr']:.2f} ± {best['std_acc_tr']:.2f} | Test: {best['mean_acc_te']:.2f} ± {best['std_acc_te']:.2f}")
        print(f"  Train time (s): {best['mean_tt']:.2f} ± {best['std_tt']:.2f} | Test time (s): {best['mean_et']:.2f} ± {best['std_et']:.2f}")
        print(f"Pior:   act={worst['act']}, n_hidden={worst['n_hidden']}")
        print(f"  Train: {worst['mean_acc_tr']:.2f} ± {worst['std_acc_tr']:.2f} | Test: {worst['mean_acc_te']:.2f} ± {worst['std_acc_te']:.2f}")
        print(f"  Train time (s): {worst['mean_tt']:.2f} ± {worst['std_tt']:.2f} | Test time (s): {worst['mean_et']:.2f} ± {worst['std_et']:.2f}")

        # gerar relatório HTML (apenas blocos Melhor/Pior)
        generate_html_report_elm(global_results, output_file='elm_report.html')

#========================================================================
def eliminateNaN(vector):
    first_row = vector.tolist()
    first_row = [elem for elem in first_row if not pd.isna(elem)]
    return first_row

#========================================================================
def eliminateNaN_All_data(all_data):
    all_data = all_data[:].to_numpy()
    all_data = all_data.astype(float)
    for ii in reversed(range(np.size(all_data,1))):
        if np.all(np.isnan(all_data[:,ii])):
            all_data = np.delete(all_data, ii, axis=1)
    return all_data

#========================================================================
def mElmStruct(AllData_File, Elm_Type, sep, verbose):
    sep_character = sep if sep else ';'
    df = pd.read_csv(AllData_File, sep=sep_character, decimal=".", low_memory=False, header=None)

    # primeira coluna = nomes/ids (ignorada); primeira linha = cabeçalho de features (ignorada)
    df_vals = df.loc[1:np.size(df,0), 1:np.size(df,1)]
    all_data = eliminateNaN_All_data(df_vals)

    # classificação: embaralhar; regressão: manter ordem
    if int(Elm_Type) != 0:
        if verbose: print('Permutation of the order of the input data')
        samples_index = np.random.permutation(np.size(all_data,0))
    else:
        samples_index = np.arange(0, np.size(all_data,0))
    return all_data, samples_index

#========================================================================
def loadingDataset(dataset):
    T = np.transpose(dataset[:,0])
    P = np.transpose(dataset[:,1:np.size(dataset,1)])
    del(dataset)
    return T, P

#========================================================================
def mElmLearning(train_data, test_data, execution, Elm_Type,
                 NumberofHiddenNeurons, ActivationFunction,
                 nSeed, kfold, virusNorm, verbose):

    [T, P] = loadingDataset(train_data)
    [TVT, TVP] = loadingDataset(test_data)

    NumberofTrainingData = np.size(P,1)
    NumberofTestingData  = np.size(TVP,1)
    NumberofInputNeurons = np.size(P,0)
    NumberofHiddenNeurons = int(NumberofHiddenNeurons)

    cm_fold = None  # retorno da matriz de confusão (classificação)

    if Elm_Type != 0:  # classification
        if verbose: print('Preprocessing the data of classification')
        sorted_target = np.sort(np.concatenate((T, TVT), axis=0))
        label = [sorted_target[0]]
        j = 0
        for i in range(1, NumberofTrainingData+NumberofTestingData):
            if sorted_target[i] != label[j]:
                j += 1
                label.append(sorted_target[i])
        number_class = j + 1
        NumberofOutputNeurons = number_class

        if verbose: print('Processing the targets of training')
        temp_T = np.zeros((NumberofOutputNeurons, NumberofTrainingData))
        for i in range(0, NumberofTrainingData):
            for j in range(0, number_class):
                if label[j] == T[i]:
                    break
            temp_T[j][i] = 1
        T = temp_T*2 - 1

        if verbose: print('Processing the targets of testing')
        temp_TV_T = np.zeros((NumberofOutputNeurons, NumberofTestingData))
        for i in range(0, NumberofTestingData):
            for j in range(0, number_class):
                if label[j] == TVT[i]:
                    break
            temp_TV_T[j][i] = 1
        TVT = temp_TV_T*2 - 1

    if verbose: print('Calculate weights & biases')
    start_time_train = process_time()

    if verbose: print('Random generate input weights and biases')
    if Elm_Type == 0:  # Regression
        if ActivationFunction in ('erosion','ero','dilation','dil','fuzzy-erosion','fuzzy_erosion',
                                  'fuzzy-dilation','fuzzy_dilation','bitwise-erosion','bitwise_erosion',
                                  'bitwise-dilation','bitwise_dilation'):
            InputWeight = np.random.uniform(np.amin(np.amin(P)), np.amax(np.amax(P)),
                                            (NumberofHiddenNeurons, NumberofInputNeurons))
        else:
            InputWeight = np.random.rand(NumberofHiddenNeurons, NumberofInputNeurons)*2 - 1
    else:
        InputWeight = np.random.rand(NumberofHiddenNeurons, NumberofInputNeurons)*2 - 1

    if virusNorm:
        InputWeight = virusNormFunction(InputWeight, verbose)

    BiasofHiddenNeurons = np.random.rand(NumberofHiddenNeurons, 1)
    if verbose: print('Calculate hidden neuron output matrix H')
    H = switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, P)

    if verbose: print('Calculate output weights (Beta)')
    OutputWeight = np.dot(np.linalg.pinv(np.transpose(H)), np.transpose(T))

    end_time_train = process_time()
    TrainingTime = end_time_train - start_time_train

    if verbose: print('Calculate the training accuracy')
    Y = np.transpose(np.dot(np.transpose(H), OutputWeight))  # (c, n_tr) ou (1, n_tr)

    TrainingAccuracy = 0
    if Elm_Type == 0:
        # RMSE (regressão)
        TrainingAccuracy = np.square(np.subtract(T, Y)).mean()
        TrainingAccuracy = round(TrainingAccuracy, 6)
    del(H)

    if verbose: print('Calculate the output of testing input')
    start_time_test = process_time()
    tempH_test = switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, TVP)
    del(TVP)
    TY = np.transpose(np.dot(np.transpose(tempH_test), OutputWeight))
    end_time_test = process_time()
    TestingTime = end_time_test - start_time_test

    TestingAccuracy = 0
    if Elm_Type == 0:
        TestingAccuracy = np.square(np.subtract(TVT, TY)).mean()
        TestingAccuracy = round(TestingAccuracy, 6)
    else:
        if verbose: print('Calculate training & testing classification accuracy')
        MissClassificationRate_Training = 0
        MissClassificationRate_Testing  = 0

        label_index_train_expected = np.argmax(T, axis=0)
        label_index_train_actual   = np.argmax(Y, axis=0)
        for i in range(0, np.size(label_index_train_expected,0)):
            if label_index_train_actual[i] != label_index_train_expected[i]:
                MissClassificationRate_Training += 1
        TrainingAccuracy = 1 - MissClassificationRate_Training/np.size(label_index_train_expected,0)
        TrainingAccuracy = round(TrainingAccuracy, 6)

        label_index_expected = np.argmax(TVT, axis=0)
        label_index_actual   = np.argmax(TY, axis=0)
        for i in range(0, np.size(label_index_expected,0)):
            if label_index_actual[i] != label_index_expected[i]:
                MissClassificationRate_Testing += 1
        TestingAccuracy = 1 - MissClassificationRate_Testing/np.size(label_index_expected,0)
        TestingAccuracy = round(TestingAccuracy, 6)

        # matriz de confusão deste fold (ordem 0..number_class-1)
        cm_fold = confusion_matrix(label_index_expected, label_index_actual, labels=list(range(number_class)))

    # prints por fold (opcional)
    if kfold:
        print(f'..................k: {execution}, k-fold: {kfold}............................')
    else:
        print('....................................................................')

    if Elm_Type == 0:
        print(f'Training RMSE: {TrainingAccuracy} ( {np.size(Y,0)} samples)')
        print(f'Testing  RMSE: {TestingAccuracy} ( {TY.shape[1]} samples)')
    else:
        print(f'Training Accuracy: {TrainingAccuracy*100:.2f}%')
        print(f'Testing  Accuracy: {TestingAccuracy*100:.2f}%')

    print(f'Training Time: {round(TrainingTime,2)} sec.')
    print(f'Testing  Time: {round(TestingTime,2)} sec.')

    return TrainingAccuracy, TestingAccuracy, TrainingTime, TestingTime, cm_fold

#========================================================================
def virusNormFunction(matrix, verbose):
    if verbose: print('virusNorm normalization')
    vector = matrix.flatten()
    maxi = np.max(vector); mini = np.min(vector)
    ra = 0.9; rb = 0.1
    R = (((ra - rb) * (matrix - mini)) / (maxi - mini)) + rb
    return R

#========================================================================
def switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, P):
    if ActivationFunction in ('sig', 'sigmoid'):
        return sig_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('sin', 'sine'):
        return sin_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction == 'hardlim':
        return hardlim_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction == 'tribas':
        return tribas_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction == 'radbas':
        return radbas_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('erosion','ero'):
        return erosion(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('dilation','dil'):
        return dilation(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('fuzzy-erosion','fuzzy_erosion'):
        return fuzzy_erosion(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('fuzzy-dilation','fuzzy_dilation'):
        return fuzzy_dilation(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('bitwise-erosion','bitwise_erosion'):
        return bitwise_erosion(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('bitwise-dilation','bitwise_dilation'):
        return bitwise_dilation(InputWeight, BiasofHiddenNeurons, P)
    else:  # 'linear'
        return linear_kernel(InputWeight, BiasofHiddenNeurons, P)

#========================================================================
# --- SIGMÓIDE SEGURA (clipping) ---
def sig_kernel(w1, b1, samples):
    tempH = np.dot(w1, samples) + b1
    tempH = np.clip(tempH, -40, 40)  # evita overflow no exp
    return 1.0 / (1.0 + np.exp(-tempH))

#========================================================================
def sin_kernel(w1, b1, samples):
    tempH = np.dot(w1, samples) + b1
    return np.sin(tempH)

#========================================================================
def hardlim_kernel(w1, b1, samples):
    tempH = np.dot(w1, samples) + b1
    H = (tempH >= 0).astype(float)
    return H

#========================================================================
def tribas_kernel(w1, b1, samples):
    tempH = np.dot(w1, samples) + b1
    H = 1 - np.abs(tempH)
    H[(tempH < -1) | (tempH > 1)] = 0
    return H

#========================================================================
def radbas_kernel(w1, b1, samples):
    tempH = np.dot(w1, samples) + b1
    return np.exp(-np.power(tempH, 2))

#========================================================================
def linear_kernel(w1, b1, samples):
    return np.dot(w1, samples) + b1

#========================================================================
def erosion(w1, b1, samples):
    H = np.zeros((np.size(w1,0), np.size(samples,1)))
    x = np.zeros(np.size(w1,1))
    for s_index in range(np.size(samples,1)):
        ss = samples[:,s_index]
        for i in range(np.size(w1,0)):
            for j in range(np.size(w1,1)):
                x[j] = max(ss[j], 1-w1[i][j])
            H[i][s_index] = min(x)+b1[i][0]
    return H

#========================================================================
def dilation(w1, b1, samples):
    H = np.zeros((np.size(w1,0), np.size(samples,1)))
    x = np.zeros(np.size(w1,1))
    for s_index in range(np.size(samples,1)):
        ss = samples[:,s_index]
        for i in range(np.size(w1,0)):
            for j in range(np.size(w1,1)):
                x[j] = min(ss[j], w1[i][j])
            H[i][s_index] = max(x)+b1[i][0]
    return H

#========================================================================
def fuzzy_erosion(w1, b1, samples):
    tempH = np.dot(w1, samples) + b1
    H = 1.0 - tempH
    return H

#========================================================================
def fuzzy_dilation(w1, b1, samples):
    tempH = np.dot(w1, samples) + b1
    H = np.ones((np.size(w1,0), np.size(samples,1)))
    for s_index in range(np.size(samples,1)):
        for i in range(np.size(w1,0)):
            for j in range(np.size(w1,1)):
                H[i][s_index] *= (1 - tempH[i][j])
    H = 1 - H
    return H

#========================================================================
def bitwise_erosion(w1, b1, samples):
    H = np.zeros((np.size(w1,0), np.size(samples,1)))
    x = np.zeros(np.size(w1,1), dtype=bytearray)
    for s_index in range(np.size(samples,1)):
        ss = samples[:,s_index]
        for i in range(np.size(w1,0)):
            for j in range(np.size(w1,1)):
                x[j] = bytes_or(ss[j], 1-w1[i][j])
            result = x[0]
            for j in range(1, np.size(w1,1)):
                result = bytes_and(result, x[j])
            temp = struct.unpack('d', result)[0]
            if math.isnan(temp): temp = 0.0
            H[i][s_index] = temp + b1[i][0]
    return H

#========================================================================
def bitwise_dilation(w1, b1, samples):
    H = np.zeros((np.size(w1,0), np.size(samples,1)))
    x = np.zeros(np.size(w1,1), dtype=bytearray)
    for s_index in range(np.size(samples,1)):
        ss = samples[:,s_index]
        for i in range(np.size(w1,0)):
            for j in range(np.size(w1,1)):
                x[j] = bytes_and(ss[j], w1[i][j])
            result = x[0]
            for j in range(1, np.size(w1,1)):
                result = bytes_or(result, x[j])
            temp = struct.unpack('d', result)[0]
            if math.isnan(temp): temp = 0.0
            H[i][s_index] = temp + b1[i][0]
    return H

#========================================================================
def bytes_and(a, b):
    a1 = bytearray(a); b1 = bytearray(b)
    c = bytearray(len(a1))
    for i in range(len(a1)):
        c[i] = a1[i] & b1[i]
    return c

#========================================================================
def bytes_or(a, b):
    a1 = bytearray(a); b1 = bytearray(b)
    c = bytearray(len(a1))
    for i in range(len(a1)):
        c[i] = a1[i] | b1[i]
    return c

#========================================================================
# Relatório HTML (apenas blocos Melhor/Pior)
def plot_and_save_cm(cm, title, filename):
    if cm is None:
        return
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=plt.cm.Blues)
    plt.title(title)
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_html_report_elm(global_results, output_file='elm_report.html'):
    img_dir = 'elm_report_images'
    os.makedirs(img_dir, exist_ok=True)

    plot_and_save_cm(global_results['max']['confusion_matrix'], 'MC Média - Melhor Desempenho Global', f'{img_dir}/cm_global_best.png')
    plot_and_save_cm(global_results['min']['confusion_matrix'], 'MC Média - Pior Desempenho Global',   f'{img_dir}/cm_global_worst.png')

    html_template = """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <title>Relatório de Resultados ELM</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #fdfdfd; }
            .container { max-width: 1000px; margin: 0 auto; background-color: #fff; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .metrics-table { width: 100%; border-collapse: collapse; margin: 25px 0; }
            .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: middle; }
            .metrics-table th { background-color: #0056b3; color: white; text-align: center; }
            .result-section { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; border-left: 5px solid #0056b3; }
            h1, h2 { color: #333; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }
            .best-result { background-color: #e7f3e7; }
            .worst-result { background-color: #fdeeee; }
            .cm-image { max-width: 350px; display: block; margin: 10px auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Relatório de Avaliação de Parâmetros ELM</h1>
            <div class="result-section">
                <h2>Resultados Globais</h2>
                <p>Melhor e pior desempenho considerando todas as combinações de ativação e n_hidden.</p>
                <table class="metrics-table">
                    <tr class="best-result">
                        <td colspan="2" style="text-align:center; font-weight:bold;">Melhor Desempenho Geral</td>
                    </tr>
                    <tr class="best-result"><td>Ativação</td><td>{{ global_results.max.kernel_name }}</td></tr>
                    <tr class="best-result"><td>Acurácia Média de Treino</td><td>{{ "%.2f"|format(global_results.max.accuracy_train) }}% &plusmn; {{ "%.2f"|format(global_results.max.std_train) }}%</td></tr>
                    <tr class="best-result"><td>Acurácia Média de Teste</td><td>{{ "%.2f"|format(global_results.max.accuracy_test) }}% &plusmn; {{ "%.2f"|format(global_results.max.std_test) }}%</td></tr>
                    <tr class="best-result"><td>Configuração (n_hidden)</td><td>{{ "%.0f"|format(global_results.max.cost) }}</td></tr>
                    <tr class="best-result"><td>Train time (seconds)</td><td>{{ "%.2f"|format(global_results.max.train_time_mean) }} &plusmn; {{ "%.2f"|format(global_results.max.train_time_std) }}</td></tr>
                    <tr class="best-result"><td>Test time (seconds)</td><td>{{ "%.2f"|format(global_results.max.test_time_mean) }} &plusmn; {{ "%.2f"|format(global_results.max.test_time_std) }}</td></tr>
                    <tr class="best-result"><td colspan="2"><img class="cm-image" src="elm_report_images/cm_global_best.png" alt="Matriz de Confusão - Melhor Global"></td></tr>

                    <tr class="worst-result">
                        <td colspan="2" style="text-align:center; font-weight:bold;">Pior Desempenho Geral</td>
                    </tr>
                    <tr class="worst-result"><td>Ativação</td><td>{{ global_results.min.kernel_name }}</td></tr>
                    <tr class="worst-result"><td>Acurácia Média de Treino</td><td>{{ "%.2f"|format(global_results.min.accuracy_train) }}% &plusmn; {{ "%.2f"|format(global_results.min.std_train) }}%</td></tr>
                    <tr class="worst-result"><td>Acurácia Média de Teste</td><td>{{ "%.2f"|format(global_results.min.accuracy_test) }}% &plusmn; {{ "%.2f"|format(global_results.min.std_test) }}%</td></tr>
                    <tr class="worst-result"><td>Configuração (n_hidden)</td><td>{{ "%.0f"|format(global_results.min.cost) }}</td></tr>
                    <tr class="worst-result"><td>Train time (seconds)</td><td>{{ "%.2f"|format(global_results.min.train_time_mean) }} &plusmn; {{ "%.2f"|format(global_results.min.train_time_std) }}</td></tr>
                    <tr class="worst-result"><td>Test time (seconds)</td><td>{{ "%.2f"|format(global_results.min.test_time_mean) }} &plusmn; {{ "%.2f"|format(global_results.min.test_time_std) }}</td></tr>
                    <tr class="worst-result"><td colspan="2"><img class="cm-image" src="elm_report_images/cm_global_worst.png" alt="Matriz de Confusão - Pior Global"></td></tr>
                </table>
            </div>
        </div>
    </body>
    </html>
    """

    template = Template(html_template)
    html_content = template.render(global_results=global_results)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\nRelatório HTML gerado com sucesso: '{output_file}'")
    except IOError as e:
        print(f"\nErro ao salvar o relatório HTML: {e}")

#========================================================================
def setOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--TrainingData_File',dest='TrainingData_File',action='store', 
        help="Filename of training data set")
    parser.add_argument('-ts', '--TestingData_File',dest='TestingData_File',action='store',
        help="Filename of testing data set")
    parser.add_argument('-tall', '--AllData_File',dest='AllData_File',action='store',
        help="Filename of all data set, including training and testing")
    parser.add_argument('-ty', '--Elm_Type',dest='Elm_Type',action='store',required=True,
        help="0 for regression; 1 for (both binary and multi-classes) classification")
    parser.add_argument('-nh', '--nHiddenNeurons',dest='nHiddenNeurons',action='store',required=True,
        help="Number of hidden neurons (ou lista separada por vírgulas)")
    parser.add_argument('-af', '--ActivationFunction',dest='ActivationFunction',action='store', 
        help="Type of activation function (ou lista separada por vírgulas)")
    parser.add_argument('-sd', '--seed',dest='nSeed',action='store', 
        help="random number generator seed:")
    parser.add_argument('-kfold', dest='kfold', action='store', default=False,
        help="K-fold validation. Use um inteiro para habilitar busca/relatório.")
    parser.add_argument('-virusNorm', dest='virusNorm', action='store_true', default=False,
        help="Normalization according to the range of VirusShare sample attributes.")  
    parser.add_argument('-sep', dest='sep', action='store_true', default=False,
        help="Character or regex pattern to treat as the delimiter. Default: space (TR/TS) e ';' (ALL).")          
    parser.add_argument('-v', dest='verbose', action='store_true', default=False,
        help="Verbose output")
    arg = parser.parse_args()
    return (arg.__dict__['TrainingData_File'], arg.__dict__['TestingData_File'],
            arg.__dict__['AllData_File'], arg.__dict__['Elm_Type'], arg.__dict__['nHiddenNeurons'],
            arg.__dict__['ActivationFunction'], arg.__dict__['nSeed'], arg.__dict__['kfold'],
            arg.__dict__['virusNorm'], arg.__dict__['sep'], arg.__dict__['verbose'])

#========================================================================
if __name__ == "__main__":
    opts = setOpts(sys.argv[1:])
    ff = melm()
    ff.main(opts[0], opts[1], opts[2], opts[3], opts[4], opts[5], opts[6], opts[7], opts[8], opts[9], opts[10])
#========================================================================
