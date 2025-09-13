# -*- coding: utf-8 -*-
"""
This script performs ELM parameter evaluation for different activation functions
and generates a "Dashboard" style HTML report, with global and per-activation
results, including the average confusion matrices for each scenario.
"""

# Dependencies
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from jinja2 import Template
import numpy as np
import pandas as pd
import argparse
import sys
from time import process_time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
import struct
from random import seed as rnd_seed

#========================================================================
# PLOTTING AND HELPER FUNCTIONS
#========================================================================

def plot_and_save_cm(cm, title, filename):
    """Generates a confusion matrix plot in percentages and saves it as an image."""
    if cm is None or cm.sum() == 0:
        return
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_percent = cm.astype('float') / cm_sum * 100
    cm_percent = np.nan_to_num(cm_percent)
    plt.figure(figsize=(6, 5))
    annot_labels = np.array([[f'{val:.1f}%' for val in row] for row in cm_percent])
    sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap=plt.cm.Blues, cbar_kws={'label': 'Percentage (%)'})
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def eliminateNaN_All_data(all_data):
    all_data = all_data[:].to_numpy().astype(float)
    for ii in reversed(range(np.size(all_data,1))):
        if np.all(np.isnan(all_data[:,ii])):
            all_data = np.delete(all_data, ii, axis=1)
    return all_data

def mElmStruct(AllData_File, Elm_Type, sep, verbose):
    sep_character = sep if sep else ';'
    df = pd.read_csv(AllData_File, sep=sep_character, decimal=".", low_memory=False, header=None)
    df_vals = df.loc[1:np.size(df,0), 1:np.size(df,1)]
    all_data = eliminateNaN_All_data(df_vals)
    if int(Elm_Type) != 0:
        if verbose: print('Permutation of the order of the input data')
        samples_index = np.random.permutation(np.size(all_data,0))
    else:
        samples_index = np.arange(0, np.size(all_data,0))
    return all_data, samples_index

def loadingDataset(dataset):
    T = np.transpose(dataset[:,0])
    P = np.transpose(dataset[:,1:np.size(dataset,1)])
    return T, P

def switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, P):
    if ActivationFunction in ('sig', 'sigmoid'): return sig_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('sin', 'sine'): return sin_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction == 'hardlim': return hardlim_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction == 'tribas': return tribas_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction == 'radbas': return radbas_kernel(InputWeight, BiasofHiddenNeurons, P)
    else: return linear_kernel(InputWeight, BiasofHiddenNeurons, P)
def sig_kernel(w1, b1, samples):
    tempH = np.dot(w1, samples) + b1; tempH = np.clip(tempH, -40, 40)
    return 1.0 / (1.0 + np.exp(-tempH))
def sin_kernel(w1, b1, samples): return np.sin(np.dot(w1, samples) + b1)
def hardlim_kernel(w1, b1, samples): return (np.dot(w1, samples) + b1 >= 0).astype(float)
def linear_kernel(w1, b1, samples): return np.dot(w1, samples) + b1
def tribas_kernel(w1, b1, samples):
    tempH = np.dot(w1, samples) + b1; H = 1 - np.abs(tempH)
    H[(tempH < -1) | (tempH > 1)] = 0
    return H
def radbas_kernel(w1, b1, samples):
    tempH = np.dot(w1, samples) + b1
    return np.exp(-np.power(tempH, 2))

#========================================================================
# ELM MAIN LOGIC
#========================================================================

def mElmLearning(train_data, test_data, Elm_Type, NumberofHiddenNeurons, ActivationFunction, execution, kfold, verbose):
    [T, P] = loadingDataset(train_data)
    [TVT, TVP] = loadingDataset(test_data)
    NumberofTrainingData=np.size(P,1); NumberofTestingData=np.size(TVP,1); NumberofInputNeurons=np.size(P,0)
    NumberofHiddenNeurons = int(NumberofHiddenNeurons)
    cm_fold_train, cm_fold_test = None, None

    if Elm_Type != 0:
        sorted_target=np.sort(np.concatenate((T,TVT),axis=0)); label=[sorted_target[0]]; j=0
        for i in range(1,NumberofTrainingData+NumberofTestingData):
            if sorted_target[i]!=label[j]: j+=1; label.append(sorted_target[i])
        number_class=j+1; NumberofOutputNeurons=number_class
        temp_T=np.zeros((NumberofOutputNeurons,NumberofTrainingData))
        for i in range(NumberofTrainingData):
            for j in range(number_class):
                if label[j]==T[i]: break
            temp_T[j][i]=1
        T=temp_T*2-1
        temp_TV_T=np.zeros((NumberofOutputNeurons,NumberofTestingData))
        for i in range(NumberofTestingData):
            for j in range(number_class):
                if label[j]==TVT[i]: break
            temp_TV_T[j][i]=1
        TVT=temp_TV_T*2-1

    start_time_train = process_time()
    InputWeight = np.random.rand(NumberofHiddenNeurons, NumberofInputNeurons)*2-1
    BiasofHiddenNeurons = np.random.rand(NumberofHiddenNeurons, 1)
    H = switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, P)
    OutputWeight = np.dot(np.linalg.pinv(np.transpose(H)), np.transpose(T))
    end_time_train = process_time()
    TrainingTime = end_time_train-start_time_train

    Y = np.transpose(np.dot(np.transpose(H), OutputWeight))
    del(H)

    start_time_test = process_time()
    tempH_test = switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, TVP)
    del(TVP)
    TY = np.transpose(np.dot(np.transpose(tempH_test), OutputWeight))
    end_time_test = process_time()
    TestingTime = end_time_test-start_time_test

    if Elm_Type == 0:
        TrainingAccuracy = round(np.square(np.subtract(T, Y)).mean(), 6)
        TestingAccuracy = round(np.square(np.subtract(TVT, TY)).mean(), 6)
    else:
        label_index_train_expected = np.argmax(T, axis=0)
        label_index_train_actual   = np.argmax(Y, axis=0)
        MissClassificationRate_Training = np.sum(label_index_train_actual != label_index_train_expected)
        TrainingAccuracy = round(1 - MissClassificationRate_Training/NumberofTrainingData, 6)

        label_index_test_expected = np.argmax(TVT, axis=0)
        label_index_test_actual   = np.argmax(TY, axis=0)
        MissClassificationRate_Testing = np.sum(label_index_test_actual != label_index_test_expected)
        TestingAccuracy = round(1 - MissClassificationRate_Testing/NumberofTestingData, 6)

        labels_range = list(range(number_class))
        cm_fold_train = confusion_matrix(label_index_train_expected, label_index_train_actual, labels=labels_range)
        cm_fold_test = confusion_matrix(label_index_test_expected, label_index_test_actual, labels=labels_range)

    # Adding verbose printing here
    if verbose:
        print(f'..................k: {execution}, k-fold: {kfold}............................')
        if Elm_Type == 0:
            print(f'Training RMSE: {TrainingAccuracy} ( {np.size(Y,0)} samples)')
            print(f'Testing  RMSE: {TestingAccuracy} ( {TY.shape[1]} samples)')
        else:
            print(f'Training Accuracy: {TrainingAccuracy*100:.2f}%')
            print(f'Testing  Accuracy: {TestingAccuracy*100:.2f}%')
        print(f'Training Time: {round(TrainingTime,2)} sec.')
        print(f'Testing  Time: {round(TestingTime,2)} sec.')

    return TrainingAccuracy, TestingAccuracy, TrainingTime, TestingTime, cm_fold_train, cm_fold_test

class melm():
    def main(self, AllData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction, nSeed, kfold, sep, verbose):
        ALL_FUNCTIONS = ['sig', 'sin', 'radbas', 'linear', 'hardlim', 'tribas']
        if ActivationFunction == 'all':
            acts = ALL_FUNCTIONS
        else:
            acts = [s.strip() for s in str(ActivationFunction or 'linear').split(',') if s.strip()]
        nh_list = [int(v.strip()) for v in str(NumberofHiddenNeurons).split(',') if str(v).strip()]

        if nSeed is None: nSeed = 1
        else: nSeed = int(nSeed)
        rnd_seed(nSeed); np.random.seed(nSeed)
        Elm_Type = int(Elm_Type)

        all_data, samples_index = mElmStruct(AllData_File, Elm_Type, sep, verbose)
        kf = KFold(n_splits=int(kfold), shuffle=True, random_state=nSeed)
        combo_results = []

        for af in acts:
            for nh in nh_list:
                if verbose: print(f'\n=== Evaluating: Activation={af}, Neurons={nh} ===')
                acc_train, acc_test, t_train, t_test = [], [], [], []
                cms_train, cms_test = [], []

                for i, (tr_idx, te_idx) in enumerate(kf.split(samples_index)):
                    train_data = all_data[samples_index[tr_idx], :]
                    test_data  = all_data[samples_index[te_idx], :]

                    # Passing 'i' and 'kfold' to the mElmLearning function for verbose printing
                    TA, TeA, TT, Tt, cm_train, cm_test = mElmLearning(train_data, test_data, Elm_Type, nh, af, i, kfold, verbose)

                    acc_train.append(TA); acc_test.append(TeA)
                    t_train.append(TT); t_test.append(Tt)
                    if cm_train is not None: cms_train.append(cm_train.astype(float))
                    if cm_test is not None: cms_test.append(cm_test.astype(float))

                mean_tr, std_tr = np.mean(acc_train)*100, np.std(acc_train)*100
                mean_te, std_te = np.mean(acc_test)*100, np.std(acc_test)*100
                mean_tt, std_tt = float(np.mean(t_train)), float(np.std(t_train))
                mean_et, std_et = float(np.mean(t_test)), float(np.std(t_test))

                combo_results.append({
                    "act": af, "n_hidden": int(nh),
                    "accuracy_train": mean_tr, "std_train": std_tr,
                    "accuracy_test": mean_te, "std_test": std_te,
                    "time_train": mean_tt, "std_time_train": std_tt,
                    "time_test": mean_et, "std_time_test": std_et,
                    "confusion_matrix_train": np.mean(cms_train, axis=0) if cms_train else None,
                    "confusion_matrix_test": np.mean(cms_test, axis=0) if cms_test else None
                })

        if verbose: print("\n==========================================")
        if verbose: print("GLOBAL RESULTS (ELM)")
        if verbose: print("==========================================")

        best_test = max(combo_results, key=lambda r: r['accuracy_test'])
        worst_test = min(combo_results, key=lambda r: r['accuracy_test'])
        global_results = {"max": best_test, "min": worst_test}

        if verbose:
            print(f"Best: act={best_test['act']}, n_hidden={best_test['n_hidden']}")
            print(f"  Train: {best_test['accuracy_train']:.2f} ± {best_test['std_train']:.2f} | Test: {best_test['accuracy_test']:.2f} ± {best_test['std_test']:.2f}")
            print(f"  Train time (s): {best_test['time_train']:.2f} ± {best_test['std_time_train']:.2f} | Test time (s): {best_test['time_test']:.2f} ± {best_test['std_time_test']:.2f}")
            print(f"Worst:  act={worst_test['act']}, n_hidden={worst_test['n_hidden']}")
            print(f"  Train: {worst_test['accuracy_train']:.2f} ± {worst_test['std_train']:.2f} | Test: {worst_test['accuracy_test']:.2f} ± {worst_test['std_test']:.2f}")
            print(f"  Train time (s): {worst_test['time_train']:.2f} ± {worst_test['std_time_train']:.2f} | Test time (s): {worst_test['time_test']:.2f} ± {worst_test['std_time_test']:.2f}")

        act_results = {}
        for act in acts:
            results_for_act = [r for r in combo_results if r['act'] == act]
            if not results_for_act: continue
            act_results[act] = {
                "max_test": max(results_for_act, key=lambda r: r['accuracy_test']),
                "min_test": min(results_for_act, key=lambda r: r['accuracy_test']),
            }

        generate_html_report_elm(global_results, act_results)


def generate_html_report_elm(global_results, act_results, output_file='elm_report.html'):
    """Generates an HTML report in a Dashboard style for ELM."""

    # Gets the directory of the current script safely
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Creates the full path for the image folder inside the script's directory
    img_dir_name = 'elm_report_images'
    img_dir_path = os.path.join(script_dir, img_dir_name)
    os.makedirs(img_dir_path, exist_ok=True)

    # Graph titles in English
    plot_and_save_cm(global_results['max']['confusion_matrix_test'], 'Avg CM - Best Global (Test)', os.path.join(img_dir_path, 'cm_global_best.png'))
    plot_and_save_cm(global_results['min']['confusion_matrix_test'], 'Avg CM - Worst Global (Test)',   os.path.join(img_dir_path, 'cm_global_worst.png'))

    for act_name, data in act_results.items():
        k_name = act_name.replace(" ", "_")
        if data.get('max_test'): plot_and_save_cm(data['max_test']['confusion_matrix_test'], f'Avg CM - Best Test {act_name}', os.path.join(img_dir_path, f'cm_act_{k_name}_best_test.png'))
        if data.get('min_test'): plot_and_save_cm(data['min_test']['confusion_matrix_test'], f'Avg CM - Worst Test {act_name}', os.path.join(img_dir_path, f'cm_act_{k_name}_worst_test.png'))

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ELM Dashboard - Results Report</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif, Arial; background: linear-gradient(135deg, #8B1538 0%, #A91E4A 50%, #6B1429 100%); min-height: 100vh; color: #333; }
            .dashboard-container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            .header { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 20px; padding: 30px; margin-bottom: 30px; text-align: center; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); }
            .header h1 { font-size: 2.5em; background: linear-gradient(45deg, #8B1538, #A91E4A); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 10px; }
            .subtitle { font-size: 1.2em; color: #666; font-weight: 300; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .stat-card { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 15px; padding: 25px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); transition: transform 0.3s ease, box-shadow 0.3s ease; }
            .stat-card:hover { transform: translateY(-5px); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15); }
            .stat-card.best { border-left: 10px solid #A5D7A7; } .stat-card.worst { border-left: 10px solid #f9a19a; }
            .card-header { display: flex; align-items: center; margin-bottom: 20px; }
            .card-icon { width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 1.5em; font-weight: bold; color: white; }
            .card-icon.best { background: linear-gradient(45deg, #4CAF50, #66BB6A); }
            .card-icon.worst { background: linear-gradient(45deg, #f44336, #EF5350); }
            .card-title { font-size: 1.3em; font-weight: 600; }
            .metric-row { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid rgba(0, 0, 0, 0.05); }
            .metric-row:last-child { border-bottom: none; }
            .metric-label { font-weight: 500; color: #555; }
            .metric-value { font-weight: 600; padding: 4px 12px; border-radius: 20px; }
            .metric-value.best { background: rgba(76, 175, 80, 0.1); }
            .metric-value.worst { background: rgba(244, 67, 54, 0.1); }
            .cm-container { text-align: center; margin-top: 20px; }
            .cm-image { max-width: 70%; height: auto; border-radius: 5px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }
            .kernels-section { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 20px; padding: 30px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); }
            .section-title { font-size: 2em; margin-bottom: 30px; text-align: center; background: linear-gradient(45deg, #8B1538, #A91E4A); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
            .kernel-group { margin-bottom: 40px; }
            .kernel-title { font-size: 1.5em; font-weight: 600; margin-bottom: 20px; padding: 15px 20px; color: white; border-radius: 10px; text-align: center; background: linear-gradient(45deg, #8B1538, #A91E4A); }
            .kernel-results { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }
            .result-card { background: rgba(255, 255, 255, 0.9); border-radius: 15px; padding: 25px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1); transition: transform 0.3s ease; }
            .result-card:hover { transform: translateY(-3px); }
            .result-card.best { border: 2px solid #4CAF50; } .result-card.worst { border: 2px solid #f44336; }
            .result-header { display: flex; align-items: center; margin-bottom: 20px; }
            .result-icon { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; color: white; font-weight: bold; }
            .logo-ufpe {
                    height: 150px;
                    width: auto;
                }
            .result-icon.best { background: #4CAF50; } .result-icon.worst { background: #f44336; }
            .result-title { font-size: 1.2em; font-weight: 600; }
            .metrics-list { list-style: none; margin-bottom: 20px; }
            .metrics-list li { padding: 8px 0; border-bottom: 1px solid rgba(0, 0, 0, 0.05); display: flex; justify-content: space-between; align-items: center; }
            .metrics-list li:last-child { border-bottom: none; }
            .metric-name { font-weight: 500; color: #555; }
            .metric-val { font-weight: 600; }
            @keyframes fadeInUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
            .stat-card, .kernels-section { animation: fadeInUp 0.6s ease forwards; }
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <div class="header">
                <img src="../../src/ufpe_logo.png" alt="UFPE Logo" class="logo-ufpe">
                <h1>ELM - Parameter Evaluation</h1>
                <p class="subtitle">Test accuracy is the metric of choice for the results</p>
            </div>
            <div class="stats-grid">
                <div class="stat-card best">
                    <div class="card-header"><div class="card-icon best">🏆</div><div class="card-title">Best Overall Performance</div></div>
                    <div class="metric-row"><span class="metric-label">Configuration (n_hidden)</span><span class="metric-value best">{{ global_results.max.n_hidden }}</span></div>
                    <div class="metric-row"><span class="metric-label">Best Activation</span><span class="metric-value best">{{ global_results.max.act }}</span></div>
                    <div class="metric-row"><span class="metric-label">Training Accuracy</span><span class="metric-value best">{{ "%.2f"|format(global_results.max.accuracy_train) }}% &plusmn; {{ "%.2f"|format(global_results.max.std_train) }}%</span></div>
                    <div class="metric-row"><span class="metric-label">Test Accuracy</span><span class="metric-value best">{{ "%.2f"|format(global_results.max.accuracy_test) }}% &plusmn; {{ "%.2f"|format(global_results.max.std_test) }}%</span></div>
                    <div class="metric-row"><span class="metric-label">Training Time</span><span class="metric-value best">{{ "%.4f"|format(global_results.max.time_train) }}s &plusmn; {{ "%.4f"|format(global_results.max.std_time_train) }}s</span></div>
                    <div class="metric-row"><span class="metric-label">Test Time</span><span class="metric-value best">{{ "%.4f"|format(global_results.max.time_test) }}s &plusmn; {{ "%.4f"|format(global_results.max.std_time_test) }}s</span></div>
                    <div class="cm-container"><img class="cm-image" src="elm_report_images/cm_global_best.png" alt="Confusion Matrix - Best Global"></div>
                </div>
                <div class="stat-card worst">
                    <div class="card-header"><div class="card-icon worst">👎</div><div class="card-title">Worst Overall Performance</div></div>
                    <div class="metric-row"><span class="metric-label">Configuration (n_hidden)</span><span class="metric-value worst">{{ global_results.min.n_hidden }}</span></div>
                    <div class="metric-row"><span class="metric-label">Worst Activation</span><span class="metric-value worst">{{ global_results.min.act }}</span></div>
                    <div class="metric-row"><span class="metric-label">Training Accuracy</span><span class="metric-value worst">{{ "%.2f"|format(global_results.min.accuracy_train) }}% &plusmn; {{ "%.2f"|format(global_results.min.std_train) }}%</span></div>
                    <div class="metric-row"><span class="metric-label">Test Accuracy</span><span class="metric-value worst">{{ "%.2f"|format(global_results.min.accuracy_test) }}% &plusmn; {{ "%.2f"|format(global_results.min.std_test) }}%</span></div>
                    <div class="metric-row"><span class="metric-label">Training Time</span><span class="metric-value worst">{{ "%.4f"|format(global_results.min.time_train) }}s &plusmn; {{ "%.4f"|format(global_results.min.std_time_train) }}s</span></div>
                    <div class="metric-row"><span class="metric-label">Test Time</span><span class="metric-value worst">{{ "%.4f"|format(global_results.min.time_test) }}s &plusmn; {{ "%.4f"|format(global_results.min.std_time_test) }}s</span></div>
                    <div class="cm-container"><img class="cm-image" src="elm_report_images/cm_global_worst.png" alt="Confusion Matrix - Worst Global"></div>
                </div>
            </div>
            <div class="kernels-section">
                <h2 class="section-title">Summary by Activation Function</h2>
                {% for act_name, data in act_results.items() %}
                <div class="kernel-group">
                    <div class="kernel-title">Activation Function: {{ act_name }}</div>
                    <div class="kernel-results">
                        {% if data.max_test %}<div class="result-card best">
                            <div class="result-header"><div class="result-icon best">👍</div><div class="result-title">Best Scenario</div></div>
                            <ul class="metrics-list">
                                <li><span class="metric-name">Configuration (n_hidden):</span><span class="metric-val">{{ data.max_test.n_hidden }}</span></li>
                                <li><span class="metric-name">Test Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.max_test.accuracy_test) }}% &plusmn; {{ "%.2f"|format(data.max_test.std_test) }}%</span></li>
                                <li><span class="metric-name">Test Time:</span><span class="metric-val">{{ "%.4f"|format(data.max_test.time_test) }}s &plusmn; {{ "%.4f"|format(data.max_test.std_time_test) }}s</span></li>
                                <li><span class="metric-name">Training Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.max_test.accuracy_train) }}% &plusmn; {{ "%.2f"|format(data.max_test.std_train) }}%</span></li>
                                <li><span class="metric-name">Training Time:</span><span class="metric-val">{{ "%.4f"|format(data.max_test.time_train) }}s &plusmn; {{ "%.4f"|format(data.max_test.std_time_train) }}s</span></li>
                            </ul>
                            <div class="cm-container"><img class="cm-image" src="elm_report_images/cm_act_{{ act_name|replace(' ', '_') }}_best_test.png" alt="CM Best Test"></div>
                        </div>{% endif %}
                        {% if data.min_test %}<div class="result-card worst">
                            <div class="result-header"><div class="result-icon worst">👎</div><div class="result-title">Worst Scenario</div></div>
                            <ul class="metrics-list">
                                <li><span class="metric-name">Configuration (n_hidden):</span><span class="metric-val">{{ data.min_test.n_hidden }}</span></li>
                                <li><span class="metric-name">Test Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.min_test.accuracy_test) }}% &plusmn; {{ "%.2f"|format(data.min_test.std_test) }}%</span></li>
                                <li><span class="metric-name">Test Time:</span><span class="metric-val">{{ "%.4f"|format(data.min_test.time_test) }}s &plusmn; {{ "%.4f"|format(data.min_test.std_time_test) }}s</span></li>
                                <li><span class="metric-name">Training Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.min_test.accuracy_train) }}% &plusmn; {{ "%.2f"|format(data.min_test.std_train) }}%</span></li>
                                <li><span class="metric-name">Training Time:</span><span class="metric-val">{{ "%.4f"|format(data.min_test.time_train) }}s &plusmn; {{ "%.4f"|format(data.min_test.std_time_train) }}s</span></li>
                            </ul>
                            <div class="cm-container"><img class="cm-image" src="elm_report_images/cm_act_{{ act_name|replace(' ', '_') }}_worst_test.png" alt="CM Worst Test"></div>
                        </div>{% endif %}
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
    </body>
    </html>
    """

    # Path corrections to save files in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_file)
    img_dir_path = os.path.join(script_dir, 'elm_report_images')

    act_names = {act: act for act in act_results.keys()}
    template = Template(html_template)
    html_content = template.render(global_results=global_results, act_results=act_results, act_names=act_names)
    try:
        with open(output_path, 'w', encoding='utf-8') as f: f.write(html_content)
        print(f"\nHTML report successfully generated: '{output_path}'")
    except IOError as e: print(f"\nError saving HTML report: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ELM Parameter Tester with HTML Report Generation.')
    parser.add_argument('-tall', '--AllData_File',dest='AllData_File',action='store', required=True)
    parser.add_argument('-ty', '--Elm_Type',dest='Elm_Type',action='store',required=True, help="0 for regression; 1 for classification")
    parser.add_argument('-nh', '--nHiddenNeurons',dest='nHiddenNeurons',action='store',required=True, help="Number of hidden neurons (list: '10,20,30')")
    parser.add_argument('-af', '--ActivationFunction',dest='ActivationFunction',action='store', required=True, help="Activation function (list: 'sig,sin' or 'all')")
    parser.add_argument('-sd', '--seed',dest='nSeed',action='store')
    parser.add_argument('-kfold', dest='kfold', action='store', default=5, help="Number of folds for cross-validation (default: 5)")
    parser.add_argument('-sep', dest='sep', action='store', default=';', help="CSV separator (default: ';')")
    parser.add_argument('-v', dest='verbose', action='store_true', default=True)
    args = parser.parse_args()

    ff = melm()
    # --- CORRECTED MAIN FUNCTION CALL ---
    ff.main(args.AllData_File, args.Elm_Type, args.nHiddenNeurons, args.ActivationFunction,
            args.nSeed, args.kfold, args.sep, args.verbose)
