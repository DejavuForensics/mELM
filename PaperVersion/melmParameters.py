# -*- coding: utf-8 -*-
"""Avaliação de ativações ELM/mELM para classificação e regressão.

Este arquivo contém as ativações-base do projeto mELM.
Variantes de pesquisa ficam no projeto irmão mELM-Experimental-Kernels.
Nenhuma acurácia é presumida pelo código.
"""

import numpy as np
import pandas as pd
import argparse
import os
import sys
import re
import csv
from time import process_time, time
from datetime import datetime, timedelta
from random import seed as rnd_seed
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader, select_autoescape


#========================================================================
# VISUALIZAÇÃO DE DADOS E RELATÓRIOS
#========================================================================

def plot_and_save_cm(cm, title, filename, cm_std=None, class_labels=None, language='pt-br'):
    """
    Gera a matriz de confusão em porcentagem (0–100%) normalizada por linha.

    Útil para visualizar:
    - Taxa de Verdadeiro Positivo (TPR)
    - Taxa de Falso Positivo (FPR)
    - Comportamento por classe

    Args:
        cm: Matriz de confusão (array numpy 2D)
        title: Título do gráfico
        filename: Caminho do arquivo PNG de saída
    """
    if cm is None:
        return

    cm = np.asarray(cm, dtype=float)
    if cm.size == 0 or np.all(cm == 0):
        return

    if cm.shape == (2, 2):
        # Interno sklearn: [[TN, FP], [FN, TP]], linhas=real, colunas=previsto.
        # Exibição como no SVM-new: linhas=previsto, colunas=real, ordem [+, -].
        display = np.array([[cm[1, 1], cm[0, 1]], [cm[1, 0], cm[0, 0]]], dtype=float)
        std = np.zeros_like(cm) if cm_std is None else np.asarray(cm_std, dtype=float)
        std_display = np.array([[std[1, 1], std[0, 1]], [std[1, 0], std[0, 0]]], dtype=float)
        col_sums = display.sum(axis=0, keepdims=True)
        cm_percent = np.divide(display, col_sums, out=np.zeros_like(display), where=col_sums != 0) * 100.0
        cm_std_percent = np.divide(std_display, col_sums, out=np.zeros_like(std_display), where=col_sums != 0) * 100.0
        raw = list(class_labels or [0, 1])
        if language == "en":
            negative = f"Negative class ({raw[0]})"
            positive = f"Positive class ({raw[1]})"
        else:
            negative = f"Classe negativa ({raw[0]})"
            positive = f"Classe positiva ({raw[1]})"
        x_labels = [positive, negative]
        y_labels = [positive, negative]
        codes = np.array([["TP", "FP"], ["FN", "TN"]])
        annot_matrix = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                annot_matrix[i, j] = f"{codes[i,j]}\n{cm_percent[i,j]:.1f}%\n±{cm_std_percent[i,j]:.2f}%"
    else:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0
        annot_matrix = np.vectorize(lambda value: f"{value:.1f}%")(cm_percent)
        x_labels = class_labels or list(range(cm.shape[0]))
        y_labels = x_labels

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm_percent,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Percentage (%)" if language == "en" else "Porcentagem (%)"},
        xticklabels=x_labels,
        yticklabels=y_labels
    )

    plt.title(title)
    if language == "en":
        plt.xlabel("Actual label" if cm.shape == (2, 2) else "Predicted label")
        plt.ylabel("Predicted label" if cm.shape == (2, 2) else "True label")
    else:
        plt.xlabel("Rótulo Real" if cm.shape == (2, 2) else "Rótulo Predito")
        plt.ylabel("Rótulo Previsto" if cm.shape == (2, 2) else "Rótulo Verdadeiro")
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()


#========================================================================
# ESTIMATIVA DE TEMPO E RASTREAMENTO DE PROGRESSO
#========================================================================

def estimate_total_time(n_kernels, n_neurons_configs, n_folds, n_samples):
    """
    Estima tempo total de execução baseado em benchmarks empíricos.
    
    Args:
        n_kernels: Número de kernels a testar
        n_neurons_configs: Número de configurações de neurônios
        n_folds: Número de folds
        n_samples: Número total de amostras no dataset
    
    Retorna:
        estimated_seconds: Tempo estimado em segundos
    """
    # Tempo base por fold baseado no tamanho do dataset
    if n_samples < 500:
        time_per_fold = 0.05  # 50ms
    elif n_samples < 2000:
        time_per_fold = 0.10  # 100ms
    else:
        time_per_fold = 0.30  # 300ms
    
    # Total de iterações
    total_iterations = n_kernels * n_neurons_configs * n_folds
    
    # Tempo estimado
    estimated_seconds = total_iterations * time_per_fold
    
    # Adiciona overhead (~10%)
    estimated_seconds *= 1.1
    
    return estimated_seconds


def format_time(seconds):
    """
    Formata segundos em formato legível.
    
    Args:
        seconds: Tempo em segundos
    
    Retorna:
        String formatada (ex: "5m 30s", "1h 15m", "45s")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def print_time_estimate(n_kernels, n_neurons_configs, n_folds, n_samples):
    """
    Imprime estimativa de tempo inicial.
    """
    estimated_seconds = estimate_total_time(n_kernels, n_neurons_configs, n_folds, n_samples)
    
    print("\n" + "="*80)
    print("⏱️  ESTIMATIVA DE TEMPO")
    print("="*80)
    print(f"  Kernels a testar:      {n_kernels}")
    print(f"  Configurações neuron.: {n_neurons_configs}")
    print(f"  K-folds:               {n_folds}")
    print(f"  Samples no dataset:    {n_samples}")
    print(f"  Total de iterações:    {n_kernels * n_neurons_configs * n_folds}")
    print(f"  ")
    print(f"  ⏱️  Tempo estimado:      ~{format_time(estimated_seconds)}")
    
    end_time = datetime.now() + timedelta(seconds=estimated_seconds)
    print(f"  🕐 Término previsto:    {end_time.strftime('%H:%M:%S')}")
    print("="*80 + "\n")


def print_progress_info(current_kernel, total_kernels, current_nh, total_nh, 
                       current_fold, total_folds, start_time):
    """
    Imprime informação de progresso com tempo restante.
    
    Args:
        current_kernel: Índice do kernel atual (base 0)
        total_kernels: Total de kernels
        current_nh: Índice da config de neurônios atual (base 0)
        total_nh: Total de configurações de neurônios
        current_fold: Fold atual (base 0)
        total_folds: Total de folds
        start_time: Timestamp do início
    """
    # Calcula progresso total
    total_iterations = total_kernels * total_nh * total_folds
    completed_iterations = (
        current_kernel * total_nh * total_folds +
        current_nh * total_folds +
        current_fold
    )
    
    percent = 100.0 * completed_iterations / total_iterations if total_iterations > 0 else 0
    
    # Calcula tempo restante
    elapsed_time = time() - start_time
    if completed_iterations > 0:
        time_per_iteration = elapsed_time / completed_iterations
        remaining_iterations = total_iterations - completed_iterations
        remaining_time = time_per_iteration * remaining_iterations
        remaining_str = format_time(remaining_time)
        elapsed_str = format_time(elapsed_time)
    else:
        remaining_str = "calculando..."
        elapsed_str = "0s"
    
    # Barra de progresso
    bar_length = 50
    filled_length = int(bar_length * completed_iterations / total_iterations) if total_iterations > 0 else 0
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    print(f"\n  ⏱️  [{bar}] {percent:.1f}%")
    print(f"  Decorrido: {elapsed_str} | Restante: {remaining_str}\n")


#========================================================================
# DIAGNÓSTICO AUTOMÁTICO DE KERNEL
#========================================================================

def diagnose_kernel_performance(results, kernel_name):
    """Gera alertas descritivos para classificação binária, sem prometer desempenho."""
    accuracy = float(results.get('accuracy_test', 0.0))
    std = float(results.get('std_test', 0.0))
    cm = results.get('confusion_matrix_test')
    recall_positive = fpr = None
    if cm is not None and np.asarray(cm).shape == (2, 2):
        tn, fp, fn, tp = np.asarray(cm, dtype=float).ravel()
        recall_positive = tp / (tp + fn) if tp + fn else None
        fpr = fp / (fp + tn) if fp + tn else None

    diagnosis = {'status': 'good', 'problem': None, 'explanation': '', 'recommendation': ''}
    if recall_positive is not None and recall_positive < 0.01:
        diagnosis.update(
            status='critical', problem='Recall da classe positiva próximo de zero',
            explanation=f"A ativação {kernel_name} teve recall positivo de {recall_positive*100:.2f}% nos folds agregados.",
            recommendation="Inspecione a matriz por fold, o balanceamento das classes, a poda e a escala dos atributos. Compare com um baseline e confirme em dados não usados para escolher a configuração."
        )
    elif std > 15.0:
        diagnosis.update(
            status='unstable', problem='Alta variação entre folds',
            explanation=f"O desvio padrão da acurácia de teste foi {std:.2f} pontos percentuais.",
            recommendation="Verifique as métricas por fold, a distribuição das classes e a sensibilidade à semente. O limite de 15 p.p. é uma regra operacional deste relatório, não um critério universal."
        )
    elif fpr is not None and fpr > 0.15:
        diagnosis.update(
            status='warning', problem='FPR elevado',
            explanation=f"A taxa de falso positivo agregada foi {fpr*100:.2f}%.",
            recommendation="Avalie o custo dos erros para o problema e examine a matriz de cada fold. Uma mudança do critério de decisão deve ser escolhida no treino/validação, sem usar o teste final."
        )
    elif accuracy < 80.0:
        diagnosis.update(
            status='poor', problem='Acurácia abaixo do limite de atenção',
            explanation=f"A acurácia média de teste foi {accuracy:.2f}%.",
            recommendation="Compare com um baseline simples e examine recall, FPR e métricas por classe antes de escolher outra ativação ou alterar parâmetros. O limite de 80% é apenas uma regra de triagem."
        )
    else:
        diagnosis.update(
            explanation=f"A acurácia média foi {accuracy:.2f}% (desvio padrão {std:.2f} p.p.). Este resultado é descritivo da validação cruzada.",
            recommendation="Registre a configuração e confirme o desempenho em um teste independente antes de tratá-la como resultado final."
        )
    return diagnosis


def print_diagnosis(diagnosis, kernel_name):
    """
    Imprime diagnóstico formatado no terminal.
    """
    status_icons = {
        'critical': '❌',
        'unstable': '⚠️',
        'warning': '⚠️',
        'poor': '⚠️',
        'good': '✅',
        'unknown': '❓'
    }
    
    icon = status_icons.get(diagnosis['status'], '❓')
    
    print(f"\n  {icon} DIAGNÓSTICO: {kernel_name}")
    if diagnosis['problem']:
        print(f"  └─ Problema: {diagnosis['problem']}")
    
    print(f"\n  📝 Explicação:")
    # Quebra texto em linhas de ~75 caracteres
    words = diagnosis['explanation'].split()
    line = "     "
    for word in words:
        if len(line) + len(word) + 1 > 75:
            print(line)
            line = "     " + word
        else:
            line += " " + word
    if line.strip():
        print(line)
    
    print(f"\n  💡 Recomendação:")
    words = diagnosis['recommendation'].split()
    line = "     "
    for word in words:
        if len(line) + len(word) + 1 > 75:
            print(line)
            line = "     " + word
        else:
            line += " " + word
    if line.strip():
        print(line)
    print()


#========================================================================
# MÉTRICAS DE DETECÇÃO PARA CLASSIFICAÇÃO DE MALWARE
#========================================================================

def calculate_detection_metrics(cm):
    """
    Calcula métricas específicas para detecção de malware (classificação binária).
    
    Métricas críticas para segurança cibernética:
    - Detection Rate (Recall/TPR): Taxa de detecção de malware
    - False Positive Rate (FPR): Taxa de falsos alarmes
    - Precision: Precisão nas detecções
    - F1-Score: Média harmônica de Precision e Recall
    
    Args:
        cm: Matriz de confusão 2x2 no formato [[TN, FP], [FN, TP]]
    
    Retorna:
        dict com todas as métricas ou None se não for 2x2
    """
    if cm is None or cm.shape != (2, 2):
        return None
    
    TN = float(cm[0, 0])  # True Negatives (Verdadeiros Negativos)
    FP = float(cm[0, 1])  # False Positives (Falsos Positivos)
    FN = float(cm[1, 0])  # False Negatives (Falsos Negativos)
    TP = float(cm[1, 1])  # True Positives (Verdadeiros Positivos)
    
    total = TN + FP + FN + TP
    if total == 0:
        return None
    
    metrics = {
        'accuracy': (TP + TN) / total,
        'detection_rate': TP / (TP + FN) if (TP + FN) > 0 else 0,
        'false_positive_rate': FP / (FP + TN) if (FP + TN) > 0 else 0,
        'precision': TP / (TP + FP) if (TP + FP) > 0 else 0,
        'specificity': TN / (TN + FP) if (TN + FP) > 0 else 0,
        'f1_score': 0,
        'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN)
    }
    
    if metrics['precision'] + metrics['detection_rate'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['detection_rate']) /                               (metrics['precision'] + metrics['detection_rate'])
    
    return metrics


def print_detection_metrics(metrics, dataset_name="Test", verbose=True):
    """
    Imprime as métricas de detecção de forma formatada.
    
    Args:
        metrics: Dicionário retornado por calculate_detection_metrics
        dataset_name: Nome do dataset (Treino/Teste)
        verbose: Se True, imprime as métricas
    """
    if not verbose or metrics is None:
        return
    
    print(f"\n{'='*70}")
    print(f"  MÉTRICAS DE DETECÇÃO DE MALWARE - Dataset: {dataset_name}")
    print(f"{'='*70}")
    
    print(f"\n  📊 Matriz de Confusão:")
    print(f"    True Negatives  (TN): {metrics['TN']:6d}  (Benignos identificados corretamente)")
    print(f"    False Positives (FP): {metrics['FP']:6d}  (Falsos alarmes)")
    print(f"    False Negatives (FN): {metrics['FN']:6d}  (Malware NÃO detectado)")
    print(f"    True Positives  (TP): {metrics['TP']:6d}  (Malware detectado)")
    
    print(f"\n  🎯 Métricas de Performance:")
    print(f"    Acurácia Geral:              {metrics['accuracy']*100:6.2f}%")
    
    # Detection Rate: dinâmico
    dr_icon = "✓" if metrics['detection_rate'] >= 0.90 else "⚠️  CRÍTICO!"
    print(f"    Taxa de Detecção (Recall):   {metrics['detection_rate']*100:6.2f}%  {dr_icon}")
    
    # False Positive Rate: dinâmico
    fpr_icon = "✓" if metrics['false_positive_rate'] <= 0.10 else "⚠️  CRÍTICO!"
    print(f"    Taxa Falso Positivo (FPR):   {metrics['false_positive_rate']*100:6.2f}%  {fpr_icon}")
    
    print(f"    Precisão (Precision):        {metrics['precision']*100:6.2f}%")
    
    # Especificidade: dinâmico
    spec_icon = "✓" if metrics['specificity'] >= 0.90 else "⚠️"
    print(f"    Especificidade (TNR):        {metrics['specificity']*100:6.2f}%  {spec_icon}")
    
    print(f"    F1-Score:                    {metrics['f1_score']*100:6.2f}%")
    
    print(f"\n  💡 Análise:")
    if metrics['detection_rate'] >= 0.95:
        print(f"    ✓ Excelente taxa de detecção (≥95%)")
    elif metrics['detection_rate'] >= 0.90:
        print(f"    ✓ Boa taxa de detecção (≥90%)")
    else:
        print(f"    ⚠️  Taxa de detecção pode ser melhorada (<90%)")
    
    if metrics['false_positive_rate'] <= 0.05:
        print(f"    ✓ Baixa taxa de falso positivo (≤5%)")
    elif metrics['false_positive_rate'] <= 0.10:
        print(f"    ⚠️  Taxa de falso positivo moderada (≤10%)")
    else:
        print(f"    ⚠️  Taxa de falso positivo alta (>10%)")
    
    print(f"{'='*70}\n")


#========================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DE DADOS
#========================================================================

def eliminateNaN_All_data(all_data):
    """
    Elimina valores NaN substituindo-os por zero.
    
    Essencial para lidar com datasets de malware do mundo real onde:
    - Valores ausentes podem ocorrer durante a extração de características
    - Algumas ferramentas de análise estática produzem saídas esparsas
    
    Args:
        all_data: Matriz de dados de entrada
    
    Retorna:
        all_data: Matriz de dados limpa
    """
    all_data = np.nan_to_num(all_data)
    return all_data

def autodetect_separator(data_file, verbose=False):
    """
    Autodetecta o separador do CSV entre vários candidatos:
    ',', ';', tab, '|', ':', '^', '~'

    Estratégia simples: conta quantas vezes cada separador aparece
    nas primeiras linhas e escolhe o mais frequente.
    """
    candidate_seps = [
        (',', 'vírgula'),
        (';', 'ponto e vírgula'),
        ('\t', 'tab'),
        ('|', 'pipe'),
        (':', 'dois-pontos'),
        ('^', 'caret'),
        ('~', 'til'),
    ]

    counts = {sep: 0 for sep, _ in candidate_seps}

    try:
        with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
            # lê algumas linhas para estimar
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                for sep, _ in candidate_seps:
                    counts[sep] += line.count(sep)
    except Exception as e:
        if verbose:
            print(f"[autodetect_separator] Erro ao ler arquivo: {e}")
        # fallback antigo
        return ';'

    # escolhe o separador com maior contagem
    best_sep = max(counts, key=lambda k: counts[k])

    # se nada apareceu, mantém o comportamento anterior e volta a ';'
    if counts[best_sep] == 0:
        best_sep = ';'

    if verbose:
        desc = dict(candidate_seps).get(best_sep, best_sep)
        print(f"[autodetect_separator] Separador detectado: '{best_sep}' ({desc})")

    return best_sep


def mElmStruct(data_file, Elm_Type, sep, verbose, return_metadata=False, n_features=None):
    """
    Carrega e estrutura o dataset a partir de um arquivo CSV.
    
    Suporta datasets de malware no formato padrão:
    - Primeira coluna: nome do arquivo/identificador (ignorado)
    - Segunda coluna: rótulo (0=benigno, 1=malware)
    - Colunas restantes: características (opcodes, syscalls, chamadas de API, etc.)
    
    Args:
        data_file: Caminho para o arquivo CSV
        Elm_Type: 0=regressão, 1=classificação
        sep: Caractere separador do CSV (ou None para auto-detecção)
        verbose: Ativa logs detalhados
        
    Retorna:
        all_data: Matriz de dados processada
        samples_index: Array de índices para amostragem
    """
    # LIBSVM é detectado pelo padrão rótulo índice:valor e convertido para
    # uma matriz densa, que é o formato interno usado pelo mELM.
    try:
        with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
            first_content = next((line.strip() for line in f if line.strip()), '')
        tokens = first_content.split()
        is_libsvm = len(tokens) > 1 and any(':' in token for token in tokens[1:])
    except OSError as exc:
        raise ValueError(f"Não foi possível ler o arquivo '{data_file}': {exc}") from exc

    if is_libsvm:
        sparse_features, target = load_svmlight_file(
            data_file, zero_based='auto', n_features=n_features
        )
        # load_svmlight_file retorna (features, rótulos). O leitor normalmente
        # fornece uma matriz esparsa, mas aceitar também um ndarray evita
        # depender de um método específico para a representação carregada.
        feature_matrix = (
            sparse_features.toarray()
            if hasattr(sparse_features, 'toarray')
            else np.asarray(sparse_features)
        ).astype(float, copy=False)
        all_data = np.column_stack((target.astype(float, copy=False), feature_matrix))
        feature_names = [f"feature_{i + 1}" for i in range(feature_matrix.shape[1])]
        samples_index = np.arange(all_data.shape[0])
        if verbose:
            print(f"[mElmStruct] LIBSVM detectado: {all_data.shape[0]} amostras, "
                  f"{feature_matrix.shape[1]} atributos.")
        return (all_data, samples_index, feature_names) if return_metadata else (all_data, samples_index)

    # Determinar separador:
    # - se não for passado ou for "auto": autodetecta entre vários candidatos
    # - se for passado: aceita apelidos como "tab", "pipe", etc.
    if not sep or str(sep).lower() == 'auto':
        sep_character = autodetect_separator(data_file, verbose)
    else:
        raw_sep = str(sep)
        sep_lower = raw_sep.lower()

        if sep_lower in ('tab', '\\t'):
            sep_character = '\t'
        elif sep_lower in ('pipe', '|'):
            sep_character = '|'
        elif sep_lower in (',', 'comma', 'vírgula', 'virgula'):
            sep_character = ','
        elif sep_lower in (';', 'semicolon', 'ponto_e_virgula', 'ponto-virgula'):
            sep_character = ';'
        else:
            sep_character = raw_sep

        if verbose:
            print(f"[mElmStruct] Separador informado via -sep: '{sep_character}'")

    # Lê o CSV bruto
    df = pd.read_csv(
        data_file,
        sep=sep_character,
        decimal=".",
        header=None,
        engine='python'
    )

    # Preserva os nomes das features para a auditoria de poda.
    feature_names = [str(v).strip() for v in df.iloc[0, 2:].tolist()]
    feature_names = [name or f"feature_{i + 1}" for i, name in enumerate(feature_names)]

    # Remove primeira linha (cabeçalho) e primeira coluna (ID), mantendo só label + features
    df_vals = df.iloc[1:, 1:]

    # Converte TUDO para numérico; o que não conseguir vira NaN
    df_vals = df_vals.apply(pd.to_numeric, errors='coerce')

    # Converte para numpy float diretamente
    all_data = df_vals.to_numpy(dtype=float)

    # Substitui NaN por 0
    all_data = eliminateNaN_All_data(all_data)

    if int(Elm_Type) != 0:
        if verbose:
            print('Permutando ordem dos dados de entrada para classificação')
        samples_index = np.random.permutation(np.size(all_data, 0))
    else:
        samples_index = np.arange(0, np.size(all_data, 0))

    return (all_data, samples_index, feature_names) if return_metadata else (all_data, samples_index)


def fit_correlation_pruning(labels, feature_matrix, threshold=None, leakage_threshold=0.999):
    """Ajusta seleção por correlação somente no treino do fold.

    Features constantes são auditadas e descartadas. Correlações quase perfeitas
    com o alvo são marcadas como suspeita de leakage e removidas independente de
    threshold. Com threshold=None, todas as demais features são mantidas.
    """
    labels = np.asarray(labels, dtype=float).reshape(-1)
    feature_matrix = np.asarray(feature_matrix, dtype=float)
    selected, suspicious, constant, correlations = [], set(), set(), {}
    for col in range(feature_matrix.shape[1]):
        values = feature_matrix[:, col]
        if np.std(values) == 0:
            correlations[col] = 0.0
            constant.add(col)
            continue
        corr = float(np.corrcoef(values, labels)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
        correlations[col] = corr
        if abs(corr) >= leakage_threshold:
            suspicious.add(col)
        elif threshold is None or abs(corr) >= threshold:
            selected.append(col)
    if not selected:
        selected = [i for i in range(feature_matrix.shape[1])
                    if i not in suspicious and i not in constant]
    if not selected:
        raise ValueError("A poda removeu todas as features (suspeitas/constantes); revise o threshold e o dataset.")
    return selected, suspicious, constant, correlations


def fit_variance_pruning(feature_matrix, threshold):
    """Seleciona por variância relativa, ajustada apenas no treino do fold.

    Cada atributo é escalado para [0, 1] usando mínimo e máximo do treino,
    seguindo o critério do script de poda enviado pelo orientador. A seleção
    mantém atributos cuja variância normalizada é >= threshold * variância
    máxima. Os valores originais dos atributos são usados pelo modelo.
    """
    feature_matrix = np.asarray(feature_matrix, dtype=float)
    if feature_matrix.ndim != 2 or feature_matrix.shape[1] == 0:
        raise ValueError("A poda por variância requer uma matriz com atributos.")

    minima = np.min(feature_matrix, axis=0)
    amplitudes = np.max(feature_matrix, axis=0) - minima
    constant = set(np.flatnonzero(amplitudes == 0).tolist())
    nonconstant = np.flatnonzero(amplitudes > 0)
    if nonconstant.size == 0:
        raise ValueError("A poda por variância não encontrou atributos não constantes no treino.")

    normalized = (feature_matrix[:, nonconstant] - minima[nonconstant]) / amplitudes[nonconstant]
    ddof = 1 if feature_matrix.shape[0] > 1 else 0
    variances = np.var(normalized, axis=0, ddof=ddof)
    max_variance = float(np.max(variances))
    if not np.isfinite(max_variance) or max_variance <= 0:
        raise ValueError("A poda por variância não encontrou variância positiva nos atributos de treino.")

    relative_variance_by_column = {
        int(col): float(value / max_variance) for col, value in zip(nonconstant, variances)
    }
    selected = [
        int(col) for col, value in zip(nonconstant, variances)
        if value >= threshold * max_variance
    ]
    if not selected:
        raise ValueError("A poda por variância não selecionou atributos; revise o limiar.")
    return selected, constant, relative_variance_by_column, max_variance


def estimate_melm_kernel_time(n_samples, n_features, n_hidden_values, n_folds, activation):
    """Estimativa operacional aproximada baseada no custo da camada oculta e pinv.

    A constante é heurística e deve ser recalibrada com tempos medidos na máquina.
    """
    n_train = max(1, int(n_samples * (n_folds - 1) / n_folds))
    activation_factor = 2.5 if str(activation).startswith('fuzzy-') else (
        1.6 if str(activation).startswith('bitwise-') else 1.0
    )
    total = 0.0
    for hidden in n_hidden_values:
        hidden = max(1, int(hidden))
        work = activation_factor * n_train * max(1, n_features) * hidden
        work += n_train * hidden * hidden + hidden ** 3
        total += 2.5e-8 * work * n_folds
    return float(total)


def loadingDataset(dataset):
    """
    Separa os alvos (T) e características (P) do dataset.
    
    Formato padrão (após mElmStruct):
    - Primeira coluna: alvo (rótulo)
    - Colunas restantes: características
    
    Args:
        dataset: Matriz de dados combinada
        
    Retorna:
        T: Vetor de alvos
        P: Matriz de características
    """
    T = np.transpose(dataset[:, 0])
    P = np.transpose(dataset[:, 1:np.size(dataset, 1)])
    return T, P


def virusNormFunction(matrix, verbose):
    """
    Normalização virusNorm - normaliza dados para o intervalo [0.1, 0.9].
    
    Projetado especificamente para vetores de características de malware do banco de dados VirusShare.
    Esta normalização previne saturação em operações morfológicas e
    mantém o poder discriminativo para classificação binária.
    
    O intervalo [0.1, 0.9] é escolhido para:
    - Evitar efeitos de borda em operações morfológicas
    - Preservar a variância das características para melhor discriminação
    - Corresponder à distribuição de datasets reais de malware
    
    Args:
        matrix: Matriz de características de entrada
        verbose: Ativa logs
        
    Retorna:
        R: Matriz normalizada no intervalo [0.1, 0.9]
    """
    if verbose:
        print('Aplicando normalização virusNorm [0.1, 0.9]')
    
    vector = matrix.flatten()
    maxi = np.max(vector)
    mini = np.min(vector)
    
    if maxi == mini:
        # Matriz constante: mapeia tudo para o meio do intervalo
        return np.ones_like(matrix) * 0.5
    
    ra = 0.9  # Limite superior
    rb = 0.1  # Limite inferior
    R = (((ra - rb) * (matrix - mini)) / (maxi - mini)) + rb
    
    return R


#========================================================================
# KERNELS MORFOLÓGICOS - Inovações Principais para mELM
#========================================================================

def erosion_kernel(w1, b1, samples):
    """
    Kernel de Erosão para ELM Morfológica.
    
    Implementa a erosão morfológica, que atua como um operador de mínimo
    entre as características para detecção de malware, capturando o comportamento de "pior caso".
    
    Args:
        w1: Pesos de entrada (elementos estruturantes)
        b1: Vetor de viés
        samples: Amostras de entrada (vetores de características)
        
    Retorna:
        H: Ativações da camada oculta após erosão
    """
    n_hidden, n_features = w1.shape
    n_samples = samples.shape[1]
    
    H = np.zeros((n_hidden, n_samples))
    for i in range(n_hidden):
        se = w1[i, :]
        # Erosão: toma o mínimo de (x - se) sobre a dimensão das características
        H[i, :] = np.min(samples - se[:, np.newaxis], axis=0)
    
    H = H + b1.flatten()[:, np.newaxis]
    return H


def dilation_kernel(w1, b1, samples):
    """
    Kernel de Dilatação para ELM Morfológica.
    
    Implementa a dilatação morfológica, que atua como um operador de máximo
    entre as características, capturando o comportamento de "melhor caso" ou espalhamento de ativação.
    
    Args:
        w1: Pesos de entrada (elementos estruturantes)
        b1: Vetor de viés
        samples: Amostras de entrada (vetores de características)
        
    Retorna:
        H: Ativações da camada oculta após dilatação
    """
    n_hidden, n_features = w1.shape
    n_samples = samples.shape[1]
    
    H = np.zeros((n_hidden, n_samples))
    for i in range(n_hidden):
        se = w1[i, :]
        # Dilatação: toma o máximo de (x + se) sobre a dimensão das características
        H[i, :] = np.max(samples + se[:, np.newaxis], axis=0)
    
    H = H + b1.flatten()[:, np.newaxis]
    return H


def fuzzy_erosion_kernel(w1, b1, samples):
    """
    Kernel Fuzzy-Erosão - Versão suave da erosão morfológica.
    
    Usa princípios de lógica fuzzy para criar fronteiras de decisão suaves.
    Computa o mínimo fuzzy através do produto de diferenças transformadas por sigmoide.
    
    Vantagens para detecção de malware:
    - Lida com incerteza na extração de características
    - Robusto a código ofuscado
    - Gradientes suaves para melhor otimização
    
    Otimizado com processamento em lote para lidar com grandes datasets de malware.
    
    Args:
        w1: Pesos de entrada
        b1: Vetor de viés
        samples: Amostras de entrada
        
    Retorna:
        H: Ativações fuzzy-erodidas
    """
    n_hidden, n_features = w1.shape
    n_samples = samples.shape[1]
    
    # Processamento em lote para eficiência de memória
    batch_size = 200 if n_samples > 500 else n_samples
    H_batches = []
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        samples_batch = samples[:, start_idx:end_idx]
        
        w1_exp = w1[:, :, np.newaxis]
        samples_exp = samples_batch[np.newaxis, :, :]
        differences = samples_exp - w1_exp
        
        # Pertinência fuzzy via sigmoide
        fuzzy_values = 1.0 / (1.0 + np.exp(-np.clip(differences, -40, 40)))
        
        # Mínimo fuzzy via produto (no espaço log para estabilidade numérica)
        log_fuzzy = np.log(fuzzy_values + 1e-10)
        H_batch = np.exp(np.sum(log_fuzzy, axis=1))
        
        H_batches.append(H_batch)
        del w1_exp, samples_exp, differences, fuzzy_values, log_fuzzy
    
    H = np.concatenate(H_batches, axis=1)
    H = H + b1.flatten()[:, np.newaxis]
    return H


def fuzzy_dilation_kernel(w1, b1, samples):
    """
    Kernel Fuzzy-Dilatação - Versão suave da dilatação morfológica.
    
    Usa operação de max fuzzy implementada via truque do complemento:
    max(a, b) = 1 - min(1-a, 1-b)
    
    Este kernel fornece expansões suaves de regiões de alta ativação,
    particularmente útil para destacar padrões maliciosos fortes.
    
    Args:
        w1: Pesos de entrada
        b1: Vetor de viés
        samples: Amostras de entrada
        
    Retorna:
        H: Ativações fuzzy-dilatadas
    """
    n_hidden, n_features = w1.shape
    n_samples = samples.shape[1]
    
    batch_size = 200 if n_samples > 500 else n_samples
    H_batches = []
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        samples_batch = samples[:, start_idx:end_idx]
        
        w1_exp = w1[:, :, np.newaxis]
        samples_exp = samples_batch[np.newaxis, :, :]
        differences = samples_exp - w1_exp
        
        # Pertinência fuzzy
        fuzzy_values = 1.0 / (1.0 + np.exp(-np.clip(differences, -40, 40)))
        
        # Dilatação fuzzy via complemento da erosão
        complement = 1.0 - fuzzy_values
        log_comp = np.log(complement + 1e-10)
        H_batch = 1.0 - np.exp(np.sum(log_comp, axis=1))
        
        H_batches.append(H_batch)
        del w1_exp, samples_exp, differences, fuzzy_values, complement, log_comp
    
    H = np.concatenate(H_batches, axis=1)
    H = H + b1.flatten()[:, np.newaxis]
    return H


def bitwise_erosion_kernel(w1, b1, samples):
    """
    Kernel Bitwise-Erosão.
    
    Aproxima a erosão morfológica usando máscaras binárias:
    - Características acima do limiar são tratadas como 1, outras como 0
    - Erosão aproximada via operações do tipo AND
    
    Útil para características comportamentais binárias (ex: presença/ausência de chamada de API).
    """
    threshold = 0.5
    samples_bin = (samples > threshold).astype(float)
    w1_bin = (w1 > threshold).astype(float)
    
    n_hidden, n_features = w1.shape
    H = np.zeros((n_hidden, samples.shape[1]))
    
    for i in range(n_hidden):
        se = w1_bin[i, :]
        # Erosão: 1 apenas se todas as características requeridas estiverem presentes
        match = np.all(samples_bin >= se[:, np.newaxis], axis=0)
        H[i, :] = match.astype(float)
    
    H = H + b1.flatten()[:, np.newaxis]
    return H


def bitwise_dilation_kernel(w1, b1, samples):
    """
    Kernel Bitwise-Dilatação.
    
    Aproxima a dilatação morfológica usando máscaras binárias:
    - Características acima do limiar são tratadas como 1, outras como 0
    - Dilatação aproximada via operações do tipo OR
    
    Para detecção de malware, isso captura se qualquer um dos padrões "maliciosos"
    está presente na amostra.
    """
    threshold = 0.5
    samples_bin = (samples > threshold).astype(float)
    w1_bin = (w1 > threshold).astype(float)
    
    n_hidden, n_features = w1.shape
    H = np.zeros((n_hidden, samples.shape[1]))
    
    for i in range(n_hidden):
        se = w1_bin[i, :]
        # Dilatação binária: existe ao menos uma característica ativa em ambos.
        match = np.any((samples_bin == 1.0) & (se[:, np.newaxis] == 1.0), axis=0)
        H[i, :] = match.astype(float)
    
    H = H + b1.flatten()[:, np.newaxis]
    return H


#========================================================================
# FUNÇÕES DE ATIVAÇÃO TRADICIONAIS
#========================================================================

def switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, P):
    """
    Interface unificada para calcular a saída da camada oculta H para uma dada ativação.
    
    Suporta:
    - Ativações tradicionais: linear, sigmoid, sine, hardlim, tribas, radbas
    - Kernels morfológicos: erosion, dilation, fuzzy-*, bitwise-*
    
    Args:
        ActivationFunction: Identificador em string
        InputWeight: Matriz de pesos de entrada
        BiasofHiddenNeurons: Vetor de viés
        P: Matriz de dados de entrada
        
    Retorna:
        H: Saída da camada oculta
    """
    tempH = np.dot(InputWeight, P)
    BiasMatrix = BiasofHiddenNeurons.repeat(tempH.shape[1], axis=1)
    tempH = tempH + BiasMatrix
    
    if ActivationFunction in ('sig', 'sigmoid'):
        H = 1.0 / (1.0 + np.exp(-tempH))
    
    elif ActivationFunction in ('sin', 'sine', 'sinusoid', 'sinusoidal'):
        H = np.sin(tempH)
    
    elif ActivationFunction in ('hardlim',):
        H = np.array(tempH > 0, dtype=float)
    
    elif ActivationFunction in ('tribas',):
        H = np.maximum(1 - np.abs(tempH), 0)
    
    elif ActivationFunction in ('radbas',):
        H = np.exp(-np.square(tempH))
    
    elif ActivationFunction in ('linear', 'lin'):
        H = tempH
    
    # Kernels morfológicos e fuzzy
    elif ActivationFunction in ('erosion', 'ero'):
        H = erosion_kernel(InputWeight, BiasofHiddenNeurons, P)
    
    elif ActivationFunction in ('dilation', 'dil'):
        H = dilation_kernel(InputWeight, BiasofHiddenNeurons, P)
    
    elif ActivationFunction in ('fuzzy-erosion', 'fuzzy_erosion'):
        H = fuzzy_erosion_kernel(InputWeight, BiasofHiddenNeurons, P)
    
    elif ActivationFunction in ('fuzzy-dilation', 'fuzzy_dilation'):
        H = fuzzy_dilation_kernel(InputWeight, BiasofHiddenNeurons, P)
    
    elif ActivationFunction in ('bitwise-erosion', 'bitwise_erosion'):
        H = bitwise_erosion_kernel(InputWeight, BiasofHiddenNeurons, P)
    
    elif ActivationFunction in ('bitwise-dilation', 'bitwise_dilation'):
        H = bitwise_dilation_kernel(InputWeight, BiasofHiddenNeurons, P)
    
    else:
        raise ValueError(f"Função de ativação desconhecida: {ActivationFunction}")
    
    return H


#========================================================================
# ALGORITMO DE APRENDIZAGEM ELM
#========================================================================

def mElmLearning(train_data, test_data, Elm_Type, NumberofHiddenNeurons, 
                 ActivationFunction, execution, kfold, verbose, virusNorm=False,
                 threshold=None, leakage_threshold=0.999, audit_writer=None,
                 selected_writer=None, feature_names=None,
                 feature_selection='none', variance_threshold=None):
    """
    Treinamento e Teste da Extreme Learning Machine (ELM).
    
    Algoritmo central ELM:
    1. Pesos de entrada e vieses aleatórios
    2. Computa saída da camada oculta H
    3. Resolve pesos de saída via pseudo-inversa: β = H† T
    4. Prediz e avalia
    
    Args:
        train_data: Dataset de treinamento
        test_data: Dataset de teste
        Elm_Type: 0=regressão, 1=classificação
        NumberofHiddenNeurons: Número de neurônios ocultos
        ActivationFunction: Nome do kernel
        execution: Número do fold atual
        kfold: Número total de folds
        verbose: Ativa logs detalhados
        virusNorm: Aplica normalização virusNorm
        
    Retorna:
        TrainingAccuracy: Métrica de performance no treino
        TestingAccuracy: Métrica de performance no teste
        TrainingTime: Tempo de treinamento (segundos)
        TestingTime: Tempo de teste (segundos)
        cm_fold_train: Matriz de confusão de treino (apenas classificação)
        cm_fold_test: Matriz de confusão de teste (apenas classificação)
    """
    # Ajusta a seleção somente no treino e aplica a mesma máscara ao teste.
    train_data = np.asarray(train_data, dtype=float)
    test_data = np.asarray(test_data, dtype=float)
    y_train_raw = train_data[:, 0].copy()
    x_train_raw = train_data[:, 1:].copy()
    x_test_raw = test_data[:, 1:].copy()
    feature_selection = str(feature_selection or 'none').lower()
    suspicious_cols, constant_cols = set(), set()
    correlations, variances = {}, {}
    if feature_selection == 'none':
        selected_cols = list(range(x_train_raw.shape[1]))
    elif feature_selection == 'correlation':
        selected_cols, suspicious_cols, constant_cols, correlations = fit_correlation_pruning(
            y_train_raw, x_train_raw, threshold=threshold,
            leakage_threshold=leakage_threshold
        )
    elif feature_selection == 'variance':
        selected_cols, constant_cols, variances, _ = fit_variance_pruning(
            x_train_raw, variance_threshold
        )
    else:
        raise ValueError(f"Método de seleção de atributos desconhecido: {feature_selection}")

    selected_set = set(selected_cols)
    names = list(feature_names or [f"feature_{i + 1}" for i in range(x_train_raw.shape[1])])
    if len(names) != x_train_raw.shape[1]:
        names = [f"feature_{i + 1}" for i in range(x_train_raw.shape[1])]
    selection_threshold = threshold if feature_selection == 'correlation' else (
        variance_threshold if feature_selection == 'variance' else ''
    )
    for col in range(x_train_raw.shape[1]):
        if feature_selection == 'none':
            status = "used_no_pruning"
            score = ""
        elif col in suspicious_cols:
            status = "suspicious_removed"
            score = correlations.get(col, "")
        elif col in constant_cols:
            status = "constant"
            score = correlations.get(col, variances.get(col, ""))
        elif col in selected_set:
            status = "selected"
            score = correlations.get(col, variances.get(col, ""))
        else:
            status = "below_threshold"
            score = correlations.get(col, variances.get(col, ""))
        score_text = f"{score:.8f}" if isinstance(score, (int, float, np.floating)) else ""
        threshold_text = f"{selection_threshold:.8g}" if isinstance(selection_threshold, (int, float, np.floating)) else ""
        row = [ActivationFunction, int(NumberofHiddenNeurons), int(execution) + 1,
               col + 1, names[col], feature_selection, score_text, threshold_text, status]
        if audit_writer is not None:
            audit_writer.writerow(row)
        if selected_writer is not None and col in selected_set:
            selected_writer.writerow(row)

    train_data = np.column_stack((y_train_raw, x_train_raw[:, selected_cols]))
    test_data = np.column_stack((test_data[:, 0], x_test_raw[:, selected_cols]))
    [T, P] = loadingDataset(train_data)
    [TVT, TVP] = loadingDataset(test_data)
    
    NumberofTrainingData = np.size(P, 1)
    NumberofTestingData = np.size(TVP, 1)
    NumberofInputNeurons = np.size(P, 0)
    NumberofHiddenNeurons = int(NumberofHiddenNeurons)
    
    cm_fold_train, cm_fold_test = None, None
    
    # Classificação: Converte para codificação one-hot
    if Elm_Type != 0:
        sorted_target = np.sort(np.concatenate((T, TVT), axis=0))
        label = [sorted_target[0]]
        j = 0
        for i in range(1, NumberofTrainingData + NumberofTestingData):
            if sorted_target[i] != label[j]:
                j += 1
                label.append(sorted_target[i])
        
        number_class = j + 1
        if number_class != 2:
            raise ValueError(
                f"A avaliação binária requer exatamente 2 classes; encontradas: {label}."
            )
        NumberofOutputNeurons = number_class
        
        # Codifica alvos de treino
        temp_T = np.zeros((NumberofOutputNeurons, NumberofTrainingData))
        for i in range(NumberofTrainingData):
            for j in range(number_class):
                if label[j] == T[i]:
                    break
            temp_T[j][i] = 1
        T = temp_T  # ✓ CORRIGIDO: Encoding [0, 1] padrão (era * 2 - 1)
        
        # Codifica alvos de teste
        temp_TV_T = np.zeros((NumberofOutputNeurons, NumberofTestingData))
        for i in range(NumberofTestingData):
            for j in range(number_class):
                if label[j] == TVT[i]:
                    break
            temp_TV_T[j][i] = 1
        TVT = temp_TV_T  # ✓ CORRIGIDO: Encoding [0, 1] padrão (era * 2 - 1)
    
    # Fase de Treinamento
    start_time_train = process_time()
    
    # Inicializa pesos - tratamento especial para kernels morfológicos em regressão
    if Elm_Type == 0:
        if ActivationFunction in ('erosion', 'ero', 'dilation', 'dil',
                                  'fuzzy-erosion', 'fuzzy_erosion',
                                  'fuzzy-dilation', 'fuzzy_dilation',
                                  'bitwise-erosion', 'bitwise_erosion',
                                  'bitwise-dilation', 'bitwise_dilation'):
            # Kernels morfológicos: inicializa baseado no intervalo dos dados
            InputWeight = np.random.uniform(
                np.amin(P),
                np.amax(P),
                (NumberofHiddenNeurons, NumberofInputNeurons)
            )
        else:
            # Kernels tradicionais: inicialização padrão
            InputWeight = np.random.rand(NumberofHiddenNeurons, NumberofInputNeurons) * 2 - 1
    else:
        # Classificação: inicialização padrão
        InputWeight = np.random.rand(NumberofHiddenNeurons, NumberofInputNeurons) * 2 - 1
    
    # Aplica normalização virusNorm se solicitado
    if virusNorm:
        InputWeight = virusNormFunction(InputWeight, verbose)
        P = virusNormFunction(P, verbose)
        TVP = virusNormFunction(TVP, verbose)
    
    BiasofHiddenNeurons = np.random.rand(NumberofHiddenNeurons, 1)
    
    # Computa saída da camada oculta
    H = switchActivationFunction(ActivationFunction, InputWeight, 
                                 BiasofHiddenNeurons, P)
    
    # Resolve pesos de saída usando pseudo-inversa
    OutputWeight = np.dot(np.linalg.pinv(np.transpose(H)), np.transpose(T))
    
    end_time_train = process_time()
    TrainingTime = end_time_train - start_time_train
    
    # Predições de Treinamento
    Y = np.transpose(np.dot(np.transpose(H), OutputWeight))
    del H
    
    # Fase de Teste
    start_time_test = process_time()
    tempH_test = switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, TVP)
    del TVP
    TY = np.transpose(np.dot(np.transpose(tempH_test), OutputWeight))
    end_time_test = process_time()
    TestingTime = end_time_test - start_time_test
    
    # Calcula métricas de performance
    if Elm_Type == 0:
        # Regressão: RMSE
        TrainingAccuracy = round(np.sqrt(np.square(np.subtract(T, Y)).mean()), 6)
        TestingAccuracy = round(np.sqrt(np.square(np.subtract(TVT, TY)).mean()), 6)
    else:
        # Classificação: Acurácia e matriz de confusão
        label_index_train_expected = np.argmax(T, axis=0)
        label_index_train_actual = np.argmax(Y, axis=0)
        MissClassificationRate_Training = np.sum(label_index_train_actual != label_index_train_expected)
        TrainingAccuracy = round(1 - MissClassificationRate_Training / NumberofTrainingData, 6)
        
        label_index_test_expected = np.argmax(TVT, axis=0)
        label_index_test_actual = np.argmax(TY, axis=0)
        MissClassificationRate_Testing = np.sum(label_index_test_actual != label_index_test_expected)
        TestingAccuracy = round(1 - MissClassificationRate_Testing / NumberofTestingData, 6)
        
        labels_range = list(range(number_class))
        cm_fold_train = confusion_matrix(label_index_train_expected, label_index_train_actual, labels=labels_range)
        cm_fold_test = confusion_matrix(label_index_test_expected, label_index_test_actual, labels=labels_range)
        
        # ✓ NOVO: Calcular métricas de detecção para classificação binária
        if number_class == 2:
            train_metrics = calculate_detection_metrics(cm_fold_train)
            test_metrics = calculate_detection_metrics(cm_fold_test)
        else:
            train_metrics = None
            test_metrics = None
    
    # Saída Verbose
    if verbose:
        print(f'..................k: {execution}, k-fold: {kfold}............................')
        if Elm_Type == 0:
            print(f'Training RMSE: {TrainingAccuracy} ({Y.size} samples)')
            print(f'Testing  RMSE: {TestingAccuracy} ({TY.size} samples)')
        else:
            print(f'Training Accuracy: {TrainingAccuracy*100:.2f}%')
            print(f'Testing  Accuracy: {TestingAccuracy*100:.2f}%')
            
            # ✓ NOVO: Exibir métricas detalhadas de detecção para classificação binária
            if 'test_metrics' in locals() and test_metrics is not None:
                print_detection_metrics(train_metrics, "Training", verbose=True)
                print_detection_metrics(test_metrics, "Testing", verbose=True)
        
        print(f'Training Time: {round(TrainingTime, 2)} s')
        print(f'Testing  Time: {round(TestingTime, 2)} s')
    
    return TrainingAccuracy, TestingAccuracy, TrainingTime, TestingTime, cm_fold_train, cm_fold_test


#========================================================================
# CLASSE PRINCIPAL ELM
#========================================================================

class melm():
    """
    Classe principal mELM para detecção de malware e reconhecimento de padrões.
    
    Suporta dois modos:
    1. Validação cruzada K-fold (usando parâmetro -tall)
    2. Arquivos separados de treino/teste (usando parâmetros -tr e -ts)
    """
    
    def main(self, TrainingData_File, TestingData_File, AllData_File, Elm_Type, 
             NumberofHiddenNeurons, ActivationFunction, nSeed, kfold, sep, verbose,
             virusNorm=False, threshold=None, leakage_threshold=0.999,
             max_time=3600.0, force_all_kernels=False, output_dir=None, language='pt-br',
             feature_selection='none', variance_threshold=None):
        """
        Função principal de execução.
        
        Args:
            TrainingData_File: Caminho do arquivo de treino (ou None)
            TestingData_File: Caminho do arquivo de teste (ou None)
            AllData_File: Caminho do arquivo combinado (ou None)
            Elm_Type: 0=regressão, 1=classificação
            NumberofHiddenNeurons: Lista separada por vírgulas de contagem de neurônios
            ActivationFunction: Lista separada por vírgulas de kernels ou 'all'
            nSeed: Semente aleatória
            kfold: Número de folds para validação cruzada
            sep: Separador CSV
            verbose: Ativa logs detalhados
            virusNorm: Aplica normalização virusNorm
        """
        language = str(language).lower()
        if language not in {'pt-br', 'en'}:
            raise ValueError("language deve ser 'pt-br' ou 'en'.")
        requested_acts = str(ActivationFunction or 'linear')
        base_acts = ['sigmoid', 'sine', 'radbas', 'linear', 'hardlim', 'tribas',
                     'erosion', 'dilation', 'fuzzy-erosion', 'fuzzy-dilation',
                     'bitwise-erosion', 'bitwise-dilation']
        aliases = {
            'sig': 'sigmoid', 'sigmoid': 'sigmoid',
            'sin': 'sine', 'sine': 'sine',
            'sinusoid': 'sine', 'sinusoidal': 'sine', 'lin': 'linear',
            'ero': 'erosion', 'dil': 'dilation',
            'fuzzy_erosion': 'fuzzy-erosion', 'fuzzy_dilation': 'fuzzy-dilation',
            'bitwise_erosion': 'bitwise-erosion', 'bitwise_dilation': 'bitwise-dilation',
        }
        if requested_acts.strip().lower() == 'all':
            acts = base_acts
        else:
            acts = [aliases.get(name, name) for raw in requested_acts.split(',')
                    if (name := raw.strip().lower())]
            if not acts:
                raise ValueError('Informe ao menos uma ativação em -af.')
            unknown = [name for name in acts if name not in base_acts]
            if unknown:
                raise ValueError(f'Ativação desconhecida: {", ".join(unknown)}')
            acts = list(dict.fromkeys(acts))
        
        # Parse das contagens de neurônios ocultos
        nh_list = [int(v.strip()) for v in str(NumberofHiddenNeurons).split(',') if str(v).strip()]
        
        # Define semente aleatória
        if nSeed is None:
            nSeed = 42
        else:
            nSeed = int(nSeed)
        rnd_seed(nSeed)
        np.random.seed(nSeed)
        
        Elm_Type = int(Elm_Type)
        
        # Determina modo de execução
        use_separate_files = TrainingData_File is not None and TestingData_File is not None
        
        combo_results = []
        skipped_info = {}
        script_dir = os.path.dirname(os.path.abspath(__file__))
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = output_dir or os.path.join(script_dir, "results", f"melm_{run_stamp}")
        os.makedirs(result_dir, exist_ok=True)
        audit_path = os.path.join(result_dir, "melm_feature_audit.csv")
        selected_path = os.path.join(result_dir, "melm_feature_audit_selected.csv")
        audit_file = open(audit_path, "w", newline="", encoding="utf-8")
        selected_file = open(selected_path, "w", newline="", encoding="utf-8")
        audit_writer = csv.writer(audit_file)
        selected_writer = csv.writer(selected_file)
        audit_header = [
            "activation", "n_hidden", "fold", "feature_id", "feature_name",
            "selection_method", "selection_score", "selection_threshold", "status"
        ]
        audit_writer.writerow(audit_header)
        selected_writer.writerow(audit_header)
        
        if use_separate_files:
            # MODO 1: Arquivos de treino/teste separados (sem K-fold)
            if verbose:
                print("="*60)
                print("MODO: Arquivos Separados de Treino/Teste")
                print("="*60)
                print(f"Arquivo Treino: {TrainingData_File}")
                print(f"Arquivo Teste:  {TestingData_File}")
                print(f"Tipo Elm: {'Classificação' if Elm_Type == 1 else 'Regressão'}")
                print(f"virusNorm: {'Habilitado' if virusNorm else 'Desabilitado'}")
                print("="*60)
            
            train_data, _, feature_names = mElmStruct(TrainingData_File, Elm_Type, sep, verbose, True)
            test_data, _, test_feature_names = mElmStruct(
                TestingData_File, Elm_Type, sep, verbose, True, n_features=len(feature_names)
            )
            if feature_names != test_feature_names:
                raise ValueError("Os atributos/colunas de treino e teste não correspondem na mesma ordem.")
            class_labels = sorted(np.unique(np.concatenate((train_data[:, 0], test_data[:, 0]))).tolist())
            if int(Elm_Type) != 0 and len(class_labels) != 2:
                raise ValueError(f"A configuração atual de recomendações exige classificação binária; classes: {class_labels}.")
            
            for af in acts:
                for nh in nh_list:
                    if verbose:
                        print(f'\n=== Avaliando: Ativação={af}, Neurônios={nh} ===')
                    
                    TA, TeA, TT, Tt, cm_train, cm_test = mElmLearning(
                        train_data, test_data, Elm_Type, nh, af, 0, 1, verbose, virusNorm,
                        threshold, leakage_threshold, audit_writer, selected_writer, feature_names,
                        feature_selection, variance_threshold
                    )
                    
                    # Para classificação, métricas em porcentagem; para regressão, usa RMSE bruto
                    if Elm_Type == 1:
                        metric_train = TA * 100.0
                        metric_test = TeA * 100.0
                        std_train = 0.0
                        std_test = 0.0
                    else:
                        metric_train = float(TA)
                        metric_test = float(TeA)
                        std_train = 0.0
                        std_test = 0.0

                    combo_results.append({
                        "act": af,
                        "n_hidden": int(nh),
                        "accuracy_train": metric_train,
                        "std_train": std_train,
                        "accuracy_test": metric_test,
                        "std_test": std_test,
                        "time_train": float(TT),
                        "std_time_train": 0.0,
                        "time_test": float(Tt),
                        "std_time_test": 0.0,
                        "confusion_matrix_train": cm_train,
                        "confusion_matrix_test": cm_test,
                        "accuracy_train_per_fold": [metric_train],
                        "accuracy_test_per_fold": [metric_test],
                        "confusion_matrix_train_std": np.zeros_like(cm_train, dtype=float) if cm_train is not None else None,
                        "confusion_matrix_test_std": np.zeros_like(cm_test, dtype=float) if cm_test is not None else None
                    })
        
        else:
            # MODO 2: Validação cruzada K-fold
            if verbose:
                print("="*60)
                print("MODO: Validação Cruzada K-fold")
                print("="*60)
                print(f"Arquivo de Dados: {AllData_File}")
                print(f"K-folds: {kfold}")
                print(f"Tipo Elm: {'Classificação' if Elm_Type == 1 else 'Regressão'}")
                print(f"virusNorm: {'Habilitado' if virusNorm else 'Desabilitado'}")
                print("="*60)
            
            all_data, samples_index, feature_names = mElmStruct(AllData_File, Elm_Type, sep, verbose, True)
            class_labels = sorted(np.unique(all_data[:, 0]).tolist())
            if int(Elm_Type) != 0 and len(class_labels) != 2:
                raise ValueError(f"A configuração atual de recomendações exige classificação binária; classes: {class_labels}.")
            if int(Elm_Type) == 0:
                kf = KFold(n_splits=int(kfold), shuffle=True, random_state=nSeed)
            else:
                kf = StratifiedKFold(n_splits=int(kfold), shuffle=True, random_state=nSeed)
            
            n_samples = len(samples_index)
            print("\nEstimativa operacional mELM (heurística; os tempos reais são registrados por fold):")
            selected_acts = []
            n_features = max(0, all_data.shape[1] - 1)
            for candidate in acts:
                estimate = estimate_melm_kernel_time(n_samples, n_features, nh_list, kfold, candidate)
                if estimate > max_time and not force_all_kernels:
                    skipped_info[candidate] = {"estimated_seconds": estimate, "limit_seconds": max_time}
                    print(f"[estimativa] Pulando {candidate}: ~{format_time(estimate)} > limite {format_time(max_time)}")
                else:
                    selected_acts.append(candidate)
                    print(f"[estimativa] {candidate}: ~{format_time(estimate)}" + (" (forçado)" if estimate > max_time else ""))
            print(f"[estimativa] Total previsto para kernels elegíveis: ~{format_time(sum(estimate_melm_kernel_time(n_samples, n_features, nh_list, kfold, a) for a in selected_acts))}")
            if not selected_acts:
                raise ValueError("Todos os kernels foram pulados pela estimativa; aumente -max-time ou use --force-all-kernels.")
            acts = selected_acts
            
            # Inicia timer global
            global_start_time = time()
            
            for kernel_idx, af in enumerate(acts):
                for nh_idx, nh in enumerate(nh_list):
                    # ✓ CABEÇALHO INFORMATIVO
                    print('\n' + '='*80)
                    print(f'  🔹 TESTANDO: Kernel = {af} | Neurônios = {nh}')
                    print('='*80)
                    
                    acc_train, acc_test, t_train, t_test = [], [], [], []
                    cms_train, cms_test = [], []
                    
                    for i, (tr_idx, te_idx) in enumerate(kf.split(samples_index, all_data[samples_index, 0] if int(Elm_Type) != 0 else None)):
                        # ✓ INFORMAÇÃO DE PROGRESSO
                        if verbose:
                            print_progress_info(kernel_idx, len(acts), nh_idx, len(nh_list), 
                                              i, kfold, global_start_time)
                        
                        # ✓ INFORMAÇÃO DO FOLD ATUAL
                        if verbose:
                            print(f'  ▶ Fold {i+1}/{kfold} - Kernel: {af}, Neurônios: {nh}')
                        
                        train_data = all_data[samples_index[tr_idx], :]
                        test_data = all_data[samples_index[te_idx], :]
                        
                        TA, TeA, TT, Tt, cm_train, cm_test = mElmLearning(
                            train_data, test_data, Elm_Type, nh, af, i, kfold, verbose, virusNorm,
                            threshold, leakage_threshold, audit_writer, selected_writer, feature_names,
                            feature_selection, variance_threshold
                        )
                        
                        # ✓ RESUMO DO FOLD
                        if verbose:
                            if Elm_Type == 1:
                                print(f'  ✓ Fold {i+1} concluído: Train={TA*100:.2f}%, Test={TeA*100:.2f}%')
                            else:
                                print(f'  ✓ Fold {i+1} concluído: Train RMSE={TA:.4f}, Test RMSE={TeA:.4f}')
                        
                        acc_train.append(TA)
                        acc_test.append(TeA)
                        t_train.append(TT)
                        t_test.append(Tt)
                        
                        if cm_train is not None:
                            cms_train.append(cm_train.astype(float))
                        if cm_test is not None:
                            cms_test.append(cm_test.astype(float))
                    
                    # Agrega métricas através dos folds
                    if Elm_Type == 1:
                        # Classificação: reporta acurácia em porcentagem
                        mean_tr = np.mean(acc_train) * 100.0
                        std_tr = np.std(acc_train) * 100.0
                        mean_te = np.mean(acc_test) * 100.0
                        std_te = np.std(acc_test) * 100.0
                    else:
                        # Regressão: reporta RMSE sem escala percentual
                        mean_tr = float(np.mean(acc_train))
                        std_tr = float(np.std(acc_train))
                        mean_te = float(np.mean(acc_test))
                        std_te = float(np.std(acc_test))

                    mean_tt, std_tt = float(np.mean(t_train)), float(np.std(t_train))
                    mean_et, std_et = float(np.mean(t_test)), float(np.std(t_test))
                    
                    # ✓ RESUMO DA COMBINAÇÃO COMPLETA
                    print('\n' + '─'*80)
                    print(f'  📊 RESUMO: {af} com {nh} neurônios ({kfold} folds completos)')
                    print('─'*80)
                    if Elm_Type == 1:
                        print(f'  Train Accuracy: {mean_tr:.2f}% ± {std_tr:.2f}%')
                        print(f'  Test Accuracy:  {mean_te:.2f}% ± {std_te:.2f}%')
                        
                        # Métricas de detecção agregadas (se disponível)
                        if cms_test:
                            avg_cm_test = np.mean(cms_test, axis=0)
                            test_det_metrics = calculate_detection_metrics(avg_cm_test)
                            print(f'  Detection Rate: {test_det_metrics["detection_rate"]*100:.2f}%')
                            print(f'  False Positive Rate: {test_det_metrics["false_positive_rate"]*100:.2f}%')
                            print(f'  F1-Score: {test_det_metrics["f1_score"]*100:.2f}%')
                    else:
                        print(f'  Train RMSE: {mean_tr:.6f} ± {std_tr:.6f}')
                        print(f'  Test RMSE:  {mean_te:.6f} ± {std_te:.6f}')
                    print(f'  Tempo médio: Train={mean_tt:.4f}s, Test={mean_et:.4f}s')
                    print('─'*80 + '\n')
                    
                    # Prepara resultado atual para combo_results e diagnóstico
                    current_result = {
                        "act": af,
                        "n_hidden": int(nh),
                        "accuracy_train": mean_tr,
                        "std_train": std_tr,
                        "accuracy_test": mean_te,
                        "std_test": std_te,
                        "time_train": mean_tt,
                        "std_time_train": std_tt,
                        "time_test": mean_et,
                        "std_time_test": std_et,
                        "confusion_matrix_train": np.mean(cms_train, axis=0) if cms_train else None,
                        "confusion_matrix_test": np.mean(cms_test, axis=0) if cms_test else None,
                        "confusion_matrix_train_std": np.std(cms_train, axis=0) if cms_train else None,
                        "confusion_matrix_test_std": np.std(cms_test, axis=0) if cms_test else None,
                        "accuracy_train_per_fold": [float(v * 100.0) for v in acc_train] if Elm_Type == 1 else list(map(float, acc_train)),
                        "accuracy_test_per_fold": [float(v * 100.0) for v in acc_test] if Elm_Type == 1 else list(map(float, acc_test))
                    }
                    
                    # ✓ DIAGNÓSTICO AUTOMÁTICO
                    if Elm_Type == 1:  # Diagnósticos binários também entram no relatório não-verbose.
                        diagnosis = diagnose_kernel_performance(current_result, af)
                        if verbose:
                            print_diagnosis(diagnosis, af)
                        current_result['diagnosis'] = diagnosis
                    
                    combo_results.append(current_result)
        
        # Exibe resultados globais
        if verbose:
            print("\n" + "="*60)
            print("RESULTADOS GLOBAIS (mELM)")
            print("="*60)
        
        # Determina melhores/piores configurações de acordo com o tipo de tarefa
        if Elm_Type == 1:
            # Classificação: maior acurácia é melhor
            best_test = max(combo_results, key=lambda r: r['accuracy_test'])
            worst_test = min(combo_results, key=lambda r: r['accuracy_test'])
        else:
            # Regressão: menor erro (RMSE) é melhor
            best_test = min(combo_results, key=lambda r: r['accuracy_test'])
            worst_test = max(combo_results, key=lambda r: r['accuracy_test'])

        global_results = {"max": best_test, "min": worst_test, "elm_type": Elm_Type}
        
        if verbose:
            print(f"\nMelhor Configuração:")
            print(f"  Ativação: {best_test['act']}")
            print(f"  Neurônios Ocultos: {best_test['n_hidden']}")
            if Elm_Type == 1:
                print(f"  Acurácia Treino: {best_test['accuracy_train']:.2f}% ± {best_test['std_train']:.2f}%")
                print(f"  Acurácia Teste:  {best_test['accuracy_test']:.2f}% ± {best_test['std_test']:.2f}%")
            else:
                print(f"  RMSE Treino: {best_test['accuracy_train']:.6f} ± {best_test['std_train']:.6f}")
                print(f"  RMSE Teste: {best_test['accuracy_test']:.6f} ± {best_test['std_test']:.6f}")
            print(f"  Tempo Treino: {best_test['time_train']:.4f}s ± {best_test['std_time_train']:.4f}s")
            print(f"  Tempo Teste:  {best_test['time_test']:.4f}s ± {best_test['std_time_test']:.4f}s")
            
            print(f"\nPior Configuração:")
            print(f"  Ativação: {worst_test['act']}")
            print(f"  Neurônios Ocultos: {worst_test['n_hidden']}")
            if Elm_Type == 1:
                print(f"  Acurácia Treino: {worst_test['accuracy_train']:.2f}% ± {worst_test['std_train']:.2f}%")
                print(f"  Acurácia Teste:  {worst_test['accuracy_test']:.2f}% ± {worst_test['std_test']:.2f}%")
            else:
                print(f"  RMSE Treino: {worst_test['accuracy_train']:.6f} ± {worst_test['std_train']:.6f}")
                print(f"  RMSE Teste: {worst_test['accuracy_test']:.6f} ± {worst_test['std_test']:.6f}")
            print(f"  Tempo Treino: {worst_test['time_train']:.4f}s ± {worst_test['std_time_train']:.4f}s")
            print(f"  Tempo Teste:  {worst_test['time_test']:.4f}s ± {worst_test['std_time_test']:.4f}s")
        
        # Prepara resultados por função de ativação
        act_results = {}
        for act in acts:
            results_for_act = [r for r in combo_results if r['act'] == act]
            if not results_for_act:
                continue
            if Elm_Type == 1:
                best_for_act = max(results_for_act, key=lambda r: r['accuracy_test'])
                worst_for_act = min(results_for_act, key=lambda r: r['accuracy_test'])
            else:
                best_for_act = min(results_for_act, key=lambda r: r['accuracy_test'])
                worst_for_act = max(results_for_act, key=lambda r: r['accuracy_test'])
            act_results[act] = {
                "max_test": best_for_act,
                "min_test": worst_for_act,
            }
        
        # O diagnóstico do relatório deve corresponder ao cenário exibido em cada ativação.
        diagnostics_dict = {}
        if Elm_Type == 1:
            for kernel_name, scenarios in act_results.items():
                best_scenario = scenarios['max_test']
                diagnostics_dict[kernel_name] = diagnose_kernel_performance(best_scenario, kernel_name)
        
        # Gera relatório HTML
        audit_file.close()
        selected_file.close()
        report_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_path = os.path.join(result_dir, f"elm_report_{'en_' if language == 'en' else ''}{report_stamp}.html")
        report_config = {
            "dataset": os.path.basename(AllData_File or TrainingData_File or "dataset"),
            "validation": "K-fold" if not use_separate_files else "separate train/test",
            "kfold": int(kfold) if not use_separate_files else 1,
            "seed": int(nSeed),
            "neurons": nh_list,
            "requested_activations": requested_acts,
            "executed_activations": list(acts),
            "feature_selection": feature_selection,
            "correlation_threshold": threshold if feature_selection == "correlation" else None,
            "variance_threshold": variance_threshold if feature_selection == "variance" else None,
            "leakage_threshold": leakage_threshold if feature_selection == "correlation" else None,
            "max_time": max_time,
            "force_all_kernels": bool(force_all_kernels),
            "virus_norm": bool(virusNorm),
            "separator": sep or "auto",
            "task": "classification" if Elm_Type == 1 else "regression",
        }
        generate_html_report_elm(
            global_results, act_results, diagnostics_dict, output_file=report_path,
            class_labels=class_labels, skipped_info=skipped_info,
            language=language, report_config=report_config
        )


#========================================================================
# GERAÇÃO DE RELATÓRIO HTML
#========================================================================

def plot_melm_best_boxplots(act_results, output_dir, is_classification=True, language='pt-br'):
    """Exporta PNGs e prepara boxplots em linhas responsivas para o HTML."""
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    series = {}
    for key, field in (("train", "accuracy_train_per_fold"), ("test", "accuracy_test_per_fold")):
        rows = []
        for name, data in act_results.items():
            best = data.get("max_test") or {}
            values = np.asarray(best.get(field, []), dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                rows.append((name, best["n_hidden"], values))
        series[key] = rows

    all_values = [values for rows in series.values() for _, _, values in rows]
    if not all_values:
        return {}
    combined = np.concatenate(all_values)
    minimum, maximum = float(np.min(combined)), float(np.max(combined))
    padding = max((maximum - minimum) * 0.07, 2.5 if is_classification else 0.1)
    axis_min, axis_max = minimum - padding, maximum + padding
    if is_classification:
        axis_min, axis_max = max(0.0, axis_min), min(100.0, axis_max)
    if axis_max <= axis_min:
        axis_max = axis_min + 1.0
    axis_span = axis_max - axis_min
    tick_values = np.linspace(axis_min, axis_max, 5)
    tick_labels = [(f"{value:.1f}%" if is_classification else f"{value:.3g}")
                   for value in tick_values]
    titles = {"train": "Training" if language == "en" else "Treino",
              "test": "Test" if language == "en" else "Teste"}
    palette = LinearSegmentedColormap.from_list(
        "melm_boxplots", ["#7778D8", "#A59ADE", "#B8C2C9", "#82B8C8", "#59D9E7", "#76D8F2"]
    )

    def position(value):
        return round(1000.0 * (float(value) - axis_min) / axis_span, 2)

    plots = {}
    os.makedirs(output_dir, exist_ok=True)
    for key, items in series.items():
        if not items:
            continue
        count = len(items)
        rows = []
        fig, ax = plt.subplots(figsize=(10.2, max(3.5, 0.45 * count + 1.6)))
        boxes = ax.boxplot(
            [values for _, _, values in items],
            tick_labels=[f"{name} (h={hidden})" for name, hidden, _ in items],
            orientation="horizontal", patch_artist=True,
            medianprops={"color": "#222222", "linewidth": 1.8},
            boxprops={"edgecolor": "#444444"}, whiskerprops={"color": "#444444"},
            capprops={"color": "#444444"},
            flierprops={"marker": "+", "markeredgecolor": "#b33a3a"},
        )
        for index, (name, hidden, values) in enumerate(items):
            color = to_hex(palette(index / max(1, count - 1)))
            boxes["boxes"][index].set_facecolor(color)
            boxes["boxes"][index].set_alpha(0.86)
            q1, median, q3 = [float(value) for value in np.percentile(values, [25, 50, 75])]
            iqr = q3 - q1
            lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            whisker_low = float(np.min(values[values >= lower_fence]))
            whisker_high = float(np.max(values[values <= upper_fence]))
            outliers = values[(values < whisker_low) | (values > whisker_high)]
            rows.append({
                "name": name, "hidden": hidden, "color": color,
                "low_x": position(whisker_low), "high_x": position(whisker_high),
                "q1_x": position(q1), "median_x": position(median), "q3_x": position(q3),
                "box_width": max(2.0, position(q3) - position(q1)),
                "outliers_x": [position(value) for value in outliers],
            })
        ax.set_title(titles[key])
        ax.set_xlabel(("Accuracy (%)" if language == "en" else "Acurácia (%)")
                      if is_classification else "RMSE")
        ax.set_xlim(axis_min, axis_max)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=9)
        fig.subplots_adjust(left=0.32, right=0.98, top=0.94, bottom=0.1)
        filename = f"best_{key}_by_activation.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots[key] = {"filename": filename, "items": rows, "ticks": tick_labels}
    return plots


def generate_html_report_elm(global_results, act_results, diagnostics=None,
                             output_file='elm_report.html', class_labels=None,
                             skipped_info=None, language='pt-br', report_config=None):
    """Renderiza o dashboard mELM com o tema, seções e seleção de idioma do SVM-new."""
    language = str(language).lower()
    is_english = language == "en"
    elm_type = int(global_results.get("elm_type", 1))
    is_classification = elm_type == 1
    config = report_config or {}
    is_cross_validation = config.get("validation") == "K-fold" and int(config.get("kfold", 1)) > 1
    labels = {
        "title": "mELM - Avaliação de Parâmetros" if not is_english else "mELM - Parameter Evaluation",
        "subtitle": "Resultados de desempenho, estabilidade e custo computacional" if not is_english else "Performance, stability, and computational cost results",
        "config": "Configuração Experimental" if not is_english else "Experimental Configuration",
        "note": "Esta seção registra os parâmetros usados nesta execução." if not is_english else "This section records the parameters used in this run.",
        "dataset": "Dataset", "validation": "Validação" if not is_english else "Validation",
        "seed": "Semente" if not is_english else "Seed",
        "executed": "Kernels/ativações executados" if not is_english else "Kernels/activations run",
        "requested": "Solicitados" if not is_english else "Requested",
        "skipped": "Pulados pela estimativa" if not is_english else "Skipped by estimate",
        "neurons": "Neurônios ocultos" if not is_english else "Hidden neurons",
        "task": "Tarefa" if not is_english else "Task",
        "controls": "Controles metodológicos" if not is_english else "Methodological controls",
        "pruning_applied": "Poda de atributos" if not is_english else "Feature pruning",
        "pruning_method": "Método de poda" if not is_english else "Pruning method",
        "threshold": "Limiar de correlação |r|" if not is_english else "Correlation threshold |r|",
        "variance_threshold": "Fração da variância máxima" if not is_english else "Fraction of maximum variance",
        "leakage": "Limite de correlação suspeita" if not is_english else "Suspicious-correlation threshold",
        "max_time": "Limite estimado" if not is_english else "Estimated time limit",
        "force": "Forçar todas as ativações" if not is_english else "Force all activations",
        "virus_norm": "virusNorm", "generated": "Gerado em" if not is_english else "Generated at",
        "best": "Melhor resultado global" if not is_english else "Best global result",
        "worst": "Pior resultado global" if not is_english else "Worst global result",
        "activation": "Ativação/kernel" if not is_english else "Activation/kernel",
        "configuration": "Configuração" if not is_english else "Configuration",
        "train_metric": ("Acurácia de treino" if not is_english else "Training accuracy") if is_classification else ("RMSE de treino" if not is_english else "Training RMSE"),
        "test_metric": ("Acurácia de teste" if not is_english else "Test accuracy") if is_classification else ("RMSE de teste" if not is_english else "Test RMSE"),
        "train_time": "Tempo de treino" if not is_english else "Training time",
        "test_time": "Tempo de teste" if not is_english else "Test time",
        "fold_box": "Distribuição dos melhores resultados por fold" if not is_english else "Fold distribution of best results",
        "box_note": "Cada caixa usa os folds da configuração de neurônios com melhor média de validação em cada ativação. A escolha foi feita com estes mesmos folds; interprete os gráficos como comparação exploratória." if not is_english else "Each box uses the folds of the neuron configuration with the best mean validation result for that activation. The same folds were used for selection, so these plots are exploratory.",
        "summary": "Resumo por kernel/função de ativação" if not is_english else "Kernel/activation summary",
        "best_scenario": "Melhor cenário" if not is_english else "Best scenario",
        "worst_scenario": "Pior cenário" if not is_english else "Worst scenario",
        "confusion": "Matriz de confusão média (melhor cenário)" if not is_english else "Mean confusion matrix (best scenario)",
        "confusion_worst": "Matriz de confusão média (pior cenário)" if not is_english else "Mean confusion matrix (worst scenario)",
        "diagnostics": "Diagnósticos automáticos" if not is_english else "Automatic diagnostics",
        "diagnostics_note": "Cada diagnóstico corresponde ao melhor cenário exibido para a ativação. As regras são heurísticas e têm prioridade: recall positivo < 1%, desvio da acurácia > 15 p.p. (quando há vários folds), FPR > 15% e acurácia < 80%. Apenas o primeiro alerta é mostrado; nenhum limiar constitui teste de significância." if not is_english else "Each diagnosis refers to the displayed best scenario for its activation. Rules are heuristic and prioritized: positive recall < 1%, accuracy standard deviation > 15 percentage points (with multiple folds), FPR > 15%, and accuracy < 80%. Only the first alert is shown; these thresholds are not significance tests.",
        "problem": "Problema" if not is_english else "Issue",
        "explanation": "Explicação" if not is_english else "Explanation",
        "recommendation": "Recomendação" if not is_english else "Recommendation",
        "audit": "Auditoria da poda" if not is_english else "Pruning audit",
        "audit_note": "Quando há seleção, o critério é ajustado apenas no treino de cada fold e aplicado à validação; no modo sem poda, todos os atributos são usados." if not is_english else "When selection is enabled, the criterion is fitted only on each fold's training data and applied to validation; no-pruning mode uses all features.",
        "audit_all": "Auditoria completa por fold" if not is_english else "Complete per-fold audit",
        "audit_selected": "Features selecionadas por fold" if not is_english else "Features selected per fold",
        "estimate": "Estimativa" if not is_english else "Estimate",
        "limit": "Limite" if not is_english else "Limit",
        "no_result": "Não executado" if not is_english else "Not run",
        "none": "Nenhuma" if not is_english else "None",
        "yes": "Sim" if not is_english else "Yes",
        "no": "Não" if not is_english else "No",
        "report": "Relatório de Resultados" if not is_english else "Results Report",
        "report_sections": "Seções do relatório" if not is_english else "Report sections",
        "overview": "Visão geral" if not is_english else "Overview",
        "distributions": "Distribuições" if not is_english else "Distributions",
        "choose_activation": "Escolha uma ativação" if not is_english else "Choose an activation",
        "train_box": "Distribuição de treino" if not is_english else "Training distribution",
        "test_box": "Distribuição de teste" if not is_english else "Test distribution",
        "box_axis": ("Acurácia (%)" if not is_english else "Accuracy (%)") if is_classification else "RMSE",
        "box_legend": "Caixa: Q1–Q3; linha interna: mediana; hastes: valores observados dentro de 1,5 × IQR; pontos vermelhos: valores fora desse intervalo." if not is_english else "Box: Q1–Q3; inner line: median; whiskers: observed values within 1.5 × IQR; red points: values outside that range.",
        "download_png": "Baixar PNG" if not is_english else "Download PNG",
        "observed_metrics": "Indicadores observados" if not is_english else "Observed metrics",
    }
    tips = {
        "dataset": "Arquivo de dados usado nesta execução." if not is_english else "Data file used for this run.",
        "validation": (
            ("Validação cruzada estratificada na classificação, com métricas resumidas por fold." if is_classification else "Validação cruzada K-fold na regressão, com métricas resumidas por fold.")
            if not is_english else
            ("Stratified cross-validation for classification, with metrics summarized across folds." if is_classification else "K-fold cross-validation for regression, with metrics summarized across folds.")
        ) if is_cross_validation else ("Arquivos de treino e teste separados; não há validação cruzada nesta execução." if not is_english else "Separate training and test files; no cross-validation in this run."),
        "seed": "Semente usada para reproduzir a divisão dos dados e a inicialização aleatória." if not is_english else "Seed used to reproduce the data split and random initialization.",
        "task": "Tipo de problema determinado pelo parâmetro -ty." if not is_english else "Problem type selected by the -ty parameter.",
        "pruning_applied": "Indica se um método de seleção de atributos foi habilitado; a quantidade efetivamente removida pode variar por fold." if not is_english else "Shows whether a feature selection method was enabled; the number actually removed may vary by fold.",
        "pruning_method": (
            "Sem poda, todos os atributos são usados. Com poda, o critério é ajustado no treino de cada fold e aplicado à validação."
            if is_cross_validation else
            "Sem poda, todos os atributos são usados. Com poda, o critério é ajustado no arquivo de treino e aplicado ao arquivo de teste."
        ) if not is_english else (
            "Without pruning, all features are used. With pruning, the criterion is fitted on each fold's training data and applied to validation."
            if is_cross_validation else
            "Without pruning, all features are used. With pruning, the criterion is fitted on the training file and applied to the test file."
        ),
        "executed": "Funções que geraram resultados nesta execução. Aqui, 'kernel' designa a transformação da camada oculta, não um kernel SVM." if not is_english else "Functions that produced results in this run. Here, 'kernel' means a hidden-layer transformation, not an SVM kernel.",
        "generated": "Data e hora em que este relatório foi produzido." if not is_english else "Date and time when this report was produced.",
        "requested": "Funções de ativação pedidas na linha de comando; 'all' solicita todas as disponíveis." if not is_english else "Activation functions requested on the command line; 'all' requests every available function.",
        "skipped": "Funções não executadas porque a estimativa operacional excedeu o limite; a previsão não é uma medição de tempo." if not is_english else "Functions not run because the operational estimate exceeded the limit; the prediction is not measured runtime.",
        "neurons": "Quantidades de neurônios na camada oculta avaliadas para cada função executada." if not is_english else "Hidden-layer neuron counts evaluated for each function run.",
        "threshold": "Na poda por correlação, atributos com |r| de Pearson abaixo deste valor são descartados, salvo o caso de nenhum atributo restar." if not is_english else "With correlation pruning, features below this absolute Pearson r are discarded, except when that would leave no features.",
        "variance_threshold": "Na poda por variância, mantém atributos cuja variância após escala min-max no treino atinge esta fração da maior variância; o modelo recebe os valores originais." if not is_english else "With variance pruning, keeps features whose training-set min-max scaled variance reaches this fraction of the largest variance; the model receives original values.",
        "leakage": "Na poda por correlação, atributos com |r| igual ou superior a este limiar são removidos como suspeitos de vazamento; correlação alta, isoladamente, não comprova vazamento." if not is_english else "With correlation pruning, features at or above this absolute r are removed as possible leakage; high correlation alone does not prove leakage.",
        "max_time": "Limite por função para a estimativa heurística de tempo. A estimativa depende de uma constante operacional e pode diferir do tempo real." if not is_english else "Per-function limit for the heuristic time estimate. The estimate uses an operational constant and may differ from actual runtime.",
        "force": "Quando habilitado, ignora o filtro por tempo estimado e tenta executar todas as funções solicitadas." if not is_english else "When enabled, bypasses the estimated-time filter and attempts all requested functions.",
        "virus_norm": "Quando habilitado, aplica a transformação virusNorm a pesos e dados de treino/teste após a seleção de atributos." if not is_english else "When enabled, applies virusNorm to weights and training/test data after feature selection.",
        "controls": "A semente registra o estado aleatório da execução; a poda, quando ativa, usa apenas dados de treino no ajuste." if not is_english else "The seed records the run's random state; pruning, when enabled, is fitted using training data only.",
    }
    metric_unit = "%" if is_classification else ""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.abspath(output_file) if os.path.isabs(output_file) else os.path.join(script_dir, output_file)
    output_dir = os.path.dirname(output_path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    image_dir_name = f"elm_report_images_{'en_' if is_english else ''}{stamp}"
    img_dir = os.path.join(output_dir, image_dir_name)
    os.makedirs(img_dir, exist_ok=True)

    def img_href(name):
        return f"{image_dir_name}/{name}"

    def safe_cm(result, title, filename):
        cm = result.get("confusion_matrix_test")
        if cm is None or not np.asarray(cm).size or np.asarray(cm).sum() <= 0:
            return None
        plot_and_save_cm(cm, title, os.path.join(img_dir, filename),
                         cm_std=result.get("confusion_matrix_test_std"),
                         class_labels=class_labels, language=language)
        return img_href(filename)

    best = global_results["max"]
    worst = global_results["min"]
    best["act"] = best.get("act", "—")
    worst["act"] = worst.get("act", "—")
    best_cm = worst_cm = None
    if is_classification:
        best_cm = safe_cm(best, labels["confusion"], "cm_global_best.png")
        worst_cm = safe_cm(worst, labels["confusion_worst"], "cm_global_worst.png")

    act_cm = {}
    for act_name, data in act_results.items():
        act_cm[act_name] = {"best": None, "worst": None}
        if is_classification and data.get("max_test"):
            safe_key = re.sub(r"[^A-Za-z0-9_-]+", "_", str(act_name))
            act_cm[act_name]["best"] = safe_cm(data["max_test"], f"{labels['confusion']} — {act_name}", f"cm_{safe_key}_best.png")
            if data.get("min_test") is not None:
                act_cm[act_name]["worst"] = safe_cm(data["min_test"], f"{labels['confusion_worst']} — {act_name}", f"cm_{safe_key}_worst.png")

    boxplots = (plot_melm_best_boxplots(act_results, img_dir, is_classification, language=language)
                if is_cross_validation else {})
    for plot in boxplots.values():
        plot["href"] = img_href(plot["filename"])
    config_view = dict(config)
    config_view["dataset"] = os.path.basename(str(config.get("dataset", "dataset")))
    config_view["validation"] = ("K-fold" if not is_english else "K-fold") if config.get("validation") == "K-fold" else ("Treino/teste separado" if not is_english else "Separate train/test")
    config_view["task"] = ("Classificação binária" if not is_english else "Binary classification") if is_classification else ("Regressão" if not is_english else "Regression")
    config_view["requested_activations"] = config.get("requested_activations", "—")
    config_view["executed_activations"] = ", ".join(config.get("executed_activations", [])) or labels["no_result"]
    config_view["skipped_activations"] = ", ".join(skipped_info.keys()) if skipped_info else labels["none"]
    config_view["neurons"] = ", ".join(map(str, config.get("neurons", []))) or "—"
    selection_method = config.get("feature_selection", "none")
    config_view["pruning_applied"] = labels["no"] if selection_method == "none" else labels["yes"]
    config_view["feature_selection"] = {
        "none": "Nenhuma — todos os atributos" if not is_english else "None — all features",
        "correlation": "Correlação de Pearson (fold-safe)" if not is_english else "Pearson correlation (fold-safe)",
        "variance": "Variância relativa min-max (fold-safe)" if not is_english else "Relative min-max variance (fold-safe)",
    }.get(selection_method, selection_method)
    correlation_threshold = config.get("correlation_threshold")
    variance_threshold = config.get("variance_threshold")
    leakage_threshold = config.get("leakage_threshold")
    not_applied = "Não aplicado" if not is_english else "Not applied"
    config_view["threshold"] = correlation_threshold if correlation_threshold is not None else not_applied
    config_view["variance_threshold"] = variance_threshold if variance_threshold is not None else not_applied
    config_view["leakage_threshold"] = leakage_threshold if leakage_threshold is not None else not_applied
    config_view["virus_norm"] = labels["yes"] if config.get("virus_norm") else labels["no"]
    config_view["force_all"] = labels["yes"] if config.get("force_all_kernels") else labels["no"]
    config_view["generated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S" if is_english else "%d/%m/%Y %H:%M:%S")

    localized_diagnostics = {}
    if is_classification:
        for name, result in act_results.items():
            item = result.get("max_test")
            if not item:
                continue
            diag = (diagnostics or {}).get(name) or diagnose_kernel_performance(item, name)
            status = diag.get("status", "unknown")
            cm = item.get("confusion_matrix_test")
            recall = fpr = None
            if cm is not None and np.asarray(cm).shape == (2, 2):
                tn, fp, fn, tp = np.asarray(cm, dtype=float).ravel()
                recall = tp / (tp + fn) if tp + fn else None
                fpr = fp / (fp + tn) if fp + tn else None
            std = float(item.get("std_test", 0.0))
            accuracy = float(item.get("accuracy_test", 0.0))
            if is_english:
                problem = {
                    "critical": "Low positive-class recall", "unstable": "High variation across folds",
                    "warning": "High false-positive rate", "poor": "Accuracy below the triage threshold",
                    "good": "No heuristic alert",
                }.get(status, "No diagnosis available")
                explanation = {
                    "critical": f"Positive-class recall was {recall*100:.2f}%, below the 1% triage threshold." if recall is not None else "Positive-class recall could not be estimated from this confusion matrix.",
                    "unstable": f"Accuracy standard deviation across folds was {std:.2f} percentage points, above the 15-point triage threshold.",
                    "warning": f"The false-positive rate was {fpr*100:.2f}%, above the 15% triage threshold." if fpr is not None else "The false-positive rate could not be estimated from this confusion matrix.",
                    "poor": f"Accuracy was {accuracy:.2f}%, below the 80% triage threshold.",
                    "good": "None of the report's heuristic thresholds was crossed. This does not establish generalization or statistical superiority.",
                }.get(status, "")
                recommendation = {
                    "critical": "Inspect class support and confusion matrices, then review feature selection and scaling. Compare with a simple baseline.",
                    "unstable": "Inspect results and class balance by fold; repeat the analysis with prespecified seeds or an independent dataset.",
                    "warning": "Assess the practical cost of false alarms and inspect confusion matrices. Tune any decision rule using training or validation data.",
                    "poor": "Compare with a simple baseline and review recall and false-positive rate before changing the activation or neuron count.",
                    "good": "Record the configuration and evaluate it on independent data before making a final performance claim.",
                }.get(status, "")
            else:
                problem = {
                    "critical": "Recall baixo da classe positiva", "unstable": "Alta variação entre folds",
                    "warning": "Taxa elevada de falsos positivos", "poor": "Acurácia abaixo do limiar de triagem",
                    "good": "Sem alerta heurístico",
                }.get(status, "Sem diagnóstico disponível")
                explanation = {
                    "critical": f"O recall da classe positiva foi {recall*100:.2f}%, abaixo do limiar de triagem de 1%." if recall is not None else "Não foi possível estimar o recall positivo a partir desta matriz de confusão.",
                    "unstable": f"O desvio padrão da acurácia entre folds foi {std:.2f} pontos percentuais, acima do limiar de 15 pontos.",
                    "warning": f"A taxa de falsos positivos foi {fpr*100:.2f}%, acima do limiar de triagem de 15%." if fpr is not None else "Não foi possível estimar a taxa de falsos positivos a partir desta matriz de confusão.",
                    "poor": f"A acurácia foi {accuracy:.2f}%, abaixo do limiar de triagem de 80%.",
                    "good": "Nenhum limiar heurístico do relatório foi ultrapassado. Isso não demonstra generalização nem superioridade estatística.",
                }.get(status, "")
                recommendation = {
                    "critical": "Inspecione o suporte das classes e as matrizes de confusão; depois revise a seleção e a escala dos atributos. Compare com um baseline simples.",
                    "unstable": "Inspecione resultados e proporções de classes por fold; repita a análise com sementes predefinidas ou outro conjunto independente.",
                    "warning": "Avalie o custo prático dos alarmes falsos e examine as matrizes de confusão. Ajuste qualquer regra de decisão usando treino ou validação.",
                    "poor": "Compare com um baseline simples e examine recall e taxa de falsos positivos antes de mudar ativação ou número de neurônios.",
                    "good": "Registre a configuração e avalie-a em dados independentes antes de concluir sobre o desempenho final.",
                }.get(status, "")
            evidence = [f"{labels['test_metric']}: {accuracy:.2f}%"]
            if is_cross_validation:
                evidence.append((f"Desvio entre folds: {std:.2f} p.p." if not is_english
                                 else f"Across-fold standard deviation: {std:.2f} percentage points"))
            if recall is not None:
                evidence.append((f"Recall positivo: {recall*100:.2f}%" if not is_english
                                 else f"Positive-class recall: {recall*100:.2f}%"))
            else:
                evidence.append("Recall positivo: indisponível (sem positivos na matriz)" if not is_english
                                else "Positive-class recall: unavailable (no positive cases in matrix)")
            if fpr is not None:
                evidence.append((f"Taxa de falsos positivos: {fpr*100:.2f}%" if not is_english
                                 else f"False-positive rate: {fpr*100:.2f}%"))
            else:
                evidence.append("Taxa de falsos positivos: indisponível (sem negativos na matriz)" if not is_english
                                else "False-positive rate: unavailable (no negative cases in matrix)")
            localized_diagnostics[name] = {
                "status": status, "problem": problem, "explanation": explanation,
                "recommendation": recommendation, "evidence": evidence,
                "n_hidden": item.get("n_hidden"),
            }

    # Same stylesheet and visual language as SVM-new.
    assets_dir = os.path.join(script_dir, "static")
    css_path = os.path.relpath(os.path.join(assets_dir, "melm_report_styles.css"), output_dir).replace("\\", "/")
    logo_dir = os.path.join(script_dir, "templates")
    logo_upe_path = os.path.relpath(os.path.join(logo_dir, "logo_upe.png"), output_dir).replace("\\", "/")
    logo_ufpe_path = os.path.relpath(os.path.join(logo_dir, "logo_ufpe.png"), output_dir).replace("\\", "/")
    template_dir = os.path.join(script_dir, "templates")
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=select_autoescape(["html", "xml"]))
    template = env.get_template("melm_report_template.html.j2")
    rendered = template.render(
        language="en" if is_english else "pt-br", labels=labels, tips=tips,
        global_results=global_results, act_results=act_results, act_cm=act_cm,
        cm_global_best=best_cm, cm_global_worst=worst_cm,
        boxplots=boxplots,
        skipped_info=skipped_info or {}, diagnostics=localized_diagnostics,
        config=config_view, metric_unit=metric_unit, is_classification=is_classification,
        css_path=css_path, logo_upe_path=logo_upe_path, logo_ufpe_path=logo_ufpe_path,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)
    print(f"{'HTML report generated' if is_english else 'Relatório HTML gerado'}: {output_path}")

#========================================================================
# INTERFACE DE LINHA DE COMANDO (CLI)
#========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='mELM - Máquina de Aprendizado Extremo Morfológico para classificação binária e regressão em dados tabulares ou LIBSVM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Validação cruzada K-fold com kernels morfológicos
  python melmParameters.py -tall dataset.csv -ty 1 -nh 100 -af erosion,dilation -kfold 10 -v
  
  # Treino/teste separados com normalização virusNorm
  python melmParameters.py -tr train.csv -ts test.csv -ty 1 -virusNorm -nh 100 -af dilation -v
  
  # Testar todas as ativações com limite por estimativa
  python melmParameters.py -tall dados.csv -ty 1 -nh 50,100,200 -af all -kfold 5 -max-time 3600
  
  # Tarefa de regressão
  python melmParameters.py -tr train.csv -ts test.csv -ty 0 -nh 100 -af linear -v

Funções de Ativação:
  Tradicionais: linear, sigmoid, sine, hardlim, tribas, radbas
  Aliases compatíveis: sig, sin, sinusoid, sinusoidal
  Morfológicas: erosion, dilation, fuzzy-erosion, fuzzy-dilation, bitwise-erosion, bitwise-dilation
  Use 'all' para executar as 12 ativações-base.
        """
    )
    
    # Opções de entrada de dados (modos mutuamente exclusivos)
    data_group = parser.add_argument_group('Entrada de Dados (escolha um modo)')
    data_group.add_argument('-tall', '--AllData_File', dest='AllData_File', action='store',
                           help='Arquivo com todos os dados para validação cruzada K-fold')
    data_group.add_argument('-tr', '--TrainingData_File', dest='TrainingData_File', action='store',
                           help='Arquivo de dados de treinamento (use com -ts)')
    data_group.add_argument('-ts', '--TestingData_File', dest='TestingData_File', action='store',
                           help='Arquivo de dados de teste (use com -tr)')
    
    # Configuração do modelo
    config_group = parser.add_argument_group('Configuração do Modelo')
    config_group.add_argument('-ty', '--Elm_Type', dest='Elm_Type', action='store', required=True,
                             help='Tipo de tarefa: 0=regressão, 1=classificação')
    config_group.add_argument('-nh', '--nHiddenNeurons', dest='nHiddenNeurons', action='store', required=True,
                             help="Número de neurônios ocultos (lista separada por vírgula, ex: '50,100,200')")
    config_group.add_argument('-af', '--ActivationFunction', dest='ActivationFunction', action='store', required=True,
                             help="Função de ativação (lista separada por vírgula ou 'all')")
    
    # Parâmetros opcionais
    optional_group = parser.add_argument_group('Parâmetros Opcionais')
    optional_group.add_argument('-virusNorm', dest='virusNorm', action='store_true',
                               help='Aplica normalização virusNorm [0.1, 0.9] para características de malware')
    optional_group.add_argument('-kfold', dest='kfold', action='store', type=int, default=5,
                               help='Número de folds para validação cruzada (padrão: 5)')
    optional_group.add_argument(
        '-sep',
        dest='sep',
        action='store',
        default=None,
        help=(
            "Separador do CSV. Exemplos:\n"
            "  -sep \",\"   (vírgula)\n"
            "  -sep \";\"   (ponto e vírgula)\n"
            "  -sep tab    (TAB)\n"
            "  -sep pipe   (|)\n"
            "  -sep auto   (padrão — autodetecta entre ',', ';', TAB, '|', ':', '^', '~', ' ')"
        )
    )
    optional_group.add_argument('-sd', '--seed', dest='nSeed', action='store', type=int, default=42,
                               help='Semente aleatória para reprodutibilidade')
    optional_group.add_argument('--feature-selection', choices=('none', 'correlation', 'variance'), default=None,
                                help='Método de seleção: none (padrão), correlation ou variance. Informar um threshold legado também seleciona o método correspondente.')
    optional_group.add_argument('-threshold', '--correlation-threshold', dest='threshold', type=float, default=None,
                                help='Para correlation: seleciona no treino do fold atributos com |r| >= valor.')
    optional_group.add_argument('--variance-threshold', type=float, default=None,
                                help='Para variance: fração da variância máxima após min-max (ex.: 0.1, 0.2 ou 0.3).')
    optional_group.add_argument('-leakage-threshold', type=float, default=0.999,
                                help='No modo correlation, |r| a partir deste valor marca atributos suspeitos para remoção (padrão 0.999).')
    optional_group.add_argument('-max-time', type=float, default=3600.0, metavar='SEGUNDOS',
                                help='Limite estimado por ativação em segundos (padrão 3600).')
    optional_group.add_argument('--force-all-kernels', action='store_true',
                                help='Executa todas as ativações mesmo quando a estimativa ultrapassa -max-time.')
    optional_group.add_argument('--language', choices=('pt-br', 'en'), default='pt-br',
                                help='Idioma do dashboard HTML: pt-br (padrão) ou en.')
    optional_group.add_argument('--output-dir', default=None,
                                help='Diretório para relatório, imagens e auditorias (padrão: results/melm_DATA-HORA).')
    optional_group.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                               help='Ativa saída detalhada (verbose)')
    
    args = parser.parse_args()

    feature_selection = args.feature_selection
    if feature_selection is None:
        if args.threshold is not None and args.variance_threshold is not None:
            parser.error('Informe apenas um método de seleção por execução.')
        if args.threshold is not None:
            feature_selection = 'correlation'
        elif args.variance_threshold is not None:
            feature_selection = 'variance'
        else:
            feature_selection = 'none'

    if feature_selection == 'correlation':
        if args.threshold is None:
            parser.error('--feature-selection correlation requer -threshold/--correlation-threshold.')
        if args.variance_threshold is not None:
            parser.error('--variance-threshold só pode ser usado com --feature-selection variance.')
        if not (0.0 <= args.threshold <= 1.0):
            parser.error('-threshold deve estar entre 0 e 1')
    elif feature_selection == 'variance':
        if args.threshold is not None:
            parser.error('-threshold/--correlation-threshold só pode ser usado com --feature-selection correlation.')
        if args.variance_threshold is None:
            parser.error('--feature-selection variance requer --variance-threshold.')
        if not (0.0 <= args.variance_threshold <= 1.0):
            parser.error('--variance-threshold deve estar entre 0 e 1')
    elif args.threshold is not None or args.variance_threshold is not None:
        parser.error('Thresholds não podem ser usados com --feature-selection none.')

    if not (0.0 < args.leakage_threshold <= 1.0):
        parser.error('-leakage-threshold deve estar em (0, 1]')
    if args.max_time <= 0:
        parser.error('-max-time deve ser maior que zero')
    
    # Valida modos de entrada
    has_separate = args.TrainingData_File and args.TestingData_File
    has_all = args.AllData_File
    
    if not has_separate and not has_all:
        parser.error("Deve especificar (-tr e -ts) ou -tall")
    
    if has_separate and has_all:
        parser.error("Não é possível usar (-tr/-ts) e -tall simultaneamente")
    
    if has_separate and (not args.TrainingData_File or not args.TestingData_File):
        parser.error("Ambos -tr e -ts devem ser especificados juntos")
    
    # Executa mELM
    try:
        ff = melm()
        ff.main(
            args.TrainingData_File,
            args.TestingData_File,
            args.AllData_File,
            args.Elm_Type,
            args.nHiddenNeurons,
            args.ActivationFunction,
            args.nSeed,
            args.kfold,
            args.sep,
            args.verbose,
            args.virusNorm,
            args.threshold,
            args.leakage_threshold,
            args.max_time,
            args.force_all_kernels,
            args.output_dir,
            args.language,
            feature_selection,
            args.variance_threshold if feature_selection == 'variance' else None
        )
    except Exception as e:
        print(f"\n✗ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
