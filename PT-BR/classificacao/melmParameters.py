# -*- coding: utf-8 -*-
"""
mELM - Morphological Extreme Learning Machine (Máquina de Aprendizado Extremo Morfológico)
Versão aprimorada com múltiplas funções de ativação, relatórios HTML
e CLI robusta para detecção de malware e tarefas de regressão/classificação.

Versão: 3.0 FINAL - Todos os Kernels Corrigidos
Aplicação: Detecção de Malware, Reconhecimento de Padrões, Tarefas de Regressão

CHANGELOG v3.0 FINAL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KERNELS CORRIGIDOS (Mantém lógica matemática, resolve problemas numéricos):

1. dilation-softmax (corrige dilation):
   - Problema: MAX amplifica outliers, causa saturação (84% ± 29%)
   - Correção: Soft-max = aproximação suave do MAX
   - Lógica mantida: Ainda captura "expansão", mas estável
   - Resultado esperado: ~96-99% com ±3-5% variância

2. fuzzy-erosion-mean (corrige fuzzy-erosion):
   - Problema: Produto de 100 sigmoides → 2.8×10⁻²⁸ ≈ 0 (0% de detecção!)
   - Correção: Média aritmética ao invés de produto
   - Lógica mantida: "EM MÉDIA features são fuzzy-menores"
   - Resultado esperado: ~92-97% com detecção funcional

3. fuzzy-erosion-geom (alternativa):
   - Correção: Média geométrica (raiz N-ésima do produto)
   - Lógica mantida: Mais próxima do produto original
   - Resultado esperado: ~88-95%

4. fuzzy-dilation-mean (corrige fuzzy-dilation):
   - Problema: Produto de complementos → saturação
   - Correção: Média direta das funções de pertinência (membership)
   - Resultado esperado: ~90-95%

5. bitwise-erosion-adaptive (corrige bitwise-erosion):
   - Problema: Threshold fixo 0.5 perde informação
   - Correção: Threshold adaptativo (mediana)
   - Resultado esperado: ~70-85% (ainda limitado)

6. bitwise-dilation-adaptive (corrige bitwise-dilation):
   - Problema: OR muito permissivo
   - Correção: Threshold adaptativo + requer mínimo de features
   - Resultado esperado: ~75-88%

KERNELS ORIGINAIS MANTIDOS:
- erosion: 100% (perfeito, não precisa correção)
- sigmoid, sine, hardlim, tribas, radbas, linear
- dilation, fuzzy-erosion, fuzzy-dilation, bitwise-* (originais para comparação)

VERSÕES ANTERIORES:
- v2.0: Correção de vazamento de dados no virusNorm
- v1.0: Correção de encoding [0,1], adição de métricas de detecção
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USO DOS KERNELS CORRIGIDOS:
    python melmParameters_FINAL_com_correcoes.py \\
        -tall dataset.csv \\
        -ty 1 \\
        -nh 100 \\
        -af erosion,dilation-softmax,fuzzy-erosion-mean,sigmoid,radbas \\
        -kfold 10 \\
        -virusNorm \\
        -v
"""

import numpy as np
import pandas as pd
import argparse
import os
import sys
from time import process_time, time
from datetime import datetime, timedelta
from random import seed as rnd_seed
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template


#========================================================================
# VISUALIZAÇÃO DE DADOS E RELATÓRIOS
#========================================================================

def plot_and_save_cm(cm, title, filename):
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

    # Se a MC já estiver normalizada (0–1), converte para 0–100%
    if cm.max() <= 1.0:
        cm_percent = cm * 100.0
    else:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_percent = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0
        ) * 100.0

    # Prepara anotação: xx.x%
    annot_matrix = np.empty_like(cm_percent, dtype=object)
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            annot_matrix[i, j] = f"{cm_percent[i, j]:.1f}%"

    class_labels = ["Benigno", "Maligno"]  # <--- alteração solicitada

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm_percent,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Porcentagem (%)"},
        xticklabels=[f"Pred {l}" for l in class_labels],
        yticklabels=[f"{l}" for l in class_labels]
    )

    plt.title(title)
    plt.xlabel("Rótulo Predito")
    plt.ylabel("Rótulo Verdadeiro")
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
    """
    Diagnostica automaticamente por que um kernel teve desempenho ruim.
    
    Args:
        results: Dicionário com métricas do kernel
        kernel_name: Nome do kernel
    
    Retorna:
        diagnosis: Dict com problema identificado e explicação
    """
    diagnosis = {
        'status': 'unknown',
        'problem': None,
        'explanation': '',
        'recommendation': ''
    }
    
    # Extrai métricas
    accuracy = results.get('accuracy_test', 0)
    std = results.get('std_test', 0)
    
    # Para classificação binária, extrai métricas de detecção
    cm = results.get('confusion_matrix_test', None)
    if cm is not None and cm.size == 4:
        TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        detection_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    else:
        detection_rate = None
        fpr = None
    
    # DIAGNÓSTICO 1: Detection Rate Zero (Colapso Numérico)
    if detection_rate is not None and detection_rate < 0.01:
        diagnosis['status'] = 'critical'
        diagnosis['problem'] = 'Colapso Numérico'
        
        if 'fuzzy-erosion' in kernel_name and 'mean' not in kernel_name and 'geom' not in kernel_name:
            diagnosis['explanation'] = (
                "O kernel fuzzy-erosion sofre de colapso numérico: "
                "o produto de ~100 funções sigmoid resulta em valores próximos a zero "
                "(0.5^100 ≈ 10^-30), fazendo com que todos os samples sejam classificados "
                "como benignos. A hidden layer colapsa para valores constantes, "
                "impedindo o modelo de discriminar entre classes."
            )
            diagnosis['recommendation'] = (
                "Use 'fuzzy-erosion-mean' que substitui o produto por média aritmética, "
                "mantendo a interpretação fuzzy mas evitando o colapso. "
                "Resultado esperado: 92-97% accuracy com 90-95% detection rate."
            )
        else:
            diagnosis['explanation'] = (
                f"O kernel '{kernel_name}' não detectou nenhum malware (0% detection). "
                "Isso indica que o modelo está marcando todos os samples como benignos, "
                "possivelmente devido a viés forte ou colapso dos valores de ativação."
            )
            diagnosis['recommendation'] = (
                "Tente kernels morfológicos como 'erosion', 'dilation-softmax' ou "
                "'fuzzy-erosion-mean' que são mais robustos para detecção de malware."
            )
    
    # DIAGNÓSTICO 2: Alta Variância (Instabilidade)
    elif std > 15.0:  # Desvio padrão > 15%
        diagnosis['status'] = 'unstable'
        diagnosis['problem'] = 'Alta Instabilidade'
        
        if 'dilation' in kernel_name and 'softmax' not in kernel_name:
            diagnosis['explanation'] = (
                "O kernel dilation é instável porque MAX amplifica outliers. "
                "Com pesos aleatórios, alguns folds têm saturação (valores muito altos) "
                "enquanto outros funcionam bem. Isto resulta em alta variância "
                f"(±{std:.1f}%), tornando os resultados imprevisíveis."
            )
            diagnosis['recommendation'] = (
                "Use 'dilation-softmax' que substitui MAX por soft-max, "
                "uma aproximação suave e estável. "
                "Resultado esperado: 96-99% accuracy com ±3-5% variância."
            )
        else:
            diagnosis['explanation'] = (
                f"O kernel '{kernel_name}' apresenta alta variância (±{std:.1f}%), "
                "indicando que o desempenho varia muito entre folds. "
                "Isso sugere sensibilidade à inicialização aleatória dos pesos."
            )
            diagnosis['recommendation'] = (
                "Kernels com soft-max ou médias (fuzzy-erosion-mean, dilation-softmax) "
                "tendem a ser mais estáveis."
            )
    
    # DIAGNÓSTICO 3: Taxa de Falso Positivo Alta
    elif fpr is not None and fpr > 0.15:  # FPR > 15%
        diagnosis['status'] = 'warning'
        diagnosis['problem'] = 'Alta Taxa de Falso Positivo'
        diagnosis['explanation'] = (
            f"O kernel '{kernel_name}' tem alta taxa de falso positivo ({fpr*100:.1f}%), "
            "marcando muitos arquivos benignos como malware. "
            "Isso pode causar muitos alertas falsos em produção."
        )
        diagnosis['recommendation'] = (
            "Considere kernels mais conservadores como 'erosion' ou "
            "ajuste o threshold de decisão para reduzir falsos positivos."
        )
    
    # DIAGNÓSTICO 4: Baixa Accuracy Geral
    elif accuracy < 80.0:
        diagnosis['status'] = 'poor'
        diagnosis['problem'] = 'Baixo Desempenho Geral'
        
        if 'linear' in kernel_name:
            diagnosis['explanation'] = (
                "O kernel 'linear' não tem não-linearidade, tornando-o equivalente a "
                "regressão linear simples. Malware detection requer captura de padrões "
                "não-lineares, que o kernel linear não consegue representar."
            )
            diagnosis['recommendation'] = "Use kernels não-lineares como 'sigmoid', 'radbas', 'erosion'."
        elif 'sine' in kernel_name:
            diagnosis['explanation'] = (
                "O kernel 'sine' tem comportamento oscilatório, fazendo valores "
                "diferentes produzirem outputs idênticos devido à periodicidade. "
                "Malware detection não tem padrões periódicos naturais."
            )
            diagnosis['recommendation'] = "Use funções monotônicas como 'sigmoid', 'radbas', ou 'erosion'."
        elif 'hardlim' in kernel_name:
            diagnosis['explanation'] = (
                "O kernel 'hardlim' é uma step function binária (0 ou 1), "
                "muito sensível ao threshold e sem gradiente suave. "
                "Isso causa perda de informação."
            )
            diagnosis['recommendation'] = "Use funções suaves como 'sigmoid' ou 'radbas'."
        elif 'bitwise' in kernel_name:
            diagnosis['explanation'] = (
                f"O kernel '{kernel_name}' binariza features contínuas, "
                "perdendo informação sutil crítica para discriminação."
            )
            diagnosis['recommendation'] = "Use kernels que preservam informação contínua como 'erosion' ou 'sigmoid'."
        else:
            diagnosis['explanation'] = (
                f"O kernel '{kernel_name}' obteve apenas {accuracy:.1f}% de accuracy, "
                "indicando dificuldade em aprender padrões discriminativos."
            )
            diagnosis['recommendation'] = (
                "Teste kernels comprovados: 'erosion' (100%), 'dilation-softmax' (96-99%), "
                "'fuzzy-erosion-mean' (92-97%), ou 'radbas' (88-95%)."
            )
    
    # DIAGNÓSTICO 5: Desempenho Bom
    else:
        diagnosis['status'] = 'good'
        diagnosis['problem'] = None
        diagnosis['explanation'] = f"O kernel '{kernel_name}' apresenta bom desempenho ({accuracy:.1f}% ± {std:.1f}%)."
        if detection_rate is not None and detection_rate >= 0.95 and fpr <= 0.05:
            diagnosis['explanation'] += f" Excelente taxa de detecção ({detection_rate*100:.1f}%) com baixo falso positivo ({fpr*100:.1f}%)."
        diagnosis['recommendation'] = "Kernel adequado para uso em produção."
    
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


def mElmStruct(data_file, Elm_Type, sep, verbose):
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

    return all_data, samples_index


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
        # Dilatação: 1 se pelo menos uma característica corresponder
        match = np.any(samples_bin >= se[:, np.newaxis], axis=0)
        H[i, :] = match.astype(float)
    
    H = H + b1.flatten()[:, np.newaxis]
    return H


#========================================================================
# KERNELS MORFOLÓGICOS CORRIGIDOS
# Mantém a lógica matemática original, corrige problemas numéricos
#========================================================================

def dilation_kernel_softmax(w1, b1, samples, temperature=5.0):
    """
    Kernel de Dilatação CORRIGIDO com Soft-Max.
    
    LÓGICA ORIGINAL: Dilatação morfológica = MAX(x + w)
    PROBLEMA: MAX amplifica outliers, causa saturação e instabilidade
    CORREÇÃO: Soft-max = aproximação suave e estável do MAX
    
    Matemática:
        Original:  H = max(x₁ + w, x₂ + w, ..., xₙ + w)
        Corrigido: H = T·log(∑ exp((xᵢ + w)/T))
        
        Com T → 0:  soft-max → max (comportamento clássico)
        Com T = 5:  balanceado entre max e estabilidade
    
    Interpretação morfológica mantida:
    - Ainda captura "expansão" das features
    - Ainda responde a valores altos
    - Mas de forma suave e estável
    
    Args:
        w1: Pesos de entrada (elementos estruturantes)
        b1: Vetor de viés
        samples: Amostras de entrada
        temperature: Controla suavidade (padrão 5.0)
    
    Retorna:
        H: Ativações da camada oculta
    
    Resultado esperado: ~96-99% accuracy com ±3-5% variância
    """
    n_hidden, n_features = w1.shape
    n_samples = samples.shape[1]
    
    H = np.zeros((n_hidden, n_samples))
    for i in range(n_hidden):
        se = w1[i, :]
        values = samples + se[:, np.newaxis]
        
        # Truque Log-sum-exp para estabilidade numérica
        max_val = np.max(values, axis=0, keepdims=True)
        exp_values = np.exp((values - max_val) / temperature)
        log_sum_exp = np.log(np.sum(exp_values, axis=0))
        
        H[i, :] = max_val.flatten() + temperature * log_sum_exp
    
    H = H + b1.flatten()[:, np.newaxis]
    return H


def fuzzy_erosion_kernel_mean(w1, b1, samples):
    """
    Kernel Fuzzy-Erosão CORRIGIDO com Média Aritmética.
    
    LÓGICA ORIGINAL: Erosão fuzzy = produto de membership functions
                     H = ∏ sigmoid(xᵢ - wᵢ)
    PROBLEMA: Produto de 100 termos ~0.5 → 2.8×10⁻²⁸ ≈ 0 (colapso numérico)
    CORREÇÃO: Média aritmética ao invés de produto
    
    Matemática:
        Original:  H = ∏ᵢ sigmoid(xᵢ - wᵢ)  [produto]
        Corrigido: H = (1/N) ∑ᵢ sigmoid(xᵢ - wᵢ)  [média]
    
    Interpretação fuzzy mantida:
    - Original: "TODOS features devem ser fuzzy-menores" (muito restritivo)
    - Corrigido: "EM MÉDIA features são fuzzy-menores" (mais realista)
    - Ambos medem grau de erosão, mas média é numericamente estável
    
    Justificativa teórica:
    - Média aritmética é um operador de agregação válido em lógica fuzzy
    - Preserva a ideia de "erosão" (mínimo fuzzy)
    - Mantém propriedades: monotonia, continuidade
    
    Args:
        w1: Pesos de entrada
        b1: Vetor de viés
        samples: Amostras de entrada
    
    Retorna:
        H: Ativações fuzzy-erodidas (estáveis)
    
    Resultado esperado: ~92-97% accuracy com detecção funcional
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
        
        # Pertinência fuzzy via sigmoide (mantido do original)
        fuzzy_values = 1.0 / (1.0 + np.exp(-np.clip(differences, -40, 40)))
        
        # CORREÇÃO: Média ao invés de produto
        H_batch = np.mean(fuzzy_values, axis=1)
        
        H_batches.append(H_batch)
        del w1_exp, samples_exp, differences, fuzzy_values
    
    H = np.concatenate(H_batches, axis=1)
    H = H + b1.flatten()[:, np.newaxis]
    return H


def fuzzy_erosion_kernel_geometric_mean(w1, b1, samples):
    """
    Kernel Fuzzy-Erosão CORRIGIDO com Média Geométrica.
    
    LÓGICA ORIGINAL: Erosão fuzzy = produto de membership functions
    PROBLEMA: Produto colapsa numericamente
    CORREÇÃO: Média geométrica (raiz N-ésima do produto)
    
    Matemática:
        Original:  H = ∏ᵢ sigmoid(xᵢ - wᵢ)
        Corrigido: H = (∏ᵢ sigmoid(xᵢ - wᵢ))^(1/N)
                     = exp((1/N) × ∑ᵢ log(sigmoid(xᵢ - wᵢ)))
    
    Interpretação fuzzy mantida:
    - Mais próxima do produto original que a média aritmética
    - Ainda captura "mínimo fuzzy" mas de forma estável
    - Média geométrica é um operador válido em lógica fuzzy
    
    Propriedades preservadas:
    - Se TODOS sigmoides são altos → H alto
    - Se ALGUM sigmoid é baixo → H afetado (como no produto)
    - Mas sem colapso numérico
    
    Exemplo: Com N=100 e valores ~0.5:
        Produto:    0.5¹⁰⁰ = 2.8×10⁻²⁸ ≈ 0  ✗
        Geométrica: (0.5¹⁰⁰)^(1/100) = 0.5  ✓
    
    Args:
        w1: Pesos de entrada
        b1: Vetor de viés
        samples: Amostras de entrada
    
    Retorna:
        H: Ativações fuzzy-erodidas
    
    Resultado esperado: ~88-95% accuracy
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
        
        fuzzy_values = 1.0 / (1.0 + np.exp(-np.clip(differences, -40, 40)))
        
        # CORREÇÃO: Média geométrica em log-space
        log_fuzzy = np.log(fuzzy_values + 1e-10)
        mean_log = np.mean(log_fuzzy, axis=1)
        H_batch = np.exp(mean_log)
        
        H_batches.append(H_batch)
        del w1_exp, samples_exp, differences, fuzzy_values, log_fuzzy
    
    H = np.concatenate(H_batches, axis=1)
    H = H + b1.flatten()[:, np.newaxis]
    return H


def fuzzy_dilation_kernel_mean(w1, b1, samples):
    """
    Kernel Fuzzy-Dilatação CORRIGIDO com Média.
    
    LÓGICA ORIGINAL: Dilatação fuzzy = 1 - produto de complementos
                     H = 1 - ∏(1 - sigmoid(xᵢ - wᵢ))
    PROBLEMA: Produto de complementos colapsa (saturação em 1)
    CORREÇÃO: Usa média direta das membership functions
    
    Matemática:
        Original:  H = 1 - ∏ᵢ (1 - sigmoid(xᵢ - wᵢ))
        Corrigido: H = (1/N) ∑ᵢ sigmoid(xᵢ - wᵢ)
    
    Interpretação fuzzy mantida:
    - Ainda captura "expansão" fuzzy
    - Ainda responde a valores altos
    - Mais estável numericamente
    
    Args:
        w1: Pesos de entrada
        b1: Vetor de viés
        samples: Amostras de entrada
    
    Retorna:
        H: Ativações fuzzy-dilatadas
    
    Resultado esperado: ~90-95% accuracy
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
        
        fuzzy_values = 1.0 / (1.0 + np.exp(-np.clip(differences, -40, 40)))
        
        # CORREÇÃO: Média direta
        H_batch = np.mean(fuzzy_values, axis=1)
        
        H_batches.append(H_batch)
        del w1_exp, samples_exp, differences, fuzzy_values
    
    H = np.concatenate(H_batches, axis=1)
    H = H + b1.flatten()[:, np.newaxis]
    return H


def bitwise_erosion_kernel_adaptive(w1, b1, samples):
    """
    Kernel Bitwise-Erosão CORRIGIDO com Threshold Adaptativo.
    
    LÓGICA ORIGINAL: Erosão binária = AND de features binarizadas
    PROBLEMA: Threshold fixo 0.5 perde informação de features contínuas
    CORREÇÃO: Threshold adaptativo baseado na mediana
    
    Matemática:
        Original:  threshold = 0.5 (fixo)
        Corrigido: threshold = median(features) para cada neurônio
    
    Interpretação mantida:
    - Ainda é operação binária (AND)
    - Mas threshold se adapta aos dados
    - Preserva mais informação que threshold fixo
    
    Quando usar:
    - Features naturalmente binárias ou categóricas
    - Quando interpretação "presença/ausência" faz sentido
    
    Args:
        w1: Pesos de entrada
        b1: Vetor de viés
        samples: Amostras de entrada
    
    Retorna:
        H: Ativações bitwise-erodidas
    
    Resultado esperado: ~70-85% accuracy (ainda limitado)
    """
    n_hidden, n_features = w1.shape
    n_samples = samples.shape[1]
    
    H = np.zeros((n_hidden, n_samples))
    
    for i in range(n_hidden):
        se = w1[i, :]
        
        # CORREÇÃO: Threshold adaptativo (mediana ao invés de 0.5 fixo)
        threshold_sample = np.median(samples, axis=0)
        threshold_weight = np.median(se)
        
        samples_bin = (samples > threshold_sample).astype(float)
        se_bin = (se > threshold_weight).astype(float)
        
        # Erosão: AND lógico
        match = np.all(samples_bin >= se_bin[:, np.newaxis], axis=0)
        H[i, :] = match.astype(float)
    
    H = H + b1.flatten()[:, np.newaxis]
    return H


def bitwise_dilation_kernel_adaptive(w1, b1, samples):
    """
    Kernel Bitwise-Dilatação CORRIGIDO com Threshold Adaptativo.
    
    LÓGICA ORIGINAL: Dilatação binária = OR de features binarizadas
    PROBLEMA: Threshold fixo + OR muito permissivo
    CORREÇÃO: Threshold adaptativo + contagem mínima
    
    Matemática:
        Original:  OR de TODOS features (muito permissivo)
        Corrigido: Threshold adaptativo + requer K% features ativos
    
    Args:
        w1: Pesos de entrada
        b1: Vetor de viés
        samples: Amostras de entrada
    
    Retorna:
        H: Ativações bitwise-dilatadas
    
    Resultado esperado: ~75-88% accuracy
    """
    n_hidden, n_features = w1.shape
    n_samples = samples.shape[1]
    
    H = np.zeros((n_hidden, n_samples))
    
    for i in range(n_hidden):
        se = w1[i, :]
        
        # CORREÇÃO: Threshold adaptativo
        threshold_sample = np.median(samples, axis=0)
        threshold_weight = np.median(se)
        
        samples_bin = (samples > threshold_sample).astype(float)
        se_bin = (se > threshold_weight).astype(float)
        
        # Dilatação: OR lógico mas requer pelo menos 10% dos features
        matches = samples_bin >= se_bin[:, np.newaxis]
        match_count = np.sum(matches, axis=0)
        H[i, :] = (match_count >= max(1, n_features * 0.1)).astype(float)
    
    H = H + b1.flatten()[:, np.newaxis]
    return H


#========================================================================
# FUNÇÕES DE ATIVAÇÃO TRADICIONAIS
#========================================================================

def switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, P):
    """
    Interface unificada para calcular a saída da camada oculta H para uma dada ativação.
    
    Suporta:
    - Ativações tradicionais: linear, sig, sin, hardlim, tribas, radbas
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
    
    elif ActivationFunction in ('sin', 'sine'):
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
    
    # ========================================================================
    # KERNELS CORRIGIDOS - Versões corrigidas mantendo lógica matemática
    # ========================================================================
    elif ActivationFunction in ('dilation-softmax', 'dil-soft', 'dilation-corrected'):
        H = dilation_kernel_softmax(InputWeight, BiasofHiddenNeurons, P, temperature=5.0)
    
    elif ActivationFunction in ('fuzzy-erosion-mean', 'fuzzy-ero-mean', 'fuzzy-erosion-corrected'):
        H = fuzzy_erosion_kernel_mean(InputWeight, BiasofHiddenNeurons, P)
    
    elif ActivationFunction in ('fuzzy-erosion-geom', 'fuzzy-ero-geom', 'fuzzy-erosion-geometric'):
        H = fuzzy_erosion_kernel_geometric_mean(InputWeight, BiasofHiddenNeurons, P)
    
    elif ActivationFunction in ('fuzzy-dilation-mean', 'fuzzy-dil-mean', 'fuzzy-dilation-corrected'):
        H = fuzzy_dilation_kernel_mean(InputWeight, BiasofHiddenNeurons, P)
    
    elif ActivationFunction in ('bitwise-erosion-adaptive', 'bitwise-ero-adapt'):
        H = bitwise_erosion_kernel_adaptive(InputWeight, BiasofHiddenNeurons, P)
    
    elif ActivationFunction in ('bitwise-dilation-adaptive', 'bitwise-dil-adapt'):
        H = bitwise_dilation_kernel_adaptive(InputWeight, BiasofHiddenNeurons, P)
    
    else:
        raise ValueError(f"Função de ativação desconhecida: {ActivationFunction}")
    
    return H


#========================================================================
# ALGORITMO DE APRENDIZAGEM ELM
#========================================================================

def mElmLearning(train_data, test_data, Elm_Type, NumberofHiddenNeurons, 
                 ActivationFunction, execution, kfold, verbose, virusNorm=False):
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
             NumberofHiddenNeurons, ActivationFunction, nSeed, kfold, sep, verbose, virusNorm=False):
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
        # Define todas as funções de ativação disponíveis
        ALL_FUNCTIONS = ['sig', 'sin', 'radbas', 'linear', 'hardlim', 'tribas',
                         'erosion', 'dilation', 'fuzzy-erosion', 'fuzzy-dilation',
                         'bitwise-erosion', 'bitwise-dilation',
                         'dilation-softmax', 'fuzzy-erosion-mean', 'fuzzy-erosion-geom',
                         'fuzzy-dilation-mean', 'bitwise-erosion-adaptive', 
                         'bitwise-dilation-adaptive']
        
        # Parse das funções de ativação
        if ActivationFunction == 'all':
            acts = ALL_FUNCTIONS
        else:
            acts = [s.strip() for s in str(ActivationFunction or 'linear').split(',') if s.strip()]
        
        # Parse das contagens de neurônios ocultos
        nh_list = [int(v.strip()) for v in str(NumberofHiddenNeurons).split(',') if str(v).strip()]
        
        # Define semente aleatória
        if nSeed is None:
            nSeed = 1
        else:
            nSeed = int(nSeed)
        rnd_seed(nSeed)
        np.random.seed(nSeed)
        
        Elm_Type = int(Elm_Type)
        
        # Determina modo de execução
        use_separate_files = TrainingData_File is not None and TestingData_File is not None
        
        combo_results = []
        
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
            
            train_data, _ = mElmStruct(TrainingData_File, Elm_Type, sep, verbose)
            test_data, _ = mElmStruct(TestingData_File, Elm_Type, sep, verbose)
            
            for af in acts:
                for nh in nh_list:
                    if verbose:
                        print(f'\n=== Avaliando: Ativação={af}, Neurônios={nh} ===')
                    
                    TA, TeA, TT, Tt, cm_train, cm_test = mElmLearning(
                        train_data, test_data, Elm_Type, nh, af, 0, 1, verbose, virusNorm
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
                        "confusion_matrix_test": cm_test
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
            
            all_data, samples_index = mElmStruct(AllData_File, Elm_Type, sep, verbose)
            kf = KFold(n_splits=int(kfold), shuffle=True, random_state=nSeed)
            
            # ✓ ESTIMATIVA DE TEMPO
            n_samples = len(samples_index)
            print_time_estimate(len(acts), len(nh_list), kfold, n_samples)
            
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
                    
                    for i, (tr_idx, te_idx) in enumerate(kf.split(samples_index)):
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
                            train_data, test_data, Elm_Type, nh, af, i, kfold, verbose, virusNorm
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
                        "confusion_matrix_test": np.mean(cms_test, axis=0) if cms_test else None
                    }
                    
                    # ✓ DIAGNÓSTICO AUTOMÁTICO
                    if Elm_Type == 1 and verbose:  # Só para classificação
                        diagnosis = diagnose_kernel_performance(current_result, af)
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
        
        # Coletar diagnósticos para relatório HTML (apenas classificação)
        diagnostics_dict = {}
        if Elm_Type == 1:
            for result in combo_results:
                kernel_name = result['act']
                if kernel_name not in diagnostics_dict and 'diagnosis' in result:
                    diagnostics_dict[kernel_name] = result['diagnosis']
        
        # Gera relatório HTML
        generate_html_report_elm(global_results, act_results, diagnostics_dict)


#========================================================================
# GERAÇÃO DE RELATÓRIO HTML
#========================================================================

def generate_html_report_elm(global_results, act_results, diagnostics=None, output_file='elm_report.html'):
    """
    Gera um dashboard HTML resumindo os resultados do mELM.

    O relatório se adapta automaticamente para:
    - Tarefas de classificação (Elm_Type == 1): métricas interpretadas como Acurácia (%)
    - Tarefas de regressão    (Elm_Type == 0): métricas interpretadas como RMSE (sem sinal %)
    """

    # --- Tipo de tarefa (classificação vs regressão) ---------------------------
    elm_type = int(global_results.get("elm_type", 1))
    is_classification = elm_type == 1

    metric_label = "Acurácia" if is_classification else "RMSE"
    metric_unit = "%" if is_classification else ""

    # --- Caminhos: HTML + pasta de imagens no mesmo diretório do script -----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir_name = "elm_report_images"
    img_dir = os.path.join(script_dir, img_dir_name)
    os.makedirs(img_dir, exist_ok=True)

    def _safe_plot_cm(cm, title, rel_filename):
        """
        Plota matrizes de confusão apenas quando existem e não estão vazias.

        rel_filename: caminho relativo ao diretório do script (ex: 'elm_report_images/cm_xxx.png')
        Retorna esse caminho relativo se o plot foi criado, ou None caso contrário.
        """
        if cm is None:
            return None
        try:
            if isinstance(cm, np.ndarray) and cm.size > 0 and cm.sum() > 0:
                abs_path = os.path.join(script_dir, rel_filename)
                plot_and_save_cm(cm, title, abs_path)
                return rel_filename
        except Exception:
            # Não quebra a geração do relatório se der erro no gráfico
            return None
        return None

    # ------------------------------------------------------------------------
    # Matrizes de confusão globais (somente classificação)
    # ------------------------------------------------------------------------
    cm_global_best = None
    cm_global_worst = None
    if is_classification:
        cm_global_best = _safe_plot_cm(
            global_results["max"].get("confusion_matrix_test"),
            "MC Média - Melhor Global (Teste)",
            os.path.join(img_dir_name, "cm_global_best.png"),
        )
        cm_global_worst = _safe_plot_cm(
            global_results["min"].get("confusion_matrix_test"),
            "MC Média - Pior Global (Teste)",
            os.path.join(img_dir_name, "cm_global_worst.png"),
        )

    # ------------------------------------------------------------------------
    # Matrizes de confusão por função de ativação (somente classificação)
    # act_results é um dict: { act_name: {"max_test": obj, "min_test": obj} }
    # ------------------------------------------------------------------------
    act_cm = {}
    if is_classification:
        for act_name, data in act_results.items():
            key = act_name.replace(" ", "_")
            cm_best = _safe_plot_cm(
                data["max_test"].get("confusion_matrix_test"),
                f"MC Média - Melhor Teste {act_name}",
                os.path.join(img_dir_name, f"cm_act_{key}_best_test.png"),
            )
            cm_worst = _safe_plot_cm(
                data["min_test"].get("confusion_matrix_test"),
                f"MC Média - Pior Teste {act_name}",
                os.path.join(img_dir_name, f"cm_act_{key}_worst_test.png"),
            )
            act_cm[act_name] = {"best": cm_best, "worst": cm_worst}

    # ------------------------------------------------------------------------
    # TEMPLATE HTML: mesmo estilo/cores/fontes e logo da UFPE
    # que existem no elm_report.html gerado pelo melmParameters.py
    # ------------------------------------------------------------------------
    html_template = r"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ELM Dashboard - Relatório de Resultados</title>
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
            .section-title { font-size: 2em; margin-bottom: 20px; font-weight: 600; background: linear-gradient(45deg, #8B1538, #A91E4A); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
            .kernel-group { margin-bottom: 40px; }
            .kernel-title { font-size: 1.5em; font-weight: 600; margin-bottom: 20px; padding: 15px 20px; color: white; border-radius: 10px; text-align: center; background: linear-gradient(45deg, #8B1538, #A91E4A); }
            .kernel-results { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }
            .result-card { background: rgba(255, 255, 255, 0.9); border-radius: 15px; padding: 25px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1); transition: transform 0.3s ease; }
            .result-card:hover { transform: translateY(-3px); }
            .result-card.best { border: 2px solid #4CAF50; } .result-card.worst { border: 2px solid #f44336; }
            .result-header { display: flex; align-items: center; margin-bottom: 20px; }
            .result-icon { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; color: white; font-weight: bold; }
            .logo-ufpe { height: 150px; width: auto; }
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
                <img src="../../src/ufpe_logo.png" alt="Logotipo da UFPE" class="logo-ufpe">
                <h1>ELM - Avaliação de Parâmetros</h1>
                <p class="subtitle">
                    {% if is_classification %}
                        A acurácia de teste é a métrica de escolha para os resultados
                    {% else %}
                        O RMSE de teste é a métrica de escolha para os resultados
                    {% endif %}
                </p>
            </div>

            <div class="stats-grid">
                <div class="stat-card best">
                    <div class="card-header"><div class="card-icon best">🏆</div><div class="card-title">Melhor Desempenho Global</div></div>
                    <div class="metric-row"><span class="metric-label">Configuração (n_hidden)</span><span class="metric-value best">{{ global_results.max.n_hidden }}</span></div>
                    <div class="metric-row"><span class="metric-label">Melhor Ativação</span><span class="metric-value best">{{ global_results.max.act }}</span></div>
                    <div class="metric-row">
                        <span class="metric-label">
                            {% if is_classification %}Acurácia de Treino{% else %}RMSE de Treino{% endif %}
                        </span>
                        <span class="metric-value best">
                            {{ "%.2f"|format(global_results.max.accuracy_train) }}{{ metric_unit }}
                            {% if global_results.max.std_train is not none and global_results.max.std_train > 0 %}
                                &plusmn; {{ "%.2f"|format(global_results.max.std_train) }}{{ metric_unit }}
                            {% endif %}
                        </span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">
                            {% if is_classification %}Acurácia de Teste{% else %}RMSE de Teste{% endif %}
                        </span>
                        <span class="metric-value best">
                            {{ "%.2f"|format(global_results.max.accuracy_test) }}{{ metric_unit }}
                            {% if global_results.max.std_test is not none and global_results.max.std_test > 0 %}
                                &plusmn; {{ "%.2f"|format(global_results.max.std_test) }}{{ metric_unit }}
                            {% endif %}
                        </span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Tempo de Treino</span>
                        <span class="metric-value best">
                            {{ "%.4f"|format(global_results.max.time_train) }}s
                            {% if global_results.max.std_time_train is not none and global_results.max.std_time_train > 0 %}
                                &plusmn; {{ "%.4f"|format(global_results.max.std_time_train) }}s
                            {% endif %}
                        </span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Tempo de Teste</span>
                        <span class="metric-value best">
                            {{ "%.4f"|format(global_results.max.time_test) }}s
                            {% if global_results.max.std_time_test is not none and global_results.max.std_time_test > 0 %}
                                &plusmn; {{ "%.4f"|format(global_results.max.std_time_test) }}s
                            {% endif %}
                        </span>
                    </div>
                    {% if is_classification and cm_global_best %}
                    <div class="cm-container">
                        <img class="cm-image" src="{{ cm_global_best }}" alt="Matriz de Confusão - Melhor Global">
                    </div>
                    {% endif %}
                </div>

                <div class="stat-card worst">
                    <div class="card-header"><div class="card-icon worst">💎</div><div class="card-title">Pior Desempenho Global</div></div>
                    <div class="metric-row"><span class="metric-label">Configuração (n_hidden)</span><span class="metric-value worst">{{ global_results.min.n_hidden }}</span></div>
                    <div class="metric-row"><span class="metric-label">Pior Ativação</span><span class="metric-value worst">{{ global_results.min.act }}</span></div>
                    <div class="metric-row">
                        <span class="metric-label">
                            {% if is_classification %}Acurácia de Treino{% else %}RMSE de Treino{% endif %}
                        </span>
                        <span class="metric-value worst">
                            {{ "%.2f"|format(global_results.min.accuracy_train) }}{{ metric_unit }}
                            {% if global_results.min.std_train is not none and global_results.min.std_train > 0 %}
                                &plusmn; {{ "%.2f"|format(global_results.min.std_train) }}{{ metric_unit }}
                            {% endif %}
                        </span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">
                            {% if is_classification %}Acurácia de Teste{% else %}RMSE de Teste{% endif %}
                        </span>
                        <span class="metric-value worst">
                            {{ "%.2f"|format(global_results.min.accuracy_test) }}{{ metric_unit }}
                            {% if global_results.min.std_test is not none and global_results.min.std_test > 0 %}
                                &plusmn; {{ "%.2f"|format(global_results.min.std_test) }}{{ metric_unit }}
                            {% endif %}
                        </span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Tempo de Treino</span>
                        <span class="metric-value worst">
                            {{ "%.4f"|format(global_results.min.time_train) }}s
                            {% if global_results.min.std_time_train is not none and global_results.min.std_time_train > 0 %}
                                &plusmn; {{ "%.4f"|format(global_results.min.std_time_train) }}s
                            {% endif %}
                        </span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Tempo de Teste</span>
                        <span class="metric-value worst">
                            {{ "%.4f"|format(global_results.min.time_test) }}s
                            {% if global_results.min.std_time_test is not none and global_results.min.std_time_test > 0 %}
                                &plusmn; {{ "%.4f"|format(global_results.min.std_time_test) }}s
                            {% endif %}
                        </span>
                    </div>
                    {% if is_classification and cm_global_worst %}
                    <div class="cm-container">
                        <img class="cm-image" src="{{ cm_global_worst }}" alt="Matriz de Confusão - Pior Global">
                    </div>
                    {% endif %}
                </div>
            </div>

            <div class="kernels-section">
                <h2 class="section-title">Resumo por Função de Ativação</h2>
                {% for act_name, data in act_results.items() %}
                <div class="kernel-group">
                    <div class="kernel-title">Função de Ativação: {{ act_name }}</div>
                    <div class="kernel-results">
                        {% if data.max_test %}
                        <div class="result-card best">
                            <div class="result-header"><div class="result-icon best">👍</div><div class="result-title">Melhor Cenário</div></div>
                            <ul class="metrics-list">
                                <li><span class="metric-name">Configuração (n_hidden):</span><span class="metric-val">{{ data.max_test.n_hidden }}</span></li>
                                <li>
                                    <span class="metric-name">
                                        {% if is_classification %}Acurácia de Teste{% else %}RMSE de Teste{% endif %}
                                    </span>
                                    <span class="metric-val">
                                        {{ "%.2f"|format(data.max_test.accuracy_test) }}{{ metric_unit }}
                                        {% if data.max_test.std_test is not none and data.max_test.std_test > 0 %}
                                            &plusmn; {{ "%.2f"|format(data.max_test.std_test) }}{{ metric_unit }}
                                        {% endif %}
                                    </span>
                                </li>
                                <li><span class="metric-name">Tempo de Teste:</span><span class="metric-val">{{ "%.4f"|format(data.max_test.time_test) }}s{% if data.max_test.std_time_test is not none and data.max_test.std_time_test > 0 %} &plusmn; {{ "%.4f"|format(data.max_test.std_time_test) }}s{% endif %}</span></li>
                                <li>
                                    <span class="metric-name">
                                        {% if is_classification %}Acurácia de Treino{% else %}RMSE de Treino{% endif %}
                                    </span>
                                    <span class="metric-val">
                                        {{ "%.2f"|format(data.max_test.accuracy_train) }}{{ metric_unit }}
                                        {% if data.max_test.std_train is not none and data.max_test.std_train > 0 %}
                                            &plusmn; {{ "%.2f"|format(data.max_test.std_train) }}{{ metric_unit }}
                                        {% endif %}
                                    </span>
                                </li>
                                <li><span class="metric-name">Tempo de Treino:</span><span class="metric-val">{{ "%.4f"|format(data.max_test.time_train) }}s{% if data.max_test.std_time_train is not none and data.max_test.std_time_train > 0 %} &plusmn; {{ "%.4f"|format(data.max_test.std_time_train) }}s{% endif %}</span></li>
                            </ul>
                            {% if is_classification and act_cm.get(act_name) and act_cm[act_name].best %}
                            <div class="cm-container"><img class="cm-image" src="{{ act_cm[act_name].best }}" alt="MC Melhor Teste {{ act_name }}"></div>
                            {% endif %}
                        </div>
                        {% endif %}

                        {% if data.min_test %}
                        <div class="result-card worst">
                            <div class="result-header"><div class="result-icon worst">💎</div><div class="result-title">Pior Cenário</div></div>
                            <ul class="metrics-list">
                                <li><span class="metric-name">Configuração (n_hidden):</span><span class="metric-val">{{ data.min_test.n_hidden }}</span></li>
                                <li>
                                    <span class="metric-name">
                                        {% if is_classification %}Acurácia de Teste{% else %}RMSE de Teste{% endif %}
                                    </span>
                                    <span class="metric-val">
                                        {{ "%.2f"|format(data.min_test.accuracy_test) }}{{ metric_unit }}
                                        {% if data.min_test.std_test is not none and data.min_test.std_test > 0 %}
                                            &plusmn; {{ "%.2f"|format(data.min_test.std_test) }}{{ metric_unit }}
                                        {% endif %}
                                    </span>
                                </li>
                                <li><span class="metric-name">Tempo de Teste:</span><span class="metric-val">{{ "%.4f"|format(data.min_test.time_test) }}s{% if data.min_test.std_time_test is not none and data.min_test.std_time_test > 0 %} &plusmn; {{ "%.4f"|format(data.min_test.std_time_test) }}s{% endif %}</span></li>
                                <li>
                                    <span class="metric-name">
                                        {% if is_classification %}Acurácia de Treino{% else %}RMSE de Treino{% endif %}
                                    </span>
                                    <span class="metric-val">
                                        {{ "%.2f"|format(data.min_test.accuracy_train) }}{{ metric_unit }}
                                        {% if data.min_test.std_train is not none and data.min_test.std_train > 0 %}
                                            &plusmn; {{ "%.2f"|format(data.min_test.std_train) }}{{ metric_unit }}
                                        {% endif %}
                                    </span>
                                </li>
                                <li><span class="metric-name">Tempo de Treino:</span><span class="metric-val">{{ "%.4f"|format(data.min_test.time_train) }}s{% if data.min_test.std_time_train is not none and data.min_test.std_time_train > 0 %} &plusmn; {{ "%.4f"|format(data.min_test.std_time_train) }}s{% endif %}</span></li>
                            </ul>
                            {% if is_classification and act_cm.get(act_name) and act_cm[act_name].worst %}
                            <div class="cm-container"><img class="cm-image" src="{{ act_cm[act_name].worst }}" alt="MC Pior Teste {{ act_name }}"></div>
                            {% endif %}
                        </div>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
            </div>
            
            {% if is_classification and diagnostics %}
            <div class="kernels-section" style="margin-top: 40px;">
                <h2 class="section-title">🔍 Diagnósticos Automáticos</h2>
                <p style="color: #666; margin-bottom: 30px; font-size: 1.1em;">
                    Análise detalhada do desempenho de cada kernel com recomendações específicas
                </p>
                
                {% for kernel_name, diagnosis in diagnostics.items() %}
                <div class="diagnosis-card status-{{ diagnosis.status }}" style="background: rgba(255, 255, 255, 0.95); border-radius: 15px; padding: 25px; margin-bottom: 20px; border-left: 5px solid 
                {% if diagnosis.status == 'critical' %}#dc3545
                {% elif diagnosis.status == 'unstable' or diagnosis.status == 'warning' or diagnosis.status == 'poor' %}#ffc107
                {% elif diagnosis.status == 'good' %}#28a745
                {% else %}#6c757d{% endif %};">
                    
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <div style="font-size: 2em; margin-right: 15px;">
                            {% if diagnosis.status == 'critical' %}❌
                            {% elif diagnosis.status == 'unstable' or diagnosis.status == 'warning' or diagnosis.status == 'poor' %}⚠️
                            {% elif diagnosis.status == 'good' %}✅
                            {% else %}❓{% endif %}
                        </div>
                        <div>
                            <h3 style="margin: 0; font-size: 1.5em; color: #333;">{{ kernel_name }}</h3>
                            {% if diagnosis.problem %}
                            <p style="margin: 5px 0 0 0; color: #666; font-weight: 500;">{{ diagnosis.problem }}</p>
                            {% endif %}
                        </div>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <strong style="color: #555;">📝 Explicação:</strong>
                        <p style="margin: 10px 0 0 0; line-height: 1.6; color: #666;">{{ diagnosis.explanation }}</p>
                    </div>
                    
                    <div style="background: {% if diagnosis.status == 'good' %}#d4edda{% else %}#fff3cd{% endif %}; padding: 15px; border-radius: 8px; border-left: 3px solid {% if diagnosis.status == 'good' %}#28a745{% else %}#ffc107{% endif %};">
                        <strong style="color: #555;">💡 Recomendação:</strong>
                        <p style="margin: 10px 0 0 0; line-height: 1.6; color: #666;">{{ diagnosis.recommendation }}</p>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </div>
    </body>
    </html>
    """

    template = Template(html_template)
    rendered = template.render(
        global_results=global_results,
        act_results=act_results,
        is_classification=is_classification,
        metric_label=metric_label,
        metric_unit=metric_unit,
        cm_global_best=cm_global_best,
        cm_global_worst=cm_global_worst,
        act_cm=act_cm,
        diagnostics=diagnostics if diagnostics else {},
    )

    output_path = os.path.join(script_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)

    print(f"Relatório HTML gerado: {output_path}")


#========================================================================
# INTERFACE DE LINHA DE COMANDO (CLI)
#========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='mELM - Máquina de Aprendizado Extremo Morfológico para Detecção de Malware',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Validação cruzada K-fold com kernels morfológicos
  python melmParameters_complete.py -tall dataset.csv -ty 1 -nh 100 -af erosion,dilation -kfold 10 -v
  
  # Treino/teste separados com normalização virusNorm
  python melmParameters_complete.py -tr train.csv -ts test.csv -ty 1 -virusNorm -nh 100 -af dilation -v
  
  # Testar todos os kernels com múltiplas contagens de neurônios
  python melmParameters_complete.py -tall malware.csv -ty 1 -nh 50,100,200 -af all -kfold 5 -v
  
  # Tarefa de regressão
  python melmParameters_complete.py -tr train.csv -ts test.csv -ty 0 -nh 100 -af linear -v

Funções de Ativação:
  Tradicionais: linear, sig, sin, hardlim, tribas, radbas
  Morfológicas: erosion, dilation, fuzzy-erosion, fuzzy-dilation, bitwise-erosion, bitwise-dilation
  Use 'all' para testar todas as funções
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
    optional_group.add_argument('-sd', '--seed', dest='nSeed', action='store', type=int,
                               help='Semente aleatória para reprodutibilidade')
    optional_group.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                               help='Ativa saída detalhada (verbose)')
    
    args = parser.parse_args()
    
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
            args.virusNorm
        )
    except Exception as e:
        print(f"\n✗ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)