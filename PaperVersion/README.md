# mELM — versão de pesquisa

Versão de pesquisa para experimentos reproduzíveis de **classificação binária** e **regressão** em datasets numéricos. O fluxo não é específico de Android, MobSF ou antivírus. Ele mantém a implementação mELM original como ponto de partida e acrescenta seleção de atributos fold-safe opcional, leitura de CSV e LIBSVM, controle de tempo, métricas por fold e relatório HTML.

## Créditos e origem

A adaptação para o mestrado é realizada por **WPSS** e **TMA**, sob orientação de **SMLL**. O algoritmo e o código-base permanecem atribuídos aos autores do projeto original:

- [Repositório DejaVuForensics/mELM](https://github.com/DejavuForensics/mELM)
- Pinheiro et al., “Antivirus applied to JAR malware detection based on runtime behaviors”, *Scientific Reports*, 12, 1945 (2022), [doi:10.1038/s41598-022-05921-5](https://doi.org/10.1038/s41598-022-05921-5).
- Huang, Zhu e Siew, “Extreme learning machine: Theory and applications”, *Neurocomputing*, 70, 489–501 (2006), [doi:10.1016/j.neucom.2005.12.126](https://doi.org/10.1016/j.neucom.2005.12.126).
- Serra, *Image Analysis and Mathematical Morphology*, Academic Press (1982), [registro bibliográfico](https://books.google.com/books?id=QJMoAQAAMAAJ).
- Zadeh, “Fuzzy sets”, *Information and Control*, 8(3), 338–353 (1965), [doi:10.1016/S0019-9958(65)90241-X](https://doi.org/10.1016/S0019-9958(65)90241-X).

## O que mudou nesta adaptação

- Mantém os operadores e a estrutura mELM do script de origem, mas aceita datasets numéricos gerais.
- Lê CSV no formato mELM e arquivos esparsos LIBSVM, incluindo `data/heart_scale`.
- Usa validação cruzada estratificada para classificação binária e `KFold` para regressão.
- Permite executar sem seleção de atributos (padrão) ou ativar, explicitamente, seleção por correlação ou por variância.
- Quando a seleção está ativa, ajusta o critério somente com os dados de treino de cada fold e aplica a mesma seleção ao fold de validação.
- Mantém a ordem semântica de classes: rótulo ordenado menor = negativo; maior = positivo. A matriz é calculada com linhas verdadeiras/colunas previstas e exibida no dashboard com linhas previstas, colunas reais, ordem positiva/negativa e TP e TN na diagonal.
- Calcula média e desvio padrão das quatro células da matriz entre folds e guarda os valores individuais para boxplots.
- Estima o custo de cada ativação a partir do custo aproximado da camada oculta e da pseudo-inversa. A estimativa é heurística; os tempos efetivos por fold são registrados durante a execução.
- Produz relatório HTML, imagens, auditorias de atributos e boxplots em um diretório próprio de resultados.

## Fundamento matemático e limites das descrições

Na ELM, os pesos e vieses da camada oculta são inicializados antes do ajuste da saída. Para uma amostra \(x\) e neurônio oculto \(j\), a ativação convencional é \(h_j=\phi(w_j^Tx+b_j)\). Juntando as ativações de todas as amostras na matriz \(H\), a saída é ajustada por mínimos quadrados: \(\beta=H^{\dagger}T\), em que \(H^{\dagger}\) é a pseudo-inversa e \(T\) contém os alvos. A predição é \(\hat T=H\beta\). A pseudo-inversa fornece uma solução de mínimos quadrados de norma mínima; ela não torna o modelo imune a sobreajuste ou a uma escolha inadequada de ativação. Essa é a estrutura geral apresentada por Huang, Zhu e Siew (2006).

Neste README, “ativação” designa a transformação aplicada aos atributos em cada neurônio oculto. O código e o dashboard também usam “kernel” como nome operacional para essas opções. Isso não significa que todas sejam kernels de Mercer, nem que sejam kernels de SVM.

O artigo mELM de Pinheiro et al. (2022) apresenta erosão e dilatação como operadores da camada oculta inspirados na morfologia matemática. No código desta pasta, as implementações escalarizadas usadas por cada neurônio são:

| Ativação | Operação implementada no código |
|---|---|
| `sigmoid` | \(\sigma(z)=1/(1+e^{-z})\), com \(z=w^Tx+b\) |
| `sine` | \(\sin(z)\) |
| `linear` | \(z\) |
| `hardlim` | \(1\) se \(z>0\), senão \(0\) |
| `tribas` | \(\max(1-|z|,0)\) |
| `radbas` | \(e^{-z^2}\) |
| `erosion` | \(\min_i(x_i-w_i)+b\) |
| `dilation` | \(\max_i(x_i+w_i)+b\) |
| `fuzzy-erosion` | \(\prod_i \sigma(x_i-w_i)+b\) |
| `fuzzy-dilation` | \(1-\prod_i[1-\sigma(x_i-w_i)]+b\) |
| `bitwise-erosion` | Binariza entrada e pesos em 0,5; saída 1 se todos os bits ativos do elemento estruturante também estiverem ativos na amostra |
| `bitwise-dilation` | Binariza em 0,5; saída 1 se existir ao menos uma posição ativa simultaneamente na amostra e no elemento estruturante |

As definições clássicas de erosão e dilatação vêm da morfologia matemática (Serra, 1982), e o artigo mELM fundamenta a inspiração dessas operações na camada oculta. A tese de Sidney Marlon Lopes de Lima descreve dez funções para ELM e deriva fuzzy-erosion e fuzzy-dilation nas equações 55–62. As expressões finais da tese são \(f_{FE}(x,w)=\prod_i[1-(1-x_i)w_i]+b\) (eq. 59) e \(f_{FD}(x,w)=1-\prod_i(1-x_iw_i)+b\) (eq. 62). As expressões fuzzy desta tabela, porém, usam uma sigmoide sobre diferenças entre atributos e pesos; elas não reproduzem literalmente as equações da tese. A lista da tese também inclui funções polinomial e wavelet que não estão disponíveis entre as 12 opções deste script. Já `radbas` aplica \(e^{-z^2}\) a um escalar afim e não é idêntica, por definição, a um kernel RBF aplicado à distância entre vetores. Logo, a tese sustenta a motivação e a linhagem das funções, mas não valida automaticamente cada implementação atual. As fórmulas da tabela descrevem **o código principal atual**.

As ativações convencionais (`sigmoid`, `sine`, `linear`, `hardlim`, `tribas` e `radbas`) são funções de ativação usuais em ELMs. `sig` e `sin` continuam aceitos como aliases de compatibilidade; `sinusoid` e `sinusoidal` também são aceitos para a ativação senoidal. A erosão e a dilatação min/max são os operadores morfológicos implementados nesta versão. As versões `fuzzy-*` e `bitwise-*` da tabela devem ser lidas como as expressões concretas do programa e avaliadas por suas propriedades e evidência experimental; a nomenclatura, sozinha, não constitui validação matemática. A comparação detalhada entre as expressões do código e a tese está no arquivo `KERNEL_PROVENANCE.md` da distribuição separada `mELM-Experimental-Kernels`.

`-af all` executa somente as **12 ativações-base**. As cinco variantes de pesquisa ficam na distribuição separada `mELM-Experimental-Kernels`; elas não fazem parte deste script nem do conjunto `all`.

## Reconhecimento de padrões e predição

Na classificação, o script atual aceita duas classes numéricas e interpreta a menor classe ordenada como negativa e a maior como positiva. O dashboard apresenta acurácia, precisão, recall/detecção, taxa de falsos positivos e métricas derivadas da matriz de confusão. A acurácia não deve ser usada isoladamente quando as classes são desbalanceadas; examine também os erros por classe e os valores absolutos de TP, FP, TN e FN.

Na regressão, a saída é avaliada pela raiz do erro quadrático médio (RMSE), na escala do alvo. Menor RMSE indica menor erro quadrático médio nesse conjunto, mas não permite comparar diretamente datasets com escalas diferentes. O script pode executar ambos os tipos de tarefa; o dashboard e os critérios de melhor cenário mudam de acordo com a tarefa.

As ativações afetam a representação produzida pela camada oculta. A afirmação do README antigo de que erosão/dilatação “se adaptam a qualquer fronteira” é uma hipótese ampla, não uma garantia matemática. A escolha adequada depende dos dados e deve ser verificada em validação, com avaliação final independente.

## Validação cruzada K-fold

Em `KFold`, os dados são divididos em \(K\) partes; cada rodada ajusta o modelo em \(K-1\) partes e avalia na parte restante. Para classificação binária, o programa usa divisão estratificada para preservar aproximadamente a proporção entre as classes. A semente controla a divisão reproduzível. A seleção de atributos é ajustada dentro do treino de cada fold e a máscara resultante é aplicada à validação daquele fold.

Comparar muitas ativações e tamanhos de camada e escolher o melhor com os mesmos folds tende a favorecer o vencedor por seleção. Os boxplots e o “melhor cenário” são, portanto, exploratórios. Para estimar desempenho final após escolher configurações, use validação cruzada aninhada ou um conjunto de teste intocado.

As recomendações do dashboard são **regras heurísticas de triagem** sobre resultados binários: recall positivo abaixo de 1%, desvio de acurácia acima de 15 pontos percentuais, FPR acima de 15% ou acurácia abaixo de 80% geram alertas. Esses limites são operacionais do relatório, não limites universais validados por aqueles artigos nem promessas de desempenho. A recomendação do sistema não substitui a análise por classe e a confirmação em conjunto de teste independente.

## Formatos de entrada

### CSV tabular

O carregador preserva o formato da implementação-base:

1. primeira linha: cabeçalho;
2. primeira coluna: identificador da amostra (não usado pelo modelo);
3. segunda coluna: alvo/rótulo;
4. demais colunas: atributos numéricos.

Para treino e teste separados, mantenha os mesmos nomes e a mesma ordem dos atributos. Valores ausentes ou não numéricos que o carregador não consiga converter viram zero; revise o dataset antes de executar.

### LIBSVM

Cada linha deve conter `rótulo índice:valor ...`. A leitura esparsa é convertida para uma matriz densa durante o treinamento. Para classificação, o experimento deve ter exatamente duas classes numéricas. A menor classe ordenada é tratada como negativa e a maior como positiva; por exemplo, em `heart_scale`, `-1` é negativa e `+1` positiva.

## Seleção de atributos

A seleção é opcional e o padrão é **não podar**: todos os atributos fornecidos ao mELM são usados. Escolha um método por execução com `--feature-selection`:

| Método | Comando | Critério aplicado no treino de cada fold |
|---|---|---|
| Sem poda (padrão) | Omitir `--feature-selection` | Usa todos os atributos, sem calcular correlação ou variância para selecionar colunas. |
| Correlação | `--feature-selection correlation -threshold 0.05` | Calcula a correlação de Pearson de cada atributo com o alvo e seleciona os que têm \(|r|\geq\text{threshold}\). Atributos constantes e os que atingem `-leakage-threshold` são removidos. Se nenhum atributo restante atingir o limiar, o script usa todos os atributos não constantes e não suspeitos para evitar uma matriz sem colunas. |
| Variância relativa | `--feature-selection variance --variance-threshold 0.1` | Aplica escala min-max usando mínimo e máximo do treino; mantém os atributos cuja variância normalizada é pelo menos a fração informada da maior variância normalizada. |

O modo `variance` implementa como base o critério do script de poda fornecido pelo orientador. Os valores recomendados no script-base são `0.1`, `0.2` e `0.3`; execute uma rodada separada para cada valor e compare os resultados. Para evitar vazamento, min, max, variância máxima e seleção são recalculados somente no treino de cada fold. O fold de validação recebe a mesma máscara de atributos. Como no script-base, os valores originais dos atributos selecionados entram no modelo; o min-max é usado para calcular o critério de poda.

A definição implementada pode ser escrita assim. Para cada fold, com treino \(X_{treino}\) de \(n\) amostras, calcule \(m_i=\min_k x_{ki}\), \(M_i=\max_k x_{ki}\) e, se \(M_i>m_i\), \(z_{ki}=(x_{ki}-m_i)/(M_i-m_i)\). Em seguida, \(v_i=\frac{1}{n-1}\sum_k(z_{ki}-\bar z_i)^2\) quando \(n>1\), \(v_{max}=\max_i v_i\) e \(r_i=v_i/v_{max}\). Mantêm-se as colunas não constantes com \(r_i\geq\lambda\). O programa interrompe a seleção com erro se não encontrar variância positiva ou se o limiar não selecionar nenhuma coluna. A escala \(z\) é temporária: os valores originais das colunas retidas entram no mELM.

A poda por variância é não supervisionada: ela não usa o rótulo e não mede diretamente poder preditivo. Atributos raros podem ter baixa variância e ainda ser úteis. O filtro de correlação, por sua vez, calcula \(r_i=\operatorname{corr}(X_{treino,i},y_{treino})\) e compara \(|r_i|\) com o limiar de seleção, depois de remover constantes e correlações extremas marcadas como suspeitas. É um filtro univariado: não estima redundância entre atributos e não equivale ao CFS de Hall. Trate os limiares como condições experimentais; se escolher um limiar com base nos resultados, use validação aninhada ou um conjunto independente para estimar o desempenho final.

Os modos são mutuamente exclusivos. `-threshold` é o limiar de correlação e `--variance-threshold` é a fração da variância máxima; ambos devem ser usados com o método correspondente. Para compatibilidade, informar somente um desses limiares sem `--feature-selection` ativa o método correspondente. O valor de `-leakage-threshold` só é usado no modo de correlação; uma correlação quase perfeita é tratada como sinal de atributo suspeito, não como prova de vazamento.

O diretório de resultados contém `melm_feature_audit.csv` com o método, o escore, o limiar e o status de cada atributo por ativação, número de neurônios e fold; `melm_feature_audit_selected.csv` lista os atributos usados. No método de correlação, o escore é o (r) de Pearson assinado; no método de variância, é a variância normalizada dividida pela variância máxima do treino daquele fold. No modo sem poda, todos os atributos aparecem como `used_no_pruning`. Os arquivos são auditorias por fold, não uma única versão global do dataset podado.

## Instalação

Use Python 3.13 para esta versão das dependências (`scikit-learn==1.6.1` suporta Python 3.9 a 3.13). Na pasta `PaperVersion`, crie o ambiente explicitamente com essa versão:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
```

O comando `python --version` deve mostrar Python 3.13.x. Use `python -m pip` para instalar os pacotes no mesmo ambiente que executará o script. Se a instalação mostrar versões diferentes das fixadas em `requirements.txt`, confira se o ambiente `.venv` está ativado e se o comando está sendo executado nesta pasta.

Veja a ajuda completa com:

```powershell
python .\melmParameters.py --help
```

Para uma execução curta com o dataset incluído, sem informar `--output-dir` (a pasta de resultados é criada automaticamente):

```powershell
python .\melmParameters.py -tall .\data\heart_scale -ty 1 -nh 50 -af linear -kfold 5 -sd 42
```

## Opções principais

| Opção | Uso |
|---|---|
| `-tall ARQUIVO` | Dataset completo para validação cruzada; aceita CSV ou LIBSVM |
| `-tr TREINO -ts TESTE` | Arquivos de treino e teste separados, em CSV ou LIBSVM |
| `-ty 1` / `-ty 0` | Classificação binária / regressão |
| `-nh 50,100` | Uma ou mais contagens de neurônios ocultos |
| `-af erosion,dilation` / `-af all` | Uma ou mais ativações ou todas as 12 ativações-base |
| `-kfold 5` | Número de folds; classificação binária usa divisão estratificada |
| `-sd 42` / `--seed 42` | Semente reprodutível; padrão 42 |
| `--feature-selection none` | Sem poda; padrão, usa todos os atributos |
| `--feature-selection correlation -threshold 0.05` | Seleção fold-safe por correlação de Pearson absoluta |
| `--feature-selection variance --variance-threshold 0.1` | Seleção fold-safe por variância min-max relativa; use `0.1`, `0.2` e `0.3` em execuções separadas para comparar |
| `-leakage-threshold 0.999` | No modo `correlation`, limite para marcar atributos suspeitos; padrão 0,999 |
| `-max-time 3600` | Limite estimado por ativação, em segundos |
| `--force-all-kernels` | Executa ativações mesmo que a estimativa exceda `-max-time` |
| `--language pt-br` / `--language en` | Idioma do dashboard HTML; padrão `pt-br` |
| `--output-dir CAMINHO` | Diretório dos resultados; por padrão `results/melm_DATA-HORA` |
| `-virusNorm` | Normalização legada [0,1]→[0.1,0.9], herdada do código original; use apenas quando fizer sentido para o dataset |
| `-sep ";"` | Separador do CSV; aceita detecção automática |
| `-v` | Logs detalhados |

## Idioma do relatório

O HTML é gerado em português por padrão; `--language pt-br` seleciona esse idioma explicitamente. Para o relatório e os gráficos em inglês, use `--language en`, por exemplo:

```powershell
python .\melmParameters.py -tall .\data\heart_scale -ty 1 -nh 50 -af linear -kfold 5 -sd 42 --language en
```

O idioma não altera o treinamento nem as auditorias CSV. O relatório usa o mesmo template com textos localizados; não há uma pasta de código separada para o inglês.

## Estimativa de duração

O estimador mELM usa uma aproximação de custo por ativação e configuração de neurônios baseada em:

\[
\widehat{T}_{ELM}=c_\phi\sum_{H\in\mathcal H}K\left[N_fFH+\max(N_f,H)\min(N_f,H)^2+N_vH\right],
\]

onde \(N_f\) é o número de amostras de treino por fold, \(N_v\) o número de amostras de validação, \(F\) o número de atributos, \(H\) a quantidade de neurônios ocultos, \(\mathcal H\) a lista de configurações testadas, \(K\) o número de folds e \(c_\phi\) uma constante associada à ativação. O primeiro termo aproxima a construção da matriz da camada oculta; o segundo representa o custo dominante de uma SVD da matriz \(N_f\times H\), usada pela pseudo-inversa; o terceiro aproxima a predição no fold. A pseudo-inversa é parte do treinamento ELM descrito por Huang, Zhu e Siew (2006), e o NumPy documenta que `pinv` é calculada por SVD.

Esta forma é a **aproximação matemática sugerida por WPSS e TMA**, coautoras desta adaptação. A implementação atual usa a aproximação simplificada \(N_fFH+N_fH^2+H^3\), multiplicadores heurísticos de 2,5 para ativações fuzzy e 1,6 para bitwise, além do fator de conversão `2.5e-8`; ela não inclui separadamente a predição e não representa todos os regimes de dimensão. Quando \(N_f\geq H\), o termo de SVD se reduz a \(O(N_fH^2)\); quando \(H>N_f\), o custo muda para aproximadamente \(O(HN_f^2)\). Portanto, a fórmula e seus fatores servem para triagem e planejamento, não são uma duração garantida nem uma constante validada pela literatura. Calibre-os comparando estimativas com tempos medidos em diferentes datasets, ativações e tamanhos de camada, registrando o hardware e a biblioteca linear utilizada. `--force-all-kernels` desativa o descarte por limite estimado.

## Resultado e dashboard

Cada execução cria uma pasta `results/melm_DATA-HORA` (ou a indicada em `--output-dir`) com:

- `elm_report_<timestamp>.html` ou `elm_report_en_<timestamp>.html`: dashboard em português ou inglês, conforme `--language`, com abas para visão geral, distribuições, resumo por ativação, diagnósticos (classificação) e auditoria. No resumo e nos diagnósticos, um seletor mostra uma ativação por vez; a impressão/PDF exibe todas as seções e ativações;
- `elm_report_images_<timestamp>/` (ou com prefixo `en_`): imagens associadas ao relatório;
- o CSS e os logotipos do dashboard são referenciados nas pastas `static/` e `templates/` do projeto; o template Jinja é usado durante a geração e não é copiado para a pasta de resultados;
- `melm_feature_audit.csv` e `melm_feature_audit_selected.csv`: auditorias fold-safe do método escolhido ou do uso de todos os atributos.

Os resultados em `results/` e ambientes virtuais estão excluídos pelo `.gitignore`; publique o código e os exemplos necessários, não os artefatos gerados localmente.

Na configuração visível do dashboard, **Poda de atributos** indica se a seleção foi ativada e **Método de poda** identifica `nenhuma`, correlação de Pearson ou variância relativa min-max. Os limiares usados ficam em **Mais detalhes da configuração**. No relatório em inglês, os campos aparecem como **Feature pruning** e **Pruning method**.

Na matriz interna, a convenção é `[[TN, FP], [FN, TP]]`, com linhas verdadeiras e colunas previstas e ordem [negativa, positiva]. A visualização reordena as linhas/colunas para [positiva, negativa], apresenta linhas previstas e colunas reais e mostra sigla, percentual médio e desvio padrão por célula. A matriz não é transposta silenciosamente durante o cálculo.

Na validação cruzada, os boxplots de treino e teste aparecem em painéis separados, com uma linha por ativação e escala comum. Os nomes permanecem como texto HTML e cada linha se ajusta à largura do cartão; em telas estreitas, o nome fica acima da caixa. Os PNGs exportados também são horizontais. Eles resumem os folds do melhor número de neurônios encontrado para cada ativação. Como esse “melhor” é escolhido usando os próprios resultados de validação cruzada, a figura é exploratória. Na execução com arquivos de treino e teste separados, a aba de distribuições é omitida, pois há apenas uma avaliação por cenário. Para estimativa final independente, reserve um conjunto de teste que não participe da escolha de parâmetros.

Na classificação, cada diagnóstico descreve o **melhor cenário daquela ativação** e mostra os indicadores usados. As regras de triagem são verificadas nesta ordem: recall da classe positiva abaixo de 1%, desvio da acurácia entre folds acima de 15 pontos percentuais, taxa de falsos positivos acima de 15% e acurácia abaixo de 80%. O relatório mostra o primeiro alerta aplicável; os limiares são heurísticos e não representam teste estatístico. Os textos são gerados no idioma escolhido. Sem JavaScript, as seções permanecem visíveis em sequência.

## Teste com o `heart_scale`

O arquivo incluído é LIBSVM esparso e pode ser lido diretamente. Um primeiro experimento com as 12 ativações-base, dois tamanhos de camada oculta e cinco folds:

```powershell
python .\melmParameters.py -tall .\data\heart_scale -ty 1 -nh 50,100 -af all -kfold 5 -sd 42 -max-time 3600 --force-all-kernels --output-dir .\results\heart_scale_melm
```

O comando acima usa todos os atributos, que é o baseline sem poda. Para selecionar por correlação, use explicitamente:

```powershell
python .\melmParameters.py -tall .\data\heart_scale -ty 1 -nh 50,100 -af all -kfold 5 -sd 42 --feature-selection correlation -threshold 0.05 -leakage-threshold 0.999 --force-all-kernels --output-dir .\results\heart_scale_correlation
```

Para comparar os valores de variância do script-base, execute uma rodada por limiar e use diretórios distintos:

```powershell
python .\melmParameters.py -tall .\data\heart_scale -ty 1 -nh 50,100 -af all -kfold 5 -sd 42 --feature-selection variance --variance-threshold 0.1 --force-all-kernels --output-dir .\results\heart_scale_variance_01
python .\melmParameters.py -tall .\data\heart_scale -ty 1 -nh 50,100 -af all -kfold 5 -sd 42 --feature-selection variance --variance-threshold 0.2 --force-all-kernels --output-dir .\results\heart_scale_variance_02
python .\melmParameters.py -tall .\data\heart_scale -ty 1 -nh 50,100 -af all -kfold 5 -sd 42 --feature-selection variance --variance-threshold 0.3 --force-all-kernels --output-dir .\results\heart_scale_variance_03
```

O conjunto `heart_scale` contém apenas 270 amostras e 13 atributos; essas execuções servem para conferir o fluxo e comparar ativações nesse dataset, não para generalizar conclusões para outros problemas.

## Referências

- Lima, S. M. L. **Sistema inteligente para classificação de lesões em imagens de mamografia**. Tese de doutorado, UFPE, 2016, seção 5.2.1, equações 55–62. [Repositório da UFPE (PDF)](https://repositorio.ufpe.br/bitstream/123456789/18361/1/tese-completa-Sidney_Lima_v21.pdf).
- Huang, G.-B., Zhu, Q.-Y. e Siew, C.-K. **Extreme learning machine: Theory and applications**. *Neurocomputing*, 70(1–3), 489–501, 2006. [DOI](https://doi.org/10.1016/j.neucom.2005.12.126).
- Pinheiro et al. **Antivirus applied to JAR malware detection based on runtime behaviors**. *Scientific Reports*, 12, 1945, 2022. [DOI](https://doi.org/10.1038/s41598-022-05921-5).
- Serra, J. **Image Analysis and Mathematical Morphology**. Academic Press, 1982. [Registro bibliográfico](https://books.google.com/books?id=BpdTAAAAYAAJ).
- Zadeh, L. A. **Fuzzy sets**. *Information and Control*, 8(3), 338–353, 1965. [DOI](https://doi.org/10.1016/S0019-9958(65)90241-X).
- Scikit-learn. **Removing features with low variance**. [Documentação do `VarianceThreshold`](https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance). A implementação deste projeto usa variância relativa após min-max por fold, não o limiar absoluto padrão do scikit-learn.
- Hall, M. A. **Correlation-based feature selection for machine learning**. Tese de doutorado, University of Waikato, 1999. [Repositório institucional](https://researchcommons.waikato.ac.nz/items/12a40834-bf51-4d87-89ef-d00c2740f0d8).
- Ambroise, C. e McLachlan, G. J. **Selection bias in gene extraction on the basis of microarray gene-expression data**. *PNAS*, 99(10), 6562–6566, 2002. [DOI](https://doi.org/10.1073/pnas.102102699).
- Cawley, G. C. e Talbot, N. L. C. **On over-fitting in model selection and subsequent selection bias in performance evaluation**. *Journal of Machine Learning Research*, 11, 2079–2107, 2010. [Artigo](https://www.jmlr.org/papers/v11/cawley10a.html).
- NumPy. **`numpy.linalg.pinv`**. A documentação registra o uso da decomposição em valores singulares para calcular a pseudo-inversa. [Documentação](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html).

