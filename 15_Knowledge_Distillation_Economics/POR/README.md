# Paper 15: Destilacao de Conhecimento para Economia

## 📋 Informacoes Basicas

**Titulo**: Knowledge Distillation for Economics: Trading Complexity for Interpretability in Econometric Models

**Titulo em Portugues**: Destilacao de Conhecimento para Economia: Negociando Complexidade por Interpretabilidade em Modelos Econometricos

**Conferencias Alvo**:
- **Journal of Econometrics** - PRINCIPAL
- Review of Economic Studies
- American Economic Review (se resultados forem excepcionais)
- NeurIPS (Economics and Computation track)

**Status**: Completo - Pronto para revisao

**Tamanho**: ~10 paginas (limite estabelecido)

---

## 🎯 Contribuicao Principal

Framework de destilacao de conhecimento econometrica que transfere conhecimento de modelos complexos (XGBoost, Neural Networks) para modelos interpretaveis (GAM, Linear), preservando simultaneamente: (1) intuicao economica, (2) restricoes economicas, (3) estabilidade de coeficientes para inferencia estatistica.

### Principais Resultados

- ✅ **Perda minima de acuracia**: 2-5% vs. modelos teacher complexos
- ✅ **Alta interpretabilidade**: Economic Interpretability Score de 91% (vs. 68% KD padrao)
- ✅ **Conformidade economica**: 95%+ das restricoes teoricas preservadas
- ✅ **Estabilidade robusta**: Coeficientes com CV < 0.15 em todos case studies
- ✅ **Superioridade vs. baselines**: +8-12% AUC vs. modelos lineares tradicionais

---

## 📊 Estrutura do Paper

### Secao 1: Introducao
- **Motivacao**: Dilema entre acuracia de ML complexo e interpretabilidade de econometria
- **Problema**: Research em KD ignora completamente requisitos economicos
- **Nossa Solucao**: Framework de destilacao com restricoes economicas integradas
- **Contribuicoes**:
  1. Framework de destilacao econometrica (primeira abordagem)
  2. Preservacao de restricoes economicas (monotonia, sinais)
  3. Analise de estabilidade de coeficientes via bootstrap
  4. Deteccao de quebras estruturais
  5. Validacao em tres dominios economicos
  6. Implementacao no DeepBridge (open-source)

### Secao 2: Fundamentacao e Trabalhos Relacionados
- **Econometria classica**: Linear, Logit, GAM
- **Knowledge Distillation**: Framework de Hinton et al., variantes
- **ML Interpretavel em Economia**: Mullainathan & Spiess, Athey & Imbens
- **Gap na Literatura**: Nenhum trabalho combina KD + restricoes economicas + estabilidade
- **Quebras Estruturais**: Chow test, CUSUM, nossa abordagem

### Secao 3: Design do Framework
- **Componentes**:
  1. Teacher Training (XGBoost, NN, RF)
  2. Economic Constraint Encoder (sign, monotonicity, magnitude)
  3. Constrained Distillation Engine (loss modificada)
  4. Coefficient Stability Analyzer (bootstrap)
  5. Structural Break Detector (rolling windows)
- **Loss Function**: $\mathcal{L}_{\text{econ}} = \alpha \mathcal{L}_{\text{KD}} + \beta \mathcal{L}_{\text{constraint}} + \gamma \mathcal{L}_{\text{hard}}$
- **Student Models**: GAM (preferido) vs. Linear
- **Integracao DeepBridge**: API declarativa para restricoes

### Secao 4: Implementacao
- **Stack**: Python 3.9+, DeepBridge, statsmodels, scikit-learn, Optuna
- **Modulos**:
  - `economics/constraints.py`: Codificacao de restricoes
  - `economics/distillation.py`: Engine de destilacao
  - `economics/stability.py`: Analise bootstrap
  - `economics/breaks.py`: Deteccao de quebras
  - `economics/metrics.py`: Metricas especializadas
- **Otimizacoes**: Caching, paralelizacao, bootstrap em subsamples

### Secao 5: Avaliacao
- **Case Study 1: Risco de Credito**
  - Dataset: 250k emprestimos, 42 features
  - Resultados: AUC 0.829 (vs. 0.847 teacher), CV 0.116, 96% compliance
  - Quebra estrutural detectada: Q4 2008 (crise financeira)

- **Case Study 2: Economia do Trabalho**
  - Dataset: 180k individuos, 38 features
  - Resultados: AUC 0.783 (vs. 0.801 teacher), CV 0.124, 96% compliance

- **Case Study 3: Economia da Saude**
  - Dataset: 95k pacientes, 51 features
  - Resultados: AUC 0.754 (vs. 0.779 teacher), Interp. Score 93%

- **Ablation Study**: Restricoes economicas custam 0.7% AUC mas ganham +20pp compliance

### Secao 6: Discussao
- **Principais Descobertas**: Trade-off favoravel (2-5% acuracia por interpretabilidade total)
- **Implicacoes Praticas**:
  - Industria financeira: Conformidade regulatoria (Basel III)
  - Policy makers: Analise de impacto com modelos acurados
  - Academia: Ponte entre ML e econometria
- **Limitacoes**:
  1. Especificacao manual de restricoes
  2. GAMs nao capturam interacoes complexas
  3. Causalidade vs. correlacao
  4. Escalabilidade (bootstrap caro)
  5. Generalidade de restricoes
- **Trabalhos Futuros**: Causal distillation, GA²Ms, multi-task, novos dominios

### Secao 7: Conclusao
- **Sintese**: Framework reconcilia acuracia ML com rigor econometrico
- **Impacto**: Nova geracao de modelos economicos: data-driven + teoricamente fundamentados
- **Disponibilidade**: DeepBridge open-source, documentacao completa
- **Mensagem Final**: Nao e necessario escolher entre ML e interpretabilidade

---

## 🔬 Metodologia

### Restricoes Economicas Implementadas

| Tipo | Descricao | Exemplo |
|------|-----------|---------|
| Sign Constraints | Coeficientes com sinal especifico | Income → Default (negativo) |
| Monotonicity | Funcoes GAM monotonicamente crescentes/decrescentes | Age → Default (crescente ate 65) |
| Magnitude Bounds | Limites para efeitos | Interest Rate effect: [0.5, 2.0] |

### Metricas de Avaliacao

```
Economic Interpretability Score =
  0.4 × Constraint Compliance +
  0.3 × Coefficient Stability +
  0.3 × Sign Stability
```

### Criterios de Estabilidade

- CV (Coefficient of Variation) < 0.15
- Sign stability ≥ 95% em bootstrap samples
- Intervalo de confianca nao cruza zero (para efeitos nao-nulos)

---

## 📈 Principais Resultados (Agregados)

| Metrica | Media | Min | Max |
|---------|-------|-----|-----|
| Perda de AUC vs. Teacher | -2.8% | -1.9% | -3.2% |
| Ganho de AUC vs. GAM Vanilla | +3.7% | +3.1% | +4.2% |
| Avg CV (Coef. Stability) | 0.118 | 0.103 | 0.129 |
| Compliance Economica | 95.3% | 94% | 97% |
| Economic Interp. Score | 91.2% | 88% | 94% |

---

## 💻 Implementacao

### Exemplo de Uso

```python
from deepbridge.distillation import AutoDistiller
from deepbridge.distillation.economics import (
    EconomicConstraints,
    StabilityAnalyzer,
    StructuralBreakDetector
)
from deepbridge.utils.model_registry import ModelType

# 1. Train teacher
teacher = xgboost.XGBClassifier()
teacher.fit(X_train, y_train)

# 2. Define economic constraints
constraints = EconomicConstraints()
constraints.add_sign('income', sign=-1,
                     justification="Higher income -> Lower default risk")
constraints.add_monotonicity('age', direction='increasing', bounds=(18, 65))

# 3. Economic distillation
distiller = AutoDistiller.from_teacher(
    teacher=teacher,
    student_type=ModelType.GAM_CLASSIFIER,
    constraints=constraints,
    temperature=2.0,
    alpha=0.5
)
student = distiller.fit(X_train, y_train)

# 4. Stability analysis
stability = StabilityAnalyzer(n_bootstrap=1000)
results = stability.analyze(distiller, X_train, y_train)

print(f"Avg CV: {np.mean(results['cv']):.3f}")
print(f"Compliance: {constraints.compliance_rate(student, X_train):.1%}")

# 5. Detect structural breaks
break_detector = StructuralBreakDetector(window_size=500)
breaks = break_detector.detect(X_train, y_train, time_var='date')
```

### Dependencias

- Python 3.9+
- DeepBridge >= 0.5.0
- statsmodels >= 0.13 (GAM implementation)
- scikit-learn >= 1.0
- NumPy >= 1.21
- SciPy >= 1.7 (testes estatisticos)
- Optuna >= 3.0 (otimizacao)
- joblib >= 1.1 (paralelizacao)

---

## 📝 Como Compilar

### Prerequisitos

```bash
# Instalar LaTeX (Ubuntu/Debian)
sudo apt-get install texlive-full

# Ou usar Docker
docker pull texlive/texlive:latest
```

### Compilacao

```bash
# Metodo 1: Usar script automatizado
cd /home/guhaase/projetos/DeepBridge/papers/15_Knowledge_Distillation_Economics/POR
./compile.sh

# Metodo 2: Compilacao manual
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Verificacao

```bash
# Verificar PDF gerado
ls -lh main.pdf

# Ver numero de paginas
pdfinfo main.pdf | grep Pages
```

---

## 🎨 Figuras e Tabelas

### Tabelas Principais

1. **Tab 1**: Datasets de Avaliacao (3 dominios)
2. **Tab 2**: Abordagens de Knowledge Distillation
3. **Tab 3**: Restricoes Economicas - Credito
4. **Tab 4**: Resultados - Risco de Credito
5. **Tab 5**: Estabilidade de Coeficientes - Credito
6. **Tab 6**: Resultados - Economia do Trabalho
7. **Tab 7**: Resultados - Economia da Saude
8. **Tab 8**: Trade-off Agregado - Tres Dominios
9. **Tab 9**: Ablation Study
10. **Tab 10**: Interpretacao de Compliance Scores

### Algoritmos

1. **Alg 1**: Constrained Economic Distillation

---

## 🔗 Referencias Principais

1. **Hinton et al. (2015)**: "Distilling the Knowledge in a Neural Network"
2. **Mullainathan & Spiess (2017)**: "Machine Learning: An Applied Econometric Approach"
3. **Athey & Imbens (2019)**: "Machine Learning Methods Economists Should Know About"
4. **Hastie & Tibshirani (1987)**: "Generalized Additive Models: Some Applications"
5. **Rudin (2019)**: "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions"
6. **Pearl (2009)**: "Causality"
7. **Angrist & Pischke (2009)**: "Mostly Harmless Econometrics"

---

## 📊 Comparacao com Ferramentas Existentes

| Feature | LIME | SHAP | InterpretML | EconML | **Ours** |
|---------|------|------|-------------|--------|----------|
| Intrinseco | ✗ | ✗ | ✓ | ✓ | ✓ |
| Restricoes Econ. | ✗ | ✗ | ✗ | Parcial | ✓ |
| Estabilidade | ✗ | ✗ | ✗ | ✓ | ✓ |
| Destilacao | ✗ | ✗ | ✗ | ✗ | ✓ |
| Quebras Estruturais | ✗ | ✗ | ✗ | ✗ | ✓ |

---

## 🌟 Diferenciais

### vs. Knowledge Distillation Classica
- Adiciona restricoes economicas
- Valida estabilidade de coeficientes
- Detecta quebras estruturais

### vs. Econometria Tradicional
- Alcanca acuracia superior (+8-12% AUC)
- Destila conhecimento de modelos complexos
- Mantem interpretabilidade

### vs. Explainable AI (SHAP/LIME)
- Modelo intrinsecamente interpretavel (nao post-hoc)
- Coeficientes estaveis para inferencia
- Conformidade com teoria economica

### vs. EconML
- Foco em destilacao (nao apenas causal inference)
- Framework completo para interpretabilidade economica
- Deteccao de quebras estruturais

---

## 📧 Contato

Para questoes sobre este paper:
- **Repositorio**: github.com/deepbridge/deepbridge
- **Documentacao**: docs.deepbridge.ai/economics
- **Issues**: github.com/deepbridge/deepbridge/issues

---

## 📄 Licenca

MIT License - Ver arquivo LICENSE para detalhes

---

**Ultima Atualizacao**: Dezembro 2025

**Paginas**: ~10 (dentro do limite estabelecido)

**Status**: ✅ Completo e pronto para submissao
