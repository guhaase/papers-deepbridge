# Experimentos - DeepBridge Fairness Framework

Pasta contendo o plano de experimentos para validação do paper "DeepBridge Fairness: Da Pesquisa à Regulação".

## 📁 Estrutura de Arquivos

```
experimentos/
├── README.md                    # Este arquivo
├── PLANO_EXPERIMENTOS.md        # Plano completo e detalhado (17 seções)
├── CHECKLIST_RAPIDO.md          # Checklist executivo para tracking
├── scripts/                     # Scripts Python para executar experimentos
│   ├── exp1_auto_detection.py
│   ├── exp2_metrics_coverage.py
│   ├── exp3_eeoc_validation.py
│   ├── exp4_case_studies.py
│   ├── exp5_usability.py
│   ├── exp6_performance.py
│   ├── exp7_threshold_opt.py
│   ├── exp8_comparison.py
│   └── utils.py                 # Funções auxiliares
├── data/                        # Datasets e ground truth
│   ├── ground_truth.csv         # Anotações manuais (500 datasets)
│   ├── case_studies/            # Dados dos 4 case studies
│   └── synthetic/               # Datasets sintéticos para testes
├── results/                     # Resultados dos experimentos
│   ├── auto_detection/
│   ├── eeoc_validation/
│   ├── case_studies/
│   ├── usability/
│   ├── performance/
│   └── comparison/
└── reports/                     # Relatórios e visualizações
    ├── experiment_summary.pdf
    ├── figures/                 # Gráficos e tabelas
    └── reproduction_guide.md    # Como reproduzir experimentos
```

## 🚀 Quick Start

### 1. Preparação do Ambiente

```bash
# Criar ambiente virtual
python -m venv venv_experiments
source venv_experiments/bin/activate  # Linux/Mac
# ou
venv_experiments\Scripts\activate  # Windows

# Instalar dependências
pip install deepbridge
pip install aif360 fairlearn aequitas  # Ferramentas para comparação
pip install pandas numpy scipy scikit-learn
pip install matplotlib seaborn plotly
pip install pytest pytest-cov

# Verificar instalação
python -c "from deepbridge import DBDataset; print('DeepBridge OK')"
```

### 2. Executar Experimento Exemplo

```bash
# Teste rápido com COMPAS dataset
cd scripts/
python exp4_case_studies.py --dataset compas --quick

# Saída esperada:
# ✅ Atributos detectados: ['race', 'sex', 'age'] (3/3)
# ✅ Tempo de análise: 7.2 min
# ✅ Violação detectada: FPR difference 22pp
# ✅ Threshold ótimo: 0.62 (FPR → 8pp)
```

### 3. Ver Checklist de Progresso

```bash
cat CHECKLIST_RAPIDO.md
```

## 📊 Experimentos Principais

### Experimentos Críticos (⭐ Prioridade MÁXIMA)

1. **Auto-Detecção** (`exp1_auto_detection.py`)
   - 500 datasets
   - Target: F1 ≥ 0.90
   - Tempo estimado: 20h

2. **Verificação EEOC/ECOA** (`exp3_eeoc_validation.py`)
   - Regra 80% + Question 21
   - Target: 100% precisão
   - Tempo estimado: 8h

3. **Case Studies** (`exp4_case_studies.py`)
   - COMPAS, German Credit, Adult, Healthcare
   - Target: 75-79% economia de tempo
   - Tempo estimado: 40h

4. **Usabilidade** (`exp5_usability.py`)
   - N=20 participantes
   - Target: SUS ≥ 85
   - Tempo estimado: 60h (inclui recrutamento)

5. **Performance** (`exp6_performance.py`)
   - 3 tamanhos de datasets
   - Target: Speedup ≥ 2.9x
   - Tempo estimado: 12h

6. **Comparação** (`exp8_comparison.py`)
   - AIF360, Fairlearn, Aequitas
   - Target: Feature matrix validada
   - Tempo estimado: 16h

**Total estimado**: ~156 horas (4 semanas full-time)

## 📖 Documentos

### [PLANO_EXPERIMENTOS.md](PLANO_EXPERIMENTOS.md)
Documento master com:
- 17 seções de experimentos detalhados
- Metodologias completas
- Métricas de validação
- Critérios de sucesso
- Timeline de 18 semanas

### [CHECKLIST_RAPIDO.md](CHECKLIST_RAPIDO.md)
Checklist executivo com:
- 6 experimentos críticos
- Tabela de validação de claims
- Red flags e riscos
- Timeline resumido

## 🎯 Claims do Paper a Validar

| Claim | Valor | Experimento |
|-------|-------|-------------|
| Auto-detecção F1-Score | 0.90 | 1.1 |
| Auto-detecção Precision | 0.92 | 1.1 |
| Auto-detecção Recall | 0.89 | 1.1 |
| Métricas totais | 15 (4+11) | 2.1 |
| SUS Score | 85.2 | 5.1 |
| NASA-TLX | 32.1 | 5.2 |
| Taxa de sucesso | 95% | 5.3 |
| Time-to-insight | 10.2 min | 5.4 |
| Speedup médio | 2.9x | 6.1 |
| Redução de memória | 40-42% | 6.2 |
| COMPAS tempo | 7.2 min | 4.1 |
| German Credit tempo | 5.8 min | 4.2 |
| Adult tempo | 12.4 min | 4.3 |
| Healthcare tempo | 9.1 min | 4.4 |

## ⚠️ Critérios Mínimos para Publicação

Para o paper ser aceito no FAccT 2026, os seguintes critérios DEVEM ser atendidos:

### ✅ Obrigatórios (Deal-breakers):
1. **EEOC/ECOA**: 100% precisão (sem margem de erro)
2. **SUS**: ≥ 75 (claim: 85.2)
3. **Speedup**: ≥ 2.0x (claim: 2.9x)
4. **Case Studies**: 4/4 completos
5. **Usabilidade N**: ≥ 15 participantes (claim: 20)

### ⭐ Recomendados:
1. **F1 auto-detecção**: ≥ 0.85 (claim: 0.90)
2. **Taxa de sucesso**: ≥ 85% (claim: 95%)
3. **Datasets**: ≥ 300 (claim: 500)

## 🔬 Execução dos Experimentos

### Ordem Recomendada:

```bash
# Semana 1-2: Setup
scripts/setup_environment.sh
scripts/collect_datasets.py

# Semana 3-4: Auto-detecção
python scripts/exp1_auto_detection.py --full

# Semana 5-6: Métricas + EEOC
python scripts/exp2_metrics_coverage.py
python scripts/exp3_eeoc_validation.py

# Semana 7-9: Case Studies
python scripts/exp4_case_studies.py --all

# Semana 10-12: Usabilidade
python scripts/exp5_usability.py --recruit --execute

# Semana 13-14: Performance
python scripts/exp6_performance.py --all-sizes

# Semana 15: Comparação
python scripts/exp8_comparison.py --tools all

# Semana 16: Robustness
python scripts/exp9_edge_cases.py

# Semana 17-18: Análise e Relatórios
python scripts/generate_reports.py --output reports/
```

## 📈 Tracking de Progresso

Use o arquivo `CHECKLIST_RAPIDO.md` para tracking diário:

```bash
# Ver status atual
grep "⬜\|🔄\|✅" CHECKLIST_RAPIDO.md

# Atualizar status de um experimento
# ⬜ Não iniciado → 🔄 Em progresso → ✅ Completo
```

## 🤝 Contribuindo com Experimentos

Se você for executar os experimentos:

1. **Clone o ambiente**:
   ```bash
   git clone <repo>
   cd papers/02_Fairness_Framework/experimentos
   ```

2. **Siga o PLANO_EXPERIMENTOS.md** para metodologia exata

3. **Salve resultados em `results/`** seguindo estrutura:
   ```
   results/
   ├── <experimento_id>/
   │   ├── raw_data.csv          # Dados brutos
   │   ├── processed_data.csv    # Dados processados
   │   ├── analysis.txt          # Análise textual
   │   └── figures/              # Gráficos
   ```

4. **Documente problemas** em `issues.md`

## 📞 Contato

**Responsável**: [Adicionar nome e email]

**Dúvidas sobre experimentos**: Consulte PLANO_EXPERIMENTOS.md seção correspondente

**Bugs ou issues**: Abra issue no repositório

## 📚 Referências

### Papers Base:
- Bellamy et al. (2018) - AI Fairness 360
- Bird et al. (2020) - Fairlearn
- Saleiro et al. (2018) - Aequitas

### Metodologias:
- Brooke (1996) - System Usability Scale
- Hart & Staveland (1988) - NASA Task Load Index

### Datasets:
- COMPAS - ProPublica
- German Credit - UCI Repository
- Adult Income - UCI Repository

---

**Última atualização**: 2025-12-06

**Status do Projeto**: ⬜ Não iniciado

**Prazo**: Submissão FAccT 2026 (verificar deadline exato)

**Boa sorte com os experimentos! 🚀**
