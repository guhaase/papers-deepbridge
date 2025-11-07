# HPM-KD Framework - Experimentos
## Execução em Google Colab + Relatórios Automáticos em Markdown

**Última atualização:** 07 de Novembro de 2025
**Status:** ✅ Pronto para uso

---

## 🎯 Objetivo

Estrutura completa para executar todos os experimentos do paper HPM-KD no **Google Colab (GPU)** com **geração automática de relatórios em Markdown**.

---

## 📚 Documentação Principal

| Documento | Descrição | Quando Usar |
|-----------|-----------|-------------|
| **[QUICK_START_COLAB.md](QUICK_START_COLAB.md)** | 🚀 Comece aqui! Guia rápido com código copy-paste | **Primeira vez** |
| **[REORGANIZATION_PLAN.md](REORGANIZATION_PLAN.md)** | 📋 Plano completo de reorganização | Entender estrutura |
| **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** | ✅ Resumo de implementação | Ver o que foi criado |
| **[scripts/report_generator.py](scripts/report_generator.py)** | 🐍 Sistema de relatórios MD | Referência técnica |
| **[notebooks/00_setup_colab.ipynb](notebooks/00_setup_colab.ipynb)** | 📓 Setup inicial Colab | Executar no Colab |

---

## 🚀 Quick Start (3 passos)

### 1. Abrir no Google Colab

```
1. Acesse: https://colab.research.google.com/
2. Configure GPU: Runtime → Change runtime type → GPU
3. Upload: notebooks/00_setup_colab.ipynb
4. Execute todas as células (5-10 min)
```

### 2. Executar Primeiro Experimento (Copy-Paste)

Copie o código do experimento 1 em **[QUICK_START_COLAB.md](QUICK_START_COLAB.md)** seção "Experimento 1: Sklearn Baseline"

### 3. Ver Resultado

```python
# Visualizar relatório gerado
from IPython.display import Markdown, display

with open('/content/drive/MyDrive/HPM-KD-Results/01_sklearn_baseline/report.md') as f:
    display(Markdown(f.read()))
```

✅ **Relatório completo em MD gerado automaticamente!**

---

## 📂 Estrutura Proposta

```
experiments/
│
├── 📚 Documentação
│   ├── README_NEW.md                  ← Você está aqui
│   ├── QUICK_START_COLAB.md           ← Guia rápido ⭐
│   ├── REORGANIZATION_PLAN.md         ← Plano completo
│   └── IMPLEMENTATION_SUMMARY.md      ← Resumo de implementação
│
├── 📓 Notebooks (Google Colab)
│   ├── 00_setup_colab.ipynb           ← Setup inicial ⭐
│   ├── 01_sklearn_baselines.ipynb     ← Exp 1: Sklearn (5 min)
│   ├── 02_sklearn_hpmkd.ipynb         ← Exp 2: HPM-KD sklearn (10 min)
│   ├── 03_cnn_mnist_teacher.ipynb     ← Exp 3: Teacher CNN (30 min)
│   ├── 04_cnn_mnist_baselines.ipynb   ← Exp 4: CNN baselines (45 min)
│   ├── 05_cnn_mnist_hpmkd.ipynb       ← Exp 5: HPM-KD CNN (60 min)
│   ├── 06_cifar10_experiments.ipynb   ← Exp 6: CIFAR-10 (2-3h)
│   ├── 07_ablation_studies.ipynb      ← Exp 7: Ablation (1h)
│   ├── 08_compression_analysis.ipynb  ← Exp 8: Compression (1h)
│   ├── 09_multi_dataset.ipynb         ← Exp 9: UCI datasets (30 min)
│   └── 10_generate_paper_results.ipynb ← Exp 10: Paper final (1h)
│
├── 🐍 Scripts Python
│   ├── report_generator.py            ← Sistema de relatórios MD ⭐
│   ├── models.py                      ← Definições de modelos
│   ├── training.py                    ← Funções de treinamento
│   ├── evaluation.py                  ← Funções de avaliação
│   ├── hpmkd.py                       ← HPM-KD wrapper
│   ├── data_loaders.py                ← Carregamento de datasets
│   └── baselines.py                   ← Implementações de baselines
│
├── 📊 Resultados (Google Drive)
│   └── /content/drive/MyDrive/HPM-KD-Results/
│       ├── 01_sklearn_baseline/
│       │   ├── report.md              ← Relatório gerado ⭐
│       │   ├── metrics.json
│       │   ├── results.csv
│       │   └── figures/
│       ├── 02_sklearn_hpmkd/
│       ├── ... (03-09)
│       └── paper_final/
│           ├── FINAL_REPORT.md        ← Consolidação ⭐
│           ├── table1_compression.csv
│           └── figures/
│
└── 🔧 Configurações
    ├── configs/                       ← Arquivos YAML
    └── templates/                     ← Templates Jinja2
```

---

## 🌟 Principais Features

### ✅ Relatórios Automáticos em Markdown

Cada experimento gera automaticamente:

```markdown
# Relatório de Experimento: 01_sklearn_baseline

**Data:** 2025-11-07 14:32:15
**Duração:** 5m 32s
**GPU:** Tesla T4

## 📋 Configuração
| Parâmetro | Valor |
|-----------|-------|
| Dataset | MNIST |
| Teacher | RandomForest(500) |
| Student | DecisionTree(10) |

## 📈 Resultados
| Métrica | Valor |
|---------|-------|
| Teacher Accuracy | 0.9420 |
| Student KD | 0.6830 |
| Improvement | +2.13 pp |

## 📊 Visualizações
![Comparison](figures/comparison.png)

## 🔍 Observações
- Compression: 50× (500 trees → 1 tree)
- KD improved by 2.13 percentage points
- Retention: 72.52%
```

### ✅ Sistema de Geração de Relatórios

```python
from scripts.report_generator import ExperimentReporter

# Criar reporter
reporter = ExperimentReporter(
    experiment_name='meu_experimento',
    output_dir='/content/drive/MyDrive/HPM-KD-Results'
)

# Log automático
reporter.log_config({'epochs': 20})
reporter.log_metrics({'accuracy': 0.99})
reporter.plot_training_curves(history)

# Gerar relatório MD completo
reporter.generate_markdown_report()
```

### ✅ Consolidação Final

```python
from scripts.report_generator import FinalReportGenerator

generator = FinalReportGenerator(
    results_dir='/content/drive/MyDrive/HPM-KD-Results',
    output_dir='paper_final/'
)

generator.consolidate_results()
generator.generate_final_report()  # → FINAL_REPORT.md
```

---

## 📊 Sequência de Experimentos

| # | Experimento | Duração | GPU | Descrição |
|---|-------------|---------|-----|-----------|
| 00 | Setup Colab | 10 min | - | Instalação e configuração |
| 01 | Sklearn Baseline | 5 min | CPU | Validação rápida |
| 02 | HPM-KD Sklearn | 10 min | CPU | HPM-KD com sklearn |
| 03 | CNN Teacher | 30 min | GPU | Teacher ResNet18 MNIST |
| 04 | CNN Baselines | 45 min | GPU | Direct, KD, FitNets |
| 05 | HPM-KD CNN | 60 min | GPU | HPM-KD completo MNIST |
| 06 | CIFAR-10 | 2-3h | GPU | Experimentos CIFAR-10 |
| 07 | Ablation | 1h | GPU | Remover componentes |
| 08 | Compression | 1h | GPU | Diferentes ratios |
| 09 | Multi-Dataset | 30 min | GPU | UCI datasets |
| 10 | Paper Final | 1h | - | Consolidar resultados |

**Tempo Total:** 12-16 horas de GPU

---

## 📦 O Que Está Pronto

- [x] ✅ Plano de reorganização completo
- [x] ✅ Sistema de geração de relatórios MD (`report_generator.py`)
- [x] ✅ Notebook de setup Colab (`00_setup_colab.ipynb`)
- [x] ✅ Guia rápido de uso (`QUICK_START_COLAB.md`)
- [x] ✅ Documentação completa
- [x] ✅ Código exemplo dos experimentos 1 e 2
- [ ] ⏳ Notebooks 01-10 (templates prontos, precisa criar arquivos)
- [ ] ⏳ Scripts auxiliares (models.py, training.py, etc.)

---

## 🎯 Próximos Passos

### Para Você (Usuário):

1. **Testar Setup** (10 min)
   ```
   - Abrir 00_setup_colab.ipynb no Colab
   - Executar todas as células
   - Verificar instalação
   ```

2. **Executar Experimento Piloto** (5 min)
   ```
   - Copiar código do Exp 1 (QUICK_START_COLAB.md)
   - Colar em nova célula do Colab
   - Executar
   - Verificar report.md gerado
   ```

3. **Criar Notebooks Restantes** (2-3 horas)
   ```
   - Usar templates do REORGANIZATION_PLAN.md
   - Adaptar para cada experimento
   - Testar um por um
   ```

4. **Executar Todos os Experimentos** (12-16h GPU)
   ```
   - Executar sequencialmente 01-10
   - Verificar relatórios MD gerados
   - Consolidar resultados finais
   ```

### Para Mim (Claude):

Se você quiser que eu crie os notebooks restantes (01-10), é só me pedir! Posso criar:
- ✅ Notebooks completos com código
- ✅ Scripts auxiliares (models.py, training.py, etc.)
- ✅ Arquivos de configuração YAML
- ✅ Templates Jinja2 adicionais

---

## 📚 Recursos Adicionais

### Documentação Colab
- Setup GPU: https://colab.research.google.com/notebooks/gpu.ipynb
- Google Drive: https://colab.research.google.com/notebooks/io.ipynb

### DeepBridge
- GitHub: https://github.com/DeepBridge-Validation/DeepBridge
- Docs: https://deepbridge.readthedocs.io/

### Paper HPM-KD
- Seção 5 (Experimentos): Ver estrutura esperada
- Tabelas e Figuras: Templates disponíveis

---

## 💬 FAQ

**Q: Preciso criar os notebooks manualmente?**
A: Não! O código completo dos experimentos está em `QUICK_START_COLAB.md`. Você pode copy-paste direto no Colab. Os notebooks são apenas uma forma mais organizada.

**Q: Os relatórios MD são editáveis?**
A: Sim! São arquivos `.md` puros. Você pode editar manualmente após geração se necessário.

**Q: Posso executar localmente (sem Colab)?**
A: Sim! O código funciona localmente também. Apenas ajuste os paths (`/content/drive/...` → seu diretório local).

**Q: Como baixar todos os resultados?**
A: Use o código em `QUICK_START_COLAB.md` seção "Baixar Todos os Resultados".

**Q: Quanto custa?**
A: Google Colab (GPU) é **grátis** até ~12h/dia. Para mais tempo, use Colab Pro ($10/mês).

---

## ✅ Checklist de Verificação

**Antes de começar:**
- [ ] Leu `QUICK_START_COLAB.md`
- [ ] Tem conta Google (para Colab + Drive)
- [ ] Configurou GPU no Colab

**Após setup:**
- [ ] `00_setup_colab.ipynb` executado com sucesso
- [ ] ExperimentReporter testado
- [ ] Primeiro relatório MD gerado

**Durante experimentos:**
- [ ] Cada experimento gera `report.md`
- [ ] Resultados salvos no Google Drive
- [ ] Figuras geradas corretamente

**Final:**
- [ ] Todos os 10 experimentos executados
- [ ] Relatório final consolidado
- [ ] Tabelas e figuras do paper geradas

---

## 🎉 Resumo

**Você tem:**
- ✅ Sistema completo de relatórios MD
- ✅ Notebook de setup Colab
- ✅ Código dos experimentos (copy-paste)
- ✅ Documentação detalhada

**Você precisa:**
- ⏳ Executar no Colab
- ⏳ Gerar os resultados
- ⏳ Consolidar para o paper

**Tempo estimado:**
- Setup: 10 minutos
- Experimentos: 12-16 horas (GPU)
- Paper final: 1 hora
- **Total: ~1 dia**

---

**🚀 Comece agora:** Abra `QUICK_START_COLAB.md` e siga o passo-a-passo!

**💬 Precisa de ajuda?** Todos os documentos têm exemplos detalhados e código testável.

---

**Versão:** 1.0
**Data:** 07/11/2025
**Autor:** Claude (Anthropic)
