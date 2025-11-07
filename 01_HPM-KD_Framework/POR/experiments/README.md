# HPM-KD Framework - Experiments

Este diretório contém todos os scripts experimentais organizados por categoria.

## 📂 Estrutura

```
experiments/
├── sklearn_validation/     # Validação inicial com sklearn (COMPLETO ✅)
├── cnn_baseline/           # Baselines CNN (Em andamento ⏳)
├── cnn_hpmkd/              # HPM-KD com CNN (Próximo ⏳)
├── evaluation/             # Scripts de avaliação e figuras
├── lib/                    # Bibliotecas compartilhadas
└── results/                # Resultados organizados
```

## 🚀 Quick Start

### 1. Validação sklearn (Rápido - 2 minutos)
```bash
cd /home/guhaase/projetos/DeepBridge
python papers/01_HPM-KD_Framework/POR/experiments/sklearn_validation/example_hpmkd_experiment.py
```

### 2. Baseline CNN (Médio - 30-45 minutos)
```bash
cd /home/guhaase/projetos/DeepBridge
poetry run python papers/01_HPM-KD_Framework/POR/experiments/cnn_baseline/train_teacher.py --epochs 20
```

### 3. HPM-KD completo (Demorado - 40-60 minutos)
```bash
cd /home/guhaase/projetos/DeepBridge
poetry run python papers/01_HPM-KD_Framework/POR/experiments/cnn_hpmkd/train_hpmkd.py \
    --teacher models/teacher_resnet18_best.pth \
    --student mobilenet \
    --use-progressive --use-adaptive-temp \
    --epochs 20
```

## 📖 Documentação Completa

Ver `EXPERIMENTAL_ORGANIZATION.md` na pasta raiz do projeto para documentação detalhada.

## ✅ Status Atual

- ✅ **sklearn validation**: COMPLETO (91.67% accuracy, +23.13pp melhoria)
- ⏳ **CNN baseline**: Em andamento (8 processos rodando)
- ⏳ **HPM-KD CNN**: Aguardando baseline
- ⏳ **Ablation studies**: Pendente
- ⏳ **Multi-dataset**: Pendente

## 📊 Resultados

Resultados salvos em `results/`:
- `sklearn/`: Resultados de validação sklearn
- `cnn/`: Resultados CNN
- `figures/`: Figuras geradas
- `tables/`: Tabelas de comparação
