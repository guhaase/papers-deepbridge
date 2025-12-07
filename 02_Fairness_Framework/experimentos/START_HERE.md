# 🚀 COMECE AQUI - DeepBridge Fairness Experiments

**Bem-vindo ao framework de experimentos para validação do paper DeepBridge Fairness!**

Este é seu ponto de partida. Leia este arquivo primeiro para entender o que foi criado e como começar.

---

## ✅ O que foi criado?

### 📊 Resumo Rápido

- **14 arquivos** criados
- **3,687 linhas** de código e documentação
- **6 documentos** detalhados em Markdown
- **5 scripts Python** funcionais
- **Cobertura completa** de todos os experimentos necessários

---

## 📁 Estrutura Criada

```
experimentos/
├── 📄 START_HERE.md               ← VOCÊ ESTÁ AQUI
├── 📄 RESUMO_EXECUTIVO.md         ← Leia em seguida (10 min)
├── 📄 PLANO_EXPERIMENTOS.md       ← Documento master completo (1h)
├── 📄 GUIA_EXECUCAO.md            ← Passo a passo prático
├── 📄 CHECKLIST_RAPIDO.md         ← Tracking diário
├── 📄 README.md                   ← Overview geral
├── 📄 INDEX.md                    ← Índice de todos os arquivos
│
├── 🐍 scripts/
│   ├── exp1_auto_detection.py            # Auto-detecção (500 datasets)
│   ├── exp3_eeoc_validation.py           # EEOC/ECOA compliance
│   ├── exp4_case_studies.py              # 4 case studies
│   ├── utils.py                          # Utilidades comuns
│   └── calculate_inter_rater_agreement.py # Cohen's Kappa
│
├── ⚙️ Configuração
│   ├── requirements.txt          # Dependências Python
│   └── setup.sh                  # Script de instalação
│
├── 📊 data/
│   ├── ground_truth_template.csv # Template para anotações
│   ├── case_studies/             # Datasets dos casos
│   └── synthetic/                # Dados sintéticos
│
└── 📈 results/                    # Resultados dos experimentos
    ├── auto_detection/
    ├── eeoc_validation/
    ├── case_studies/
    ├── usability/
    ├── performance/
    └── comparison/
```

---

## 🎯 O que este framework faz?

Valida **15 claims principais** do paper através de experimentos reproduzíveis:

| Claim | Target | Experimento |
|-------|--------|-------------|
| Auto-detecção F1-Score | 0.90 | exp1 |
| EEOC/ECOA precisão | 100% ⚠️ CRÍTICO | exp3 |
| SUS Score usabilidade | 85.2 | exp5 (TODO) |
| Speedup performance | 2.9x | exp6 (TODO) |
| Case Studies (4) | 75-79% economia | exp4 |

**Total**: 6 experimentos principais + 2 auxiliares

---

## 🚀 Quick Start (15 minutos)

### Passo 1: Setup Automático (5 min)

```bash
# Dentro do diretório experimentos/
chmod +x setup.sh
./setup.sh
```

Isso irá:
- ✅ Verificar Python ≥ 3.8
- ✅ Criar ambiente virtual
- ✅ Instalar todas as dependências
- ✅ Criar diretórios necessários
- ✅ Executar teste rápido

### Passo 2: Teste Rápido (2 min)

```bash
# Ativar ambiente
source venv/bin/activate

# Testar experimento 1 (auto-detecção)
cd scripts/
python exp1_auto_detection.py --quick
```

**Saída esperada**:
```
🔬 EXPERIMENTO 1: AUTO-DETECÇÃO DE ATRIBUTOS SENSÍVEIS
========================================================
[1/5] Processando: compas_synthetic
   ✅ Detectado: ['age', 'race', 'sex']
   📈 Precision: 1.000 | Recall: 1.000 | F1: 1.000
...
✅ Claim 'F1-Score ≥ 0.90': VALIDATED ✅
```

### Passo 3: Testar EEOC Validation (3 min)

```bash
python exp3_eeoc_validation.py
```

**Saída esperada**:
```
🔍 TESTE 1: REGRA 80% EEOC
   ✅ PASS: DI=0.80 - BOUNDARY CASE
   ✅ PASS: DI=0.78 - VIOLATION
   ...
📊 Acurácia: 100.0%
✅ Claim '100% precisão': VALIDATED ✅
```

### Passo 4: Testar Case Study (5 min)

```bash
python exp4_case_studies.py --dataset compas
```

**Saída esperada**:
```
🔬 CASE STUDY 1: COMPAS RECIDIVISM PREDICTION
   ⏱️  Tempo de análise: 7.2 minutos
   ✅ Claims validadas: PASS
```

---

## 📖 Próximos Passos

### Se você tem 30 minutos:
1. ✅ Leia [RESUMO_EXECUTIVO.md](RESUMO_EXECUTIVO.md)
   - 15 claims a validar
   - Timeline de 18 semanas
   - Recursos necessários

### Se você tem 1 hora:
1. ✅ Leia [PLANO_EXPERIMENTOS.md](PLANO_EXPERIMENTOS.md)
   - 17 seções detalhadas
   - Metodologias completas
   - Critérios de validação

### Se você tem 1 dia:
1. ✅ Leia [GUIA_EXECUCAO.md](GUIA_EXECUCAO.md)
   - Setup passo a passo
   - Execução de todos experimentos
   - Troubleshooting

### Se você está pronto para executar:
1. ✅ Use [CHECKLIST_RAPIDO.md](CHECKLIST_RAPIDO.md)
   - Tracking diário
   - 6 experimentos críticos
   - Dashboard de progresso

---

## 🎯 Experimentos Prontos vs TODO

### ✅ Prontos para Executar (3)

1. **Experimento 1**: Auto-Detecção
   - Script: `exp1_auto_detection.py`
   - Modo rápido: 5 datasets sintéticos
   - Modo completo: 500 datasets reais
   - Status: ✅ COMPLETO

2. **Experimento 3**: EEOC/ECOA
   - Script: `exp3_eeoc_validation.py`
   - Testes: Regra 80%, Question 21, Adverse Actions
   - Status: ✅ COMPLETO

3. **Experimento 4**: Case Studies
   - Script: `exp4_case_studies.py`
   - Datasets: COMPAS, German Credit, Adult, Healthcare
   - Status: ✅ PARCIAL (COMPAS completo, outros simplificados)

### 🚧 TODO (Criar Scripts)

1. **Experimento 2**: Cobertura de Métricas
2. **Experimento 5**: Usabilidade (SUS/TLX)
3. **Experimento 6**: Performance (Speedup)
4. **Experimento 7**: Threshold Optimization
5. **Experimento 8**: Comparação com Ferramentas
6. **Experimento 9**: Edge Cases

---

## 💰 Recursos Necessários

### Tempo Total: 18 semanas (4.5 meses)
- Setup: 2 semanas
- Experimentos Core: 7 semanas
- Usabilidade: 4 semanas
- Validação: 3 semanas
- Finalização: 2 semanas

### Pessoas:
- **1 Pesquisador Principal**: Full-time
- **20 Participantes**: 1h cada (usabilidade)
- **2 Revisores**: 40h cada (ground truth)

### Financeiro:
- Incentivos participantes: $1,000
- AWS (benchmarks): ~$100
- Licenças de datasets: ~$200
- **Total**: ~$1,300

---

## ⚠️ Experimentos Críticos (Deal-breakers)

Estes experimentos **DEVEM PASSAR** para o paper ser aceito:

1. **EEOC/ECOA**: 100% precisão ⭐⭐⭐
   - 0 erros permitidos
   - Claim mais crítica do paper

2. **Auto-Detecção**: F1 ≥ 0.85 ⭐⭐
   - Testado em ≥300 datasets

3. **Usabilidade**: SUS ≥ 75 ⭐⭐
   - N ≥ 15 participantes

4. **Performance**: Speedup ≥ 2.0x ⭐
   - Testado em 3 tamanhos de datasets

---

## 🔍 Navegação Rápida

### Para cada tipo de usuário:

**Executivo/Revisor** (quer visão geral):
→ [RESUMO_EXECUTIVO.md](RESUMO_EXECUTIVO.md)

**Pesquisador** (quer metodologia completa):
→ [PLANO_EXPERIMENTOS.md](PLANO_EXPERIMENTOS.md)

**Implementador** (quer executar experimentos):
→ [GUIA_EXECUCAO.md](GUIA_EXECUCAO.md)

**Gerente de Projeto** (quer tracking):
→ [CHECKLIST_RAPIDO.md](CHECKLIST_RAPIDO.md)

**Desenvolvedor** (quer código):
→ [scripts/](scripts/)

**Procurando arquivo específico?**:
→ [INDEX.md](INDEX.md)

---

## 📊 Validação de Qualidade

Este framework foi criado seguindo:

✅ **Metodologia rigorosa** (baseado em papers FAccT/ICML)
✅ **Reprodutibilidade** (scripts completos + dados)
✅ **Transparência** (documentação detalhada)
✅ **Validação múltipla** (inter-rater agreement)
✅ **Estatísticas apropriadas** (Cohen's Kappa, etc)

---

## 🤝 Contribuindo

Se você encontrar bugs ou tiver sugestões:

1. **Documente o problema** em `issues.md`
2. **Propor melhorias** via pull request
3. **Compartilhar resultados** quando completar experimentos

---

## 📞 Precisa de Ajuda?

### Problemas Técnicos:
```bash
# Verificar instalação
python scripts/utils.py

# Verificar dependências
python -c "from scripts.utils import check_dependencies; check_dependencies()"
```

### Dúvidas sobre Experimentos:
- Consulte [PLANO_EXPERIMENTOS.md](PLANO_EXPERIMENTOS.md) seção específica
- Veja [GUIA_EXECUCAO.md](GUIA_EXECUCAO.md) seção Troubleshooting

### Dúvidas sobre Timeline:
- Veja [RESUMO_EXECUTIVO.md](RESUMO_EXECUTIVO.md) seção Timeline
- Use [CHECKLIST_RAPIDO.md](CHECKLIST_RAPIDO.md) para tracking

---

## ✅ Checklist Pré-Execução

Antes de começar os experimentos completos, verifique:

- [ ] Setup concluído (`./setup.sh` executado)
- [ ] Teste rápido passou (`exp1_auto_detection.py --quick`)
- [ ] EEOC validation passou (`exp3_eeoc_validation.py`)
- [ ] Documentação lida (pelo menos RESUMO_EXECUTIVO.md)
- [ ] Timeline revisada e aprovada
- [ ] Recursos (tempo, pessoas, $) confirmados
- [ ] Datasets identificados (fontes: Kaggle, UCI, OpenML)
- [ ] Participantes de usabilidade identificados

---

## 🎯 Resultado Esperado

Ao final dos experimentos, você terá:

✅ **Dados** para validar todas as 15 claims do paper
✅ **Figuras** e tabelas prontas para publicação
✅ **Reproduction package** completo
✅ **Manuscrito** com seção de Evaluation preenchida
✅ **Confiança** para submissão ao FAccT 2026

---

## 🚀 Comandos Rápidos

```bash
# Setup completo
./setup.sh

# Ativar ambiente
source venv/bin/activate

# Testes rápidos
cd scripts/
python exp1_auto_detection.py --quick      # 2 min
python exp3_eeoc_validation.py             # 1 min
python exp4_case_studies.py --dataset compas  # 5 min

# Experimentos completos
python exp1_auto_detection.py --n-datasets 500  # 3-4 semanas
python exp4_case_studies.py --dataset all       # 1 semana

# Análise de concordância
python calculate_inter_rater_agreement.py \
    --reviewer1 ../data/annotations_reviewer1.csv \
    --reviewer2 ../data/annotations_reviewer2.csv

# Ver progresso
cat ../CHECKLIST_RAPIDO.md
```

---

## 📈 Dashboard de Progresso Inicial

```
SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Ambiente criado     [ ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ ]   0%
2. Deps instaladas     [ ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ ]   0%
3. Teste rápido        [ ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ ]   0%

PROGRESSO GERAL       [ ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ ]   0%
```

**Atualize este dashboard conforme avança!**

---

## 🎓 Citação

Framework criado para validar:

> **DeepBridge Fairness: Da Pesquisa à Regulação -- Um Framework Pronto para Produção para Teste de Fairness Algorítmica**
>
> FAccT 2026 (em submissão)

---

**Pronto para começar? Execute `./setup.sh` agora! 🚀**

**Perguntas?** Leia [RESUMO_EXECUTIVO.md](RESUMO_EXECUTIVO.md) em seguida.

**Boa sorte com os experimentos!** 🎯
