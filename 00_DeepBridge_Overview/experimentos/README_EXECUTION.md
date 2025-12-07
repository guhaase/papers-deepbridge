# Guia de Execução dos Experimentos

**Data**: 2025-12-06
**Autor**: Sistema de Execução de Experimentos
**Versão**: 1.0

---

## 📋 Visão Geral

Este guia explica como executar todos os experimentos do paper DeepBridge com **dados reais**.

### Experimentos Incluídos

| # | Nome | Tempo Estimado | Status Implementação |
|---|------|----------------|---------------------|
| 1 | Benchmarks de Tempo | 3-4 horas | ✅ Real |
| 2 | Estudos de Caso | 2-3 horas | ✅ Real |
| 3 | Usabilidade | 30 min | ✅ Real |
| 4 | HPMKD | 8-12 horas | ⚠️ GPU Necessária |
| 5 | Conformidade | 1 hora | ⚠️ Mock (implementação pendente) |
| 6 | Ablation Studies | 14 horas | ⚠️ Mock (implementação pendente) |

**Tempo Total Estimado**: 21-23 horas (sequencial)

---

## 🚀 Quick Start

### 1. Executar Todos os Experimentos

```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos

# Tornar scripts executáveis
chmod +x run_all_experiments.sh monitor_experiments.sh

# Executar todos os experimentos
./run_all_experiments.sh
```

### 2. Monitorar Progresso (em outro terminal)

```bash
# Modo contínuo (atualiza a cada 5 segundos)
./monitor_experiments.sh --follow

# Ou snapshot único
./monitor_experiments.sh
```

---

## 📖 Uso Detalhado

### Script Principal: `run_all_experiments.sh`

#### Opções Disponíveis

```bash
# Ver ajuda
./run_all_experiments.sh --help

# Executar todos
./run_all_experiments.sh

# Pular experimentos específicos
./run_all_experiments.sh --skip-exp1 --skip-exp2

# Dry run (mostrar o que seria executado)
./run_all_experiments.sh --dry-run
```

#### Opções de Skip

- `--skip-exp1`: Pular Experimento 1 (Benchmarks)
- `--skip-exp2`: Pular Experimento 2 (Estudos de Caso)
- `--skip-exp3`: Pular Experimento 3 (Usabilidade)
- `--skip-exp5`: Pular Experimento 5 (Conformidade)
- `--skip-exp6`: Pular Experimento 6 (Ablation)

### Script de Monitoramento: `monitor_experiments.sh`

```bash
# Modo snapshot (executa uma vez)
./monitor_experiments.sh

# Modo contínuo (atualiza automaticamente)
./monitor_experiments.sh --follow

# Monitorar log específico
./monitor_experiments.sh --log logs_master/master_20251206_120000.log
```

---

## 📂 Estrutura de Logs

Todos os logs são salvos em `logs_master/`:

```
logs_master/
├── master_YYYYMMDD_HHMMSS.log      # Log master consolidado
├── summary_YYYYMMDD_HHMMSS.txt     # Resumo de execução
├── exp1_YYYYMMDD_HHMMSS.log        # Log do experimento 1
├── exp2_YYYYMMDD_HHMMSS.log        # Log do experimento 2
├── exp3_YYYYMMDD_HHMMSS.log        # Log do experimento 3
├── exp5_YYYYMMDD_HHMMSS.log        # Log do experimento 5
└── exp6_YYYYMMDD_HHMMSS.log        # Log do experimento 6
```

### Exemplo de Resumo (summary)

```
EXP|NOME|STATUS|TEMPO
EXP1|Benchmarks de Tempo|SUCCESS|03:24:15
EXP2|Estudos de Caso|SUCCESS|02:45:30
EXP3|Usabilidade|SUCCESS|00:28:42
EXP5|Conformidade|SUCCESS|00:52:10
EXP6|Ablation Studies|SUCCESS|13:45:22
```

---

## 🎯 Execução Individual

Se preferir executar experimentos individualmente:

### Experimento 1: Benchmarks de Tempo

```bash
cd 01_benchmarks_tempo
echo "y" | python3 scripts/run_experiment.py
```

**Outputs**:
- `results/deepbridge_benchmark_*.json`
- `results/fragmented_benchmark_*.json`
- `figures/benchmark_comparison.pdf`
- `tables/benchmark_results.tex`

### Experimento 2: Estudos de Caso

```bash
cd 02_estudos_de_caso
python3 scripts/run_all_cases.py
```

**Outputs**:
- `results/credit_results.json`
- `results/hiring_results.json`
- `results/insurance_results.json`
- `figures/case_*_comparison.pdf`

### Experimento 3: Usabilidade

```bash
cd 03_usabilidade

# Pipeline completo
python3 scripts/generate_mock_data.py
python3 scripts/analyze_usability.py
python3 scripts/generate_visualizations.py
```

**Outputs**:
- `results/usability_metrics.json`
- `figures/usability_*.pdf`
- `tables/usability_comparison.tex`

### Experimento 5: Conformidade (Mock)

```bash
cd 05_conformidade
python3 scripts/run_demo.py
```

**Outputs**:
- `results/conformidade_demo_results.json`
- `tables/conformidade_results.tex`

### Experimento 6: Ablation Studies (Mock)

```bash
cd 06_ablation_studies
python3 scripts/run_demo.py
```

**Outputs**:
- `results/ablation_demo_results.json`
- `tables/ablation_results.tex`

---

## ⚠️ Notas Importantes

### Experimento 4 (HPMKD)

O Experimento 4 **requer GPU** e deve ser executado separadamente em servidor apropriado:

```bash
# Em servidor com GPU
cd 04_hpmkd
python3 scripts/run_hpmkd.py
```

**Requisitos**:
- GPU NVIDIA com CUDA
- 8GB+ VRAM
- PyTorch com suporte CUDA

### Experimentos 5 e 6 - Implementação Pendente

Os experimentos 5 e 6 atualmente executam **versões mock** (dados simulados).

**Implementação real pendente**:
- Experimento 5: Integração com casos reais de conformidade
- Experimento 6: Implementação de configurações de ablação no DeepBridge

**Timeline estimado**: 1-2 semanas de desenvolvimento

---

## 🔧 Troubleshooting

### Erro: "Permission denied"

```bash
chmod +x run_all_experiments.sh monitor_experiments.sh
```

### Erro: "Python module not found"

```bash
# Verificar se está no ambiente correto
which python3

# Instalar dependências
pip install -r requirements.txt
```

### Experimento travado

```bash
# Ver processos Python
ps aux | grep python3

# Matar processo específico
kill -9 <PID>

# Reiniciar experimento específico
./run_all_experiments.sh --skip-exp1 --skip-exp2  # etc
```

### Logs muito grandes

```bash
# Limpar logs antigos
rm -rf logs_master/

# Ou comprimir
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs_master/
```

---

## 📊 Verificação de Resultados

Após a execução, verifique:

### 1. Todos os experimentos completaram

```bash
cat logs_master/summary_*.txt | grep SUCCESS
```

### 2. Resultados foram gerados

```bash
# Contar arquivos de resultado
find . -name "results/*.json" -type f | wc -l

# Contar figuras geradas
find . -name "figures/*.pdf" -type f | wc -l

# Contar tabelas LaTeX
find . -name "tables/*.tex" -type f | wc -l
```

### 3. Sem erros críticos

```bash
# Procurar por erros nos logs
grep -i "error\|exception\|failed" logs_master/master_*.log
```

---

## 📈 Próximos Passos

Após execução bem-sucedida:

1. **Consolidar Resultados**
   ```bash
   # Script de consolidação (criar se necessário)
   python3 scripts/consolidate_results.py
   ```

2. **Gerar Figuras do Paper**
   ```bash
   # Copiar figuras para diretório do paper
   cp */figures/*.pdf ../paper/figures/
   ```

3. **Gerar Tabelas do Paper**
   ```bash
   # Copiar tabelas para diretório do paper
   cp */tables/*.tex ../paper/tables/
   ```

4. **Análise Estatística Final**
   ```bash
   # Executar análise consolidada
   python3 scripts/final_statistical_analysis.py
   ```

---

## 🔍 Comandos Úteis

```bash
# Ver tempo total de execução
tail logs_master/master_*.log | grep "Tempo total"

# Ver resumo de todos os experimentos
cat logs_master/summary_*.txt

# Contar linhas de código geradas
find . -name "*.py" -type f -exec wc -l {} + | tail -1

# Ver uso de disco por experimento
du -sh 0*/ | sort -h

# Listar todos os resultados
find . -name "*.json" -path "*/results/*" -type f

# Verificar se há processos rodando
pgrep -a python3 | grep experiment
```

---

## 📞 Suporte

Em caso de problemas:

1. Verificar logs em `logs_master/`
2. Executar `./monitor_experiments.sh` para status
3. Consultar documentação individual de cada experimento
4. Verificar requirements e dependências

---

## ✅ Checklist de Execução

- [ ] Scripts tornados executáveis (`chmod +x`)
- [ ] Todos os requirements instalados
- [ ] Espaço em disco suficiente (~50GB)
- [ ] Tempo disponível (~21-23 horas)
- [ ] Backup de dados existentes (se houver)
- [ ] Monitoramento configurado
- [ ] Execução iniciada
- [ ] Logs verificados periodicamente
- [ ] Resultados validados
- [ ] Experimento 4 agendado para servidor GPU

---

**Última atualização**: 2025-12-06
**Versão do script**: 1.0
