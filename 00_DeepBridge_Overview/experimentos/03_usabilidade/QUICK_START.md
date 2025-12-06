# Quick Start - Experimento 3: Estudo de Usabilidade

## Instalação Rápida

```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/03_usabilidade

# Instalar dependências
pip install -r requirements.txt
```

## Execução Rápida (Dados Mock)

### Opção 1: Pipeline Completo (Recomendado)

```bash
# Executa tudo: dados → métricas → estatísticas → visualizações → relatórios
python scripts/analyze_usability.py
```

**Tempo**: ~30 segundos

**Outputs**:
- Dados sintéticos de 20 participantes
- Métricas calculadas (SUS, TLX, etc.)
- Análise estatística completa
- 4 figuras PDF
- Tabela LaTeX
- Relatório textual

### Opção 2: Etapas Individuais

```bash
# 1. Gerar dados mock
python scripts/generate_mock_data.py

# 2. Calcular métricas
python scripts/calculate_metrics.py

# 3. Análise estatística
python scripts/statistical_analysis.py

# 4. Visualizações
python scripts/generate_visualizations.py
```

## Resultados Esperados

Após execução, você terá:

```
03_usabilidade/
├── results/
│   ├── 03_usability_sus_scores.csv              # Dados SUS
│   ├── 03_usability_nasa_tlx.csv                # Dados TLX
│   ├── 03_usability_task_times.csv              # Tempos
│   ├── 03_usability_errors.csv                  # Erros
│   ├── 03_usability_metrics.json                # Métricas
│   ├── 03_usability_statistical_analysis.json   # Estatísticas
│   └── 03_usability_summary_report.txt          # Relatório
├── figures/
│   ├── sus_score_distribution.pdf
│   ├── nasa_tlx_dimensions.pdf
│   ├── task_completion_times.pdf
│   └── success_rate_by_task.pdf
└── tables/
    └── usability_summary.tex
```

## Visualizar Resultados

### Relatório Textual

```bash
cat results/03_usability_summary_report.txt
```

### Figuras PDF

```bash
# Abrir figuras (Linux)
xdg-open figures/sus_score_distribution.pdf
xdg-open figures/task_completion_times.pdf

# macOS
open figures/*.pdf

# Windows
start figures/*.pdf
```

### Métricas JSON

```bash
# Ver métricas formatadas
python -m json.tool results/03_usability_metrics.json
```

## Resultados Mock Esperados

### Métricas Principais

```
SUS Score:     87.5 ± 3.2  (Excellent, Grade A, Top 10%)
NASA TLX:      28.0 ± 5.1  (Low Workload)
Success Rate:  95% (19/20 participants)
Mean Time:     12.0 ± 2.5 min
Mean Errors:   1.3 ± 0.9
```

### Status vs. Metas

```
✓ SUS Score ≥ 85       → 87.5  PASS
✓ NASA TLX ≤ 30        → 28.0  PASS
✓ Success Rate ≥ 90%   → 95.0% PASS
✓ Mean Time ≤ 15 min   → 12.0  PASS
✓ Mean Errors ≤ 2      → 1.3   PASS
```

**Todas as metas atingidas!** ✅

## Usar com Dados Reais

Para usar dados reais de estudo:

1. **Coletar dados do estudo real** (seguir protocolo em `materials/`)

2. **Salvar CSVs** no formato esperado:
   ```
   results/03_usability_sus_scores.csv
   results/03_usability_nasa_tlx.csv
   results/03_usability_task_times.csv
   results/03_usability_errors.csv
   ```

3. **Executar análise** (pular generate_mock_data):
   ```bash
   python scripts/calculate_metrics.py
   python scripts/statistical_analysis.py
   python scripts/generate_visualizations.py
   ```

## Materiais do Estudo

### Questionários

```bash
# Visualizar questionários
cat materials/SUS_questionnaire.md
cat materials/NASA_TLX_questionnaire.md
```

### Tarefas

```bash
# Visualizar descrição das tarefas
cat materials/study_tasks.md
```

## Customização

### Alterar Número de Participantes

Editar `scripts/generate_mock_data.py`:
```python
def generate_participant_demographics(n_participants=30, seed=42):  # 20 → 30
```

### Alterar Metas

Editar `config/experiment_config.yaml`:
```yaml
targets:
  sus_score: 85  # Alterar aqui
  nasa_tlx: 30
  success_rate: 90
```

## Troubleshooting

### Erro: Module not found

```bash
pip install -r requirements.txt
```

### Erro: No such file 'results/...'

```bash
# Gerar dados primeiro
python scripts/generate_mock_data.py
```

### Figuras não aparecem

```bash
# Verificar se matplotlib backend está configurado
python -c "import matplotlib; print(matplotlib.get_backend())"

# Se necessário, instalar backend
pip install pyqt5  # ou outro backend
```

## Integração com Paper

### Copiar Tabela LaTeX

```bash
# Copiar para área de transferência
cat tables/usability_summary.tex | pbcopy  # macOS
cat tables/usability_summary.tex | xclip   # Linux
```

### Incluir Figuras

```latex
% No LaTeX do paper
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{../experimentos/03_usabilidade/figures/sus_score_distribution.pdf}
  \caption{Distribuição dos SUS Scores}
  \label{fig:sus_distribution}
\end{figure}
```

## Próximo Passo

Se esta é sua primeira execução:

1. ✅ Execute o pipeline completo
2. ✅ Verifique os outputs
3. ✅ Explore as visualizações
4. ⏳ Planeje recrutamento de participantes reais
5. ⏳ Execute estudo piloto

---

**Dúvidas?** Consulte `README.md` ou `STATUS.md`
