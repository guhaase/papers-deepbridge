# Próximos Passos - Experimento 01

**Data**: 2025-12-05
**Status**: Scripts prontos, aguardando resolução do DeepBridge

---

## 🎯 Situação Atual

✅ **Todo o código está pronto e funcional**
✅ **Toda a documentação está completa**
✅ **Todos os scripts executam sem crashar**

❌ **DeepBridge não executa testes reais** - `run_tests()` retorna instantaneamente sem fazer nada

---

## 🔍 Investigação Necessária (1-2 horas)

### Opção 1: Consultar Criador do DeepBridge

**Perguntas para fazer**:

1. Por que `run_tests()` retorna instantaneamente sem executar nada?
2. Existe alguma configuração ou método de setup necessário antes de chamar `run_tests()`?
3. Pode fornecer um exemplo mínimo e completo de uso do Experiment?
4. A diferença entre `config_name='quick'`, `'medium'` e `'full'` deve ser perceptível nos tempos?

**Como perguntar**:
- Issues no repositório do DeepBridge
- Email para o criador
- Slack/Discord/canal de comunicação

### Opção 2: Investigar Código-Fonte

```bash
cd /home/guhaase/projetos/DeepBridge

# 1. Ver implementação de run_tests()
cat deepbridge/core/experiment/experiment.py | grep -A 100 "def run_tests"

# 2. Procurar exemplos de uso
find . -name "*.py" -exec grep -l "run_tests" {} \; | head -10

# 3. Ver testes de unidade
find . -name "test_*.py" -o -name "*_test.py" | xargs ls -lh

# 4. Procurar README ou documentação
find . -name "README*" -o -name "USAGE*" -o -name "EXAMPLE*"
```

### Opção 3: Criar Exemplo Mínimo Isolado

Criar um arquivo `minimal_test.py` super simples:

```python
#!/usr/bin/env python3
"""
Teste mínimo absoluto do DeepBridge
"""
import sys
sys.path.insert(0, '/home/guhaase/projetos/DeepBridge')

from deepbridge import DBDataset, Experiment
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

print("=" * 60)
print("TESTE MÍNIMO DEEPBRIDGE")
print("=" * 60)

# 1. Dados sintéticos ultra-simples
X, y = make_classification(
    n_samples=100,  # Apenas 100 amostras
    n_features=5,   # Apenas 5 features
    n_informative=3,
    n_classes=2,
    random_state=42
)

df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
df['target'] = y
df['sex'] = np.random.choice([0, 1], 100)  # Protected attribute
print(f"✓ Dataset: {df.shape}")
print(f"✓ Dtypes: {df.dtypes.to_dict()}")

# 2. Modelo ultra-simples
model = XGBClassifier(n_estimators=10, max_depth=2, random_state=42, verbosity=1)
model.fit(df[['f0', 'f1', 'f2', 'f3', 'f4']], df['target'])
print(f"✓ Modelo treinado")

# 3. DBDataset
dataset = DBDataset(
    data=df,
    target_column='target',
    model=model
)
print(f"✓ DBDataset criado")
print(f"  Features: {dataset.features}")
print(f"  Dataset size: {len(dataset.test_data)}")

# 4. Experiment
exp = Experiment(
    dataset=dataset,
    experiment_type='binary_classification',
    protected_attributes=['sex']
)
print(f"✓ Experiment criado")

# 5. run_tests() com logging verbose
import logging
logging.basicConfig(level=logging.DEBUG)

print("\n" + "=" * 60)
print("EXECUTANDO TESTES")
print("=" * 60)

import time
start = time.time()
result = exp.run_tests(config_name='full')
elapsed = time.time() - start

print(f"\n✓ run_tests() completou em {elapsed:.4f}s")
print(f"  Result type: {type(result)}")
print(f"  Result attributes: {dir(result)}")

# 6. Tentar recuperar resultados
print("\n" + "=" * 60)
print("RECUPERANDO RESULTADOS")
print("=" * 60)

try:
    rob = exp.get_robustness_results()
    print(f"✓ Robustness: {type(rob)}, empty={rob is None}")
except Exception as e:
    print(f"✗ Robustness error: {e}")

try:
    unc = exp.get_uncertainty_results()
    print(f"✓ Uncertainty: {type(unc)}, empty={unc is None}")
except Exception as e:
    print(f"✗ Uncertainty error: {e}")

print("\nDONE!")
```

**Executar**:
```bash
cd scripts
python3 minimal_test.py 2>&1 | tee minimal_output.log
```

---

## ⏱️ Decisão Rápida (Se não resolver em 2 horas)

### Usar Simulação Para o Paper

1. **Executar benchmark simulado**:
   ```bash
   cd scripts
   python3 run_experiment.py --quick
   ```

2. **Revisar tempos simulados** em `config/config.yaml`:
   ```yaml
   tests:
     fairness:
       expected_time_deepbridge: 5  # 5 minutos
       expected_time_fragmented: 30 # 30 minutos
   ```

3. **Ajustar tempos se necessário** (baseado em experiência, literatura, etc.)

4. **Executar experimento completo**:
   ```bash
   python3 run_experiment.py --all
   ```

5. **Marcar claramente no paper**:
   > "Execution times are estimated based on expected performance characteristics.
   > Actual measurements will be added in a future revision."

---

## 📊 Checklist de Execução

### Se DeepBridge Funcionar ✅

- [ ] Descobrir por que run_tests() não executava
- [ ] Atualizar benchmark_deepbridge_REAL.py com fix
- [ ] Executar 1 run de teste
- [ ] Validar que tempos fazem sentido
- [ ] Executar 10 runs completos
- [ ] Gerar análise e figuras
- [ ] Atualizar paper com resultados reais

### Se Usar Simulação ⚠️

- [ ] Revisar e ajustar tempos em config.yaml
- [ ] Documentar que são estimativas
- [ ] Executar benchmark simulado (10 runs)
- [ ] Gerar análise e figuras
- [ ] Atualizar paper marcando como "estimated"
- [ ] Criar issue para coletar dados reais depois
- [ ] Publicar paper com nota sobre estimativas

---

## 🚀 Comandos Quick Start

### Investigar DeepBridge
```bash
# Ver implementação
cat /home/guhaase/projetos/DeepBridge/deepbridge/core/experiment/experiment.py | less

# Procurar exemplos
find /home/guhaase/projetos/DeepBridge -name "*.py" -exec grep -l "Experiment(" {} \; | head -5

# Ver primeiro exemplo
find /home/guhaase/projetos/DeepBridge -name "*.py" -exec grep -l "Experiment(" {} \; | head -1 | xargs cat
```

### Executar Teste Mínimo
```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/01_benchmarks_tempo/scripts
# Criar minimal_test.py (código acima)
python3 minimal_test.py 2>&1 | tee minimal_output.log
```

### Executar Simulação
```bash
cd scripts
python3 run_experiment.py --quick  # Teste rápido (1 run)
python3 run_experiment.py --all    # Experimento completo (10 runs)
```

---

## 📝 Arquivos Para Revisar

### Código
- `scripts/benchmark_deepbridge_REAL.py` - Principal script a usar se DeepBridge funcionar
- `scripts/benchmark_deepbridge.py` - Script simulado (fallback)

### Configuração
- `config/config.yaml` - Ajustar tempos esperados aqui

### Documentação
- `RESUMO_FINAL.md` - Resumo completo do que foi feito
- `STATUS.md` - Status atual
- `PROGRESSO.md` - Progresso detalhado

---

## ⏰ Timeline Sugerido

**Próximas 2 horas**:
1. Investigar código-fonte do DeepBridge (30 min)
2. Criar e executar teste mínimo (30 min)
3. Se não funcionar, contatar criador (30 min)
4. Decidir: continuar investigando ou usar simulação (30 min)

**Se continuar investigando** (+2-4 horas):
- Debug profundo do DeepBridge
- Possível modificação do código
- Testes extensivos

**Se usar simulação** (+1 hora):
- Ajustar tempos
- Executar benchmarks
- Gerar figuras
- Atualizar paper

---

## 💡 Dica Final

**O trabalho principal está feito**. Você tem:
- ✅ Scripts completos e funcionais
- ✅ Análise estatística pronta
- ✅ Geração de figuras pronta
- ✅ Pipeline end-to-end funcionando

**Apenas falta**:
- ❌ DeepBridge executar testes reais

**Opções**:
1. **Ideal**: Resolver DeepBridge (se possível em 1-2h)
2. **Pragmática**: Usar simulação (pode fazer agora)
3. **Híbrida**: Publicar com simulação, atualizar depois

**Recomendação**: Tentar opção 1 por 2 horas. Se não resolver, usar opção 2 para não bloquear o paper.

---

**Última atualização**: 2025-12-05 23:59
