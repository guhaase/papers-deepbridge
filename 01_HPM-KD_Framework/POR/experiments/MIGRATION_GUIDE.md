# 🔄 Guia de Migração - DeepBridge Imports

**Data:** 2025-11-07
**Versão:** DeepBridge 0.1.54+

---

## ⚠️ Mudanças Importantes nas Importações

A estrutura de módulos do DeepBridge foi refatorada. Este guia mostra as mudanças necessárias nos notebooks.

---

## 📦 Importações Antigas (❌ NÃO USAR)

```python
# ❌ INCORRETO - Módulos não existem mais
from deepbridge.core.knowledge_distillation import HPM_KD
from deepbridge.data import DBDataset
```

---

## ✅ Importações Corretas (Versão 0.1.54+)

### 1. Dataset

```python
# ✅ CORRETO
from deepbridge.core.db_data import DBDataset

# Uso
dataset = DBDataset(X_train, y_train)
```

### 2. Knowledge Distillation

```python
# ✅ CORRETO
from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation

# Uso
kd = KnowledgeDistillation(
    teacher=teacher_model,
    student=student_model,
    temperature=3.0,
    alpha=0.7
)
```

### 3. Auto Distiller

```python
# ✅ CORRETO
from deepbridge.distillation.auto_distiller import AutoDistiller

# Uso
distiller = AutoDistiller(
    teacher=teacher_model,
    student_architecture='resnet20'
)
```

### 4. Experiment

```python
# ✅ CORRETO
from deepbridge.core.experiment import Experiment

# Uso
exp = Experiment(
    model=model,
    dataset=dataset,
    name='mnist_experiment'
)
```

### 5. Surrogate Model

```python
# ✅ CORRETO
from deepbridge.distillation.techniques.surrogate import SurrogateModel

# Uso
surrogate = SurrogateModel(
    input_dim=784,
    output_dim=10
)
```

---

## 🔍 Como Verificar Importações

### Teste Rápido

```python
# Copie e execute este código para testar todas as importações

import sys

def test_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"✅ {class_name:.<30} OK")
        return True
    except (ImportError, AttributeError) as e:
        print(f"❌ {class_name:.<30} {str(e)[:40]}")
        return False

print("🧪 Testando importações DeepBridge:\n")

# Core components
test_import('deepbridge.core.db_data', 'DBDataset')
test_import('deepbridge.core.experiment', 'Experiment')

# Distillation components
test_import('deepbridge.distillation.auto_distiller', 'AutoDistiller')
test_import('deepbridge.distillation.techniques.knowledge_distillation', 'KnowledgeDistillation')
test_import('deepbridge.distillation.techniques.surrogate', 'SurrogateModel')

# Utils
test_import('deepbridge.utils.model_registry', 'ModelType')

print("\n✅ Teste de importações concluído!")
```

---

## 📝 Checklist de Migração

Para atualizar notebooks antigos:

- [ ] Substituir `from deepbridge.data import DBDataset` → `from deepbridge.core.db_data import DBDataset`
- [ ] Substituir `from deepbridge.core.knowledge_distillation import HPM_KD` → `from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation`
- [ ] Adicionar import do AutoDistiller se necessário
- [ ] Verificar se todas as importações funcionam (executar código de teste acima)
- [ ] Testar execução do notebook completo

---

## 🔧 Troubleshooting

### Erro: "No module named 'deepbridge.data'"

**Solução:**
```python
# Mudar de:
from deepbridge.data import DBDataset

# Para:
from deepbridge.core.db_data import DBDataset
```

### Erro: "No module named 'deepbridge.core.knowledge_distillation'"

**Solução:**
```python
# Mudar de:
from deepbridge.core.knowledge_distillation import HPM_KD

# Para:
from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation
```

### Erro: "cannot import name 'HPM_KD'"

**Explicação:** A classe `HPM_KD` não existe mais. Use `KnowledgeDistillation` ou `AutoDistiller` dependendo do caso de uso.

**Solução - Opção 1 (Knowledge Distillation Manual):**
```python
from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation

kd = KnowledgeDistillation(
    teacher=teacher_model,
    student=student_model,
    temperature=3.0,
    alpha=0.7
)
```

**Solução - Opção 2 (Auto Distiller):**
```python
from deepbridge.distillation.auto_distiller import AutoDistiller

distiller = AutoDistiller(
    teacher=teacher_model,
    student_architecture='resnet20',
    optimize=True  # Usa otimização automática
)
```

---

## 📚 Documentação Adicional

- **API Docs:** https://deepbridge.readthedocs.io/
- **Exemplos:** `/examples/notebooks/`
- **Changelog:** `CHANGELOG.md`

---

## ✅ Status dos Notebooks Atualizados

- [x] `00_setup_colab_UPDATED.ipynb` - ✅ Atualizado (2025-11-07)
- [ ] `00_setup_colab.ipynb` - ⚠️ Precisa atualização
- [ ] `01_compression_efficiency.ipynb` - ⚠️ Precisa atualização
- [ ] `02_ablation_studies.ipynb` - ⚠️ Precisa atualização
- [ ] `03_generalization.ipynb` - ⚠️ Precisa atualização
- [ ] `04_computational_efficiency.ipynb` - ⚠️ Precisa atualização

---

## 🚀 Quick Fix para Notebooks Antigos

Se você tem um notebook antigo e quer rodá-lo rapidamente, adicione esta célula no início:

```python
# 🔧 QUICK FIX - Compatibility Layer
# Adicione esta célula no INÍCIO do notebook

import sys
from types import ModuleType

# Create compatibility aliases
deepbridge_data = ModuleType('deepbridge.data')
from deepbridge.core.db_data import DBDataset
deepbridge_data.DBDataset = DBDataset
sys.modules['deepbridge.data'] = deepbridge_data

deepbridge_kd = ModuleType('deepbridge.core.knowledge_distillation')
from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation
deepbridge_kd.HPM_KD = KnowledgeDistillation  # Alias
sys.modules['deepbridge.core.knowledge_distillation'] = deepbridge_kd

print("✅ Compatibility layer loaded!")
print("⚠️ RECOMENDAÇÃO: Atualize as importações para a versão nova (veja MIGRATION_GUIDE.md)")
```

**⚠️ ATENÇÃO:** Este é apenas um quick fix temporário. O ideal é atualizar as importações para as corretas.

---

**Última atualização:** 2025-11-07
**Versão DeepBridge:** 0.1.54
