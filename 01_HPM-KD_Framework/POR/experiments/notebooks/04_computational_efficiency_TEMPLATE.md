# Notebook 4: Computational Efficiency (RQ4) - Template

**Status:** Template criado - adaptar do notebook 01

## Estrutura

1. Carregar Configuração
2. Imports + Profiling tools
3. Experimento 4.1: Time Breakdown
   - Profile cada componente
   - Medir tempo de: busca config, treino teachers, destilação
4. Experimento 4.2: Inference Latency
   - Medir CPU latency (batch=1, 32, 128)
   - Medir GPU latency
   - Medir memória
5. Experimento 4.3: Speedup Parallelization
   - Treinar com 1, 2, 4, 8 workers
   - Calcular speedup
6. Experimento 14: Cost-Benefit
   - Plot Accuracy vs Time (Pareto frontier)
7. Visualizações
   - Stacked bar chart (time breakdown)
   - Speedup curves
   - Pareto frontier
8. Gerar Relatório

## Tempo Estimado
- Quick: 30 minutos
- Full: 1 hora

## Key Code Snippets

### Time Breakdown
```python
import time

times = {}

# Config search
t0 = time.time()
config = adaptive_config_manager.search()
times['config_search'] = time.time() - t0

# Teacher training
t0 = time.time()
teacher = train_teacher()
times['teacher_training'] = time.time() - t0

# Distillation
t0 = time.time()
student = distill_knowledge()
times['distillation'] = time.time() - t0
```

### Inference Latency
```python
# Warmup
for _ in range(10):
    model(dummy_input)

# Measure
latencies = []
for _ in range(100):
    t0 = time.time()
    model(input)
    latencies.append(time.time() - t0)

mean_latency = np.mean(latencies) * 1000  # ms
```

### Speedup Measurement
```python
for n_workers in [1, 2, 4, 8]:
    dataloader = DataLoader(dataset, num_workers=n_workers)
    
    t0 = time.time()
    train_model(dataloader)
    time_elapsed = time.time() - t0
    
    speedup = baseline_time / time_elapsed
    efficiency = speedup / n_workers
```

