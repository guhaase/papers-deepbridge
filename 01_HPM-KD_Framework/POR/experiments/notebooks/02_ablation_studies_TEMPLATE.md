# Notebook 2: Ablation Studies (RQ2) - Template

**Status:** Template criado - adaptar do notebook 01

## Estrutura

1. Carregar Configuração (copiar do 01)
2. Imports (copiar do 01)
3. Configuração do Experimento
   - 6 componentes para ablation
   - Experimentos: 5, 6, 7, 8, 9
4. Helper Functions (copiar do 01)
5. Experimento 5: Component Ablation
   - Treinar Full HPM-KD
   - Treinar sem cada componente (loop)
   - Calcular impacto (Δ accuracy)
6. Experimento 6: Component Interactions
   - Remover pares de componentes
   - Detectar sinergias
7. Experimento 7: Hyperparameter Sensitivity
   - Grid search T ∈ {2,4,6,8}, α ∈ {0.3,0.5,0.7,0.9}
8. Experimento 8: Progressive Chain Length
   - Testar 0-5 passos
   - Encontrar ponto ótimo
9. Experimento 9: Number of Teachers
   - Variar 1-8 teachers
   - Encontrar saturação
10. Visualizações
    - Heatmap de ablation
    - Gráficos de sensibilidade
11. Gerar Relatório

## Tempo Estimado
- Quick: 1 hora
- Full: 2 horas

## Key Code Snippets

### Component Ablation
```python
components = ['ProgChain', 'AdaptConf', 'MultiTeach', 'MetaTemp', 'Parallel', 'Memory']

for component in components:
    # Train with component disabled
    model = train_hpmkd(disable=[component])
    acc = evaluate(model)
    impact = full_hpmkd_acc - acc
    print(f"{component}: Δ={impact:.2f}pp")
```

### Component Interactions
```python
for c1, c2 in combinations(components, 2):
    # Train with both disabled
    model = train_hpmkd(disable=[c1, c2])
    acc = evaluate(model)
    
    # Compare to sum of individual impacts
    combined_impact = full_acc - acc
    expected_impact = impact[c1] + impact[c2]
    synergy = combined_impact - expected_impact
```

