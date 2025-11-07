# Notebook 3: Generalization (RQ3) - Template

**Status:** Template criado - adaptar do notebook 01

## Estrutura

1. Carregar Configuração
2. Imports + Functions
3. Experimento 10: Class Imbalance
   - Criar CIFAR-10 desbalanceado (10:1, 50:1, 100:1)
   - Treinar HPM-KD vs TAKD
   - Medir degradação
4. Experimento 11: Label Noise
   - Adicionar ruído (10%, 20%, 30%)
   - Treinar HPM-KD vs TAKD
   - Medir robustez
5. Experimento 13: Representation Visualization
   - Extrair features penúltima camada
   - t-SNE plot
   - Calcular Silhouette Score
6. Visualizações
   - Curvas de degradação
   - t-SNE plots
   - Boxplots cross-domain
7. Gerar Relatório

## Tempo Estimado
- Quick: 1.5 horas
- Full: 3 horas

## Key Code Snippets

### Class Imbalance
```python
def create_imbalanced_cifar10(ratio=10):
    # Subsample minority classes
    for class_idx in range(5, 10):  # Last 5 classes
        indices = np.where(train_labels == class_idx)[0]
        keep = len(indices) // ratio
        subsample = np.random.choice(indices, keep, replace=False)
    return imbalanced_dataset
```

### Label Noise
```python
def add_label_noise(labels, noise_rate=0.1):
    n_flip = int(len(labels) * noise_rate)
    flip_indices = np.random.choice(len(labels), n_flip, replace=False)
    labels[flip_indices] = np.random.randint(0, n_classes, n_flip)
    return labels
```

### t-SNE Visualization
```python
from sklearn.manifold import TSNE

# Extract features
features = extract_features(model, dataloader, device)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# Plot
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
```

