# HPM-KD: Destilação de Conhecimento Hierárquica Progressiva Multi-Professor

## Informações do Paper

**Título Completo**: HPM-KD: Hierarchical Progressive Multi-Teacher Knowledge Distillation for Efficient Model Compression

**Título em Português**: HPM-KD: Destilação de Conhecimento Hierárquica Progressiva Multi-Professor para Compressão Eficiente de Modelos

**Autores**:
- Gustavo Coelho Haase (Universidade Católica de Brasília)
- Paulo Dourado (Universidade Católica de Brasília)

**Status**: Em desenvolvimento

**Versão**: 0.1

**Data de Início**: Novembro de 2025

---

## Resumo

Este paper apresenta o HPM-KD (Hierarchical Progressive Multi-Teacher Knowledge Distillation), um framework inovador para destilação de conhecimento que combina múltiplas técnicas avançadas para alcançar compressão de modelos de 10x+ mantendo a performance. O HPM-KD introduz seis componentes principais:

1. **Adaptive Configuration Manager**: Meta-aprendizado para seleção automática de configuração
2. **Progressive Distillation Chain**: Cadeia progressiva com rastreamento de melhoria mínima
3. **Attention-Weighted Multi-Teacher**: Ensemble multi-professor com pesos de atenção aprendidos
4. **Meta-Temperature Scheduler**: Agendamento adaptativo de temperatura
5. **Parallel Processing Pipeline**: Pipeline paralelo com cache inteligente
6. **Shared Optimization Memory**: Memória compartilhada entre experimentos

---

## Conferências Alvo

- **Primary**: NeurIPS (Conference on Neural Information Processing Systems)
- **Alternative**: ICML, ICLR, AAAI

---

## Estrutura do Paper

1. **Introduction**: Motivação e contribuições
2. **Related Work**: Knowledge Distillation, Multi-teacher, Progressive approaches
3. **Experimental Setup**: Datasets, baselines, métricas
4. **Methodology (HPM-KD Framework)**: Arquitetura detalhada dos 6 componentes
5. **Results**: Compression efficiency, ablation studies, computational analysis
6. **Ablation Studies**: Análise de cada componente
7. **Discussion**: Limitações, trade-offs, quando usar HPM-KD

---

## Experimentos Planejados

### Datasets
- MNIST, Fashion-MNIST (baseline)
- CIFAR-10, CIFAR-100
- UCI ML: Adult, Credit, Wine Quality
- OpenML-CC18 benchmark suite

### Baselines
- Knowledge Distillation (Hinton et al., 2015)
- FitNets (Romero et al., 2015)
- Deep Mutual Learning (Zhang et al., 2018)
- TAKD (Mirzadeh et al., 2020)
- Self-supervised KD (Xu et al., 2020)

### Métricas
- Compression ratio
- Accuracy retention
- Training time
- Inference latency
- Memory footprint

---

## Como Compilar

### Requisitos
- LaTeX (TeXLive ou MikTeX)
- BibTeX
- Pacotes: elsarticle, amsmath, graphicx, hyperref

### Comandos

```bash
# Compilação completa (com bibliografia)
make

# Compilação rápida (sem atualizar bibliografia)
make quick

# Limpar arquivos auxiliares
make clean

# Limpar tudo incluindo PDF
make distclean

# Ver PDF gerado
make view
```

---

## Status das Seções

- [x] README criado
- [ ] Introduction (em desenvolvimento)
- [ ] Related Work (planejado)
- [ ] Experimental Setup (planejado)
- [ ] Methodology (planejado)
- [ ] Results (aguardando experimentos)
- [ ] Ablation Studies (aguardando experimentos)
- [ ] Discussion (planejado)
- [ ] References (coletando)

---

## Referências Principais Necessárias

Ver `bibliography/references_needed.txt` para lista completa.

Principais:
- Hinton et al. (2015) - Knowledge Distillation original
- Romero et al. (2015) - FitNets
- Zhang et al. (2018) - Deep Mutual Learning
- Mirzadeh et al. (2020) - TAKD
- Papers sobre meta-learning, attention mechanisms, progressive training

---

## Notas de Desenvolvimento

### TODO
- [ ] Implementar experimentos comparativos
- [ ] Gerar figuras da arquitetura
- [ ] Criar tabelas de resultados
- [ ] Adicionar ablation studies
- [ ] Revisar com orientador
- [ ] Preparar código reproduzível

### Decisões de Design
- Usar elsarticle.cls para compatibilidade com Elsevier journals
- Estrutura modular em seções separadas
- Figuras em formato PDF (vetorial)
- Tabelas usando booktabs

---

## Contato

**Autor Principal**: Gustavo Coelho Haase
**Email**: gustavohaase@ucb.edu.br
**Orientador**: Prof. Dr. Osvaldo Candido da Silva Filho

---

**Última Atualização**: Novembro de 2025
