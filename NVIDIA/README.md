# Proposta para NVIDIA Academic Grant Program

Esta pasta contém a proposta de pesquisa para o **NVIDIA Academic Grant Program**, solicitando suporte de hardware GPU para completar os experimentos do artigo **HPM-KD: Hierarchical Progressive Multi-Teacher Knowledge Distillation**.

## 📋 Conteúdo da Pasta

### Documentos Principais

1. **`nvidia_research_proposal.tex`** - Proposta completa em LaTeX (fonte)
2. **`nvidia_research_proposal.pdf`** - Proposta compilada (24 páginas, pronta para submissão)
3. **`compile_proposal.sh`** - Script para compilar o documento LaTeX

### Documentos de Referência

4. **`NEW Academic Grant Program Terms and Conditions (Nov2023).docx`** - Edital oficial da NVIDIA
5. **`Gustavo - Resume.pdf`** - Currículo do Pesquisador Principal

## 🎯 Objetivo da Proposta

Solicitar suporte de hardware GPU da NVIDIA para completar experimentos computacionalmente intensivos necessários para validar o framework HPM-KD, incluindo:

- **Datasets:** CIFAR-10, CIFAR-100, ImageNet subsets
- **Arquiteturas:** ResNets, VGG, Vision Transformers
- **Experimentos:** ~2,884 GPU-horas estimadas
- **Prazo:** 12 meses de pesquisa

## 🔧 Como Compilar o Documento

### Opção 1: Usando o Script (Recomendado)

```bash
cd /home/guhaase/projetos/DeepBridge/papers/NVIDIA
./compile_proposal.sh
```

### Opção 2: Compilação Manual

```bash
cd /home/guhaase/projetos/DeepBridge/papers/NVIDIA
pdflatex nvidia_research_proposal.tex
pdflatex nvidia_research_proposal.tex  # Segunda passagem para TOC
```

### Requisitos

- LaTeX (texlive-full ou similar)
- Pacotes: geometry, hyperref, amsmath, booktabs, xcolor, enumitem, titlesec, fancyhdr, setspace

## 📊 Estrutura da Proposta

A proposta de 24 páginas está organizada nas seguintes seções:

1. **Executive Summary** - Resumo do projeto e necessidades
2. **Research Background and Motivation** - Contexto e lacunas científicas
3. **Research Objectives** - Objetivos primários e secundários
4. **Methodology and Technical Approach** - Detalhes do framework HPM-KD
5. **Computational Requirements** - Justificativa detalhada para GPUs
6. **Expected Outcomes and Impact** - Contribuições científicas e práticas
7. **Project Timeline** - Cronograma de 12 meses
8. **Broader Impact and Sustainability** - Compromisso com ciência aberta
9. **Requested Support** - Especificações de hardware solicitado
10. **Institutional Support** - UCB e Banco do Brasil
11. **Risk Assessment** - Análise e mitigação de riscos
12. **Conclusion** - Resumo e compromissos
13. **Appendices** - Resultados preliminares, biografia estendida

## 🖥️ Hardware Solicitado

### Opção Preferencial (Primary Request)

- **2× NVIDIA A100 (40GB ou 80GB)**
  - VRAM suficiente para Vision Transformers
  - Tensor Cores para mixed-precision training
  - NVLink para comunicação multi-GPU eficiente

### Opções Alternativas

- **4× NVIDIA RTX 4090 (24GB)** - Custo-benefício excelente
- **2× NVIDIA RTX 4090 + 2× RTX 4080** - Configuração híbrida
- **2× NVIDIA RTX 4080 (16GB)** - Opção mínima viável

## 📈 Impacto Esperado

### Contribuições Científicas

- Framework inovador com 6 componentes integrados
- Validação empírica em múltiplos datasets e arquiteturas
- Biblioteca open-source DeepBridge
- Publicações em conferências top-tier (NeurIPS, ICML, ICLR)

### Benefícios Práticos

- **Compressão:** 10-15× menor tamanho de modelo
- **Acurácia:** 95-98% de retenção vs. teacher
- **Desempenho:** 3-7% superior aos baselines SOTA
- **Aplicabilidade:** Validado em sistemas de produção (Banco do Brasil)

### Impacto na Comunidade

- Código open-source (licença MIT)
- Documentação completa e tutoriais
- Redução de custos computacionais
- Sustentabilidade (menor consumo energético)
- Democratização de ML avançado

## 👤 Pesquisador Principal

**Gustavo Coelho Haase**

- **Posição:** Senior Risk Analyst, Banco do Brasil
- **Afiliação Acadêmica:** Universidade Católica de Brasília (M.Sc. Economics)
- **Experiência:** 13+ anos em validação de modelos, data science, ML
- **Especialização:** Validação de modelos, detecção de viés, fraud detection
- **Contato:** gustavohaase@ucb.edu.br | +55 61 98288 8797
- **LinkedIn:** [linkedin.com/in/gushaase](https://www.linkedin.com/in/gushaase)

## 📝 Reconhecimentos (conforme edital NVIDIA)

Conforme os termos e condições do programa, todos os materiais publicados incluirão o seguinte reconhecimento:

> *"This research and curriculum was supported by grants from NVIDIA and utilized NVIDIA [modelo GPU] for training and validating the HPM-KD framework."*

Este reconhecimento aparecerá em:
- Papers e publicações científicas
- Apresentações em conferências
- Documentação do GitHub
- Posts em blogs e mídia

## 🔗 Links Relevantes

- **Artigo HPM-KD:** `/home/guhaase/projetos/DeepBridge/papers/01_HPM-KD_Framework/`
- **Biblioteca DeepBridge:** `https://github.com/DeepBridge-Validation/DeepBridge`
- **Programa NVIDIA:** [NVIDIA Academic Grant Program](https://www.nvidia.com/en-us/research/academic-partnerships/)

## 📅 Cronograma de Submissão

### Próximos Passos

1. **Revisão Final** - Verificar todos os detalhes da proposta
2. **Coleta de Documentos Suporte** - Cartas de recomendação (UCB, Banco do Brasil)
3. **Submissão Online** - Via portal da NVIDIA
4. **Acompanhamento** - Responder notificações em até 14 dias

### Timeline Estimado

- **Submissão:** Novembro 2025
- **Avaliação NVIDIA:** 1-3 meses
- **Notificação:** Janeiro-Março 2026
- **Início do Projeto:** Após aprovação

## ✅ Checklist de Submissão

- [x] Proposta completa em PDF (24 páginas)
- [x] CV do Pesquisador Principal
- [ ] Cartas de apoio institucional (UCB)
- [ ] Carta de apoio da indústria (Banco do Brasil)
- [ ] Comprovante de vínculo acadêmico
- [ ] Formulário de application preenchido
- [ ] Aprovação do departamento/universidade

## 📧 Contato para Dúvidas

Para questões sobre a proposta ou programa:

- **Email NVIDIA:** NVIDIAacademicgrants@nvidia.com
- **Pesquisador Principal:** gustavohaase@ucb.edu.br
- **Co-Investigador:** paulo.dourado@ucb.edu.br (UCB)

## 📚 Referências Principais

1. Hinton et al. (2015) - Knowledge Distillation original
2. Romero et al. (2014) - FitNets
3. Zhang et al. (2018) - Deep Mutual Learning
4. Mirzadeh et al. (2020) - Teacher Assistant KD
5. Chen et al. (2021) - Knowledge Review

## 💡 Notas Importantes

### Uso dos Equipamentos

De acordo com o edital, os equipamentos recebidos:

1. **Não podem ser vendidos, transferidos ou cedidos** por 3 anos
2. Devem ser usados **exclusivamente para o projeto aprovado**
3. Requerem **relatórios periódicos de progresso** à NVIDIA
4. Impostos e taxas de importação são de **responsabilidade do recipiente**

### Compromissos

1. **Progresso regular:** Relatórios trimestrais para NVIDIA
2. **Publicações:** Enviar cópias de todos os papers
3. **Open-source:** Código disponibilizado publicamente
4. **Reconhecimento:** NVIDIA citado em todas as publicações

---

**Data de Criação:** 20 de Novembro de 2025
**Última Atualização:** 20 de Novembro de 2025
**Status:** Pronto para Submissão
**Versão:** 1.0
