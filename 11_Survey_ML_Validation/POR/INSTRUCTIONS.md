# Instruções de Compilação e Uso

## Paper 11: Survey sobre Validação de Modelos ML

Este documento contém instruções detalhadas para compilar e trabalhar com o paper.

---

## 📋 Pré-requisitos

### Opção 1: Instalação Local (Ubuntu/Debian)

```bash
# Pacotes básicos LaTeX
sudo apt-get update
sudo apt-get install texlive-latex-base texlive-latex-extra

# Suporte a português
sudo apt-get install texlive-lang-portuguese

# Ferramentas adicionais
sudo apt-get install texlive-science texlive-publishers

# Para análise de PDFs
sudo apt-get install poppler-utils

# Visualizadores de PDF (opcional)
sudo apt-get install evince  # ou okular
```

### Opção 2: Instalação Completa (recomendado)

```bash
# Instalação completa do TeXLive (pode demorar)
sudo apt-get install texlive-full
```

### Opção 3: Docker

```bash
# Usar container Docker com LaTeX
docker pull texlive/texlive:latest

# Compilar usando Docker
docker run --rm -v $(pwd):/workdir texlive/texlive:latest \
  bash -c "cd /workdir && ./compile.sh"
```

---

## 🔨 Compilação

### Método 1: Script Bash (Recomendado)

```bash
# Tornar script executável (apenas primeira vez)
chmod +x compile.sh

# Compilar
./compile.sh
```

O script irá:
1. Limpar arquivos auxiliares antigos
2. Executar pdflatex (1ª vez)
3. Executar bibtex (processar referências)
4. Executar pdflatex (2ª vez - resolver referências)
5. Executar pdflatex (3ª vez - garantir consistência)
6. Verificar resultado e reportar número de páginas

### Método 2: Makefile

```bash
# Compilação completa
make

# Compilação rápida (sem processar referências)
make quick

# Limpar arquivos auxiliares
make clean

# Visualizar PDF (abre automaticamente)
make view

# Ver número de páginas
make pages

# Ajuda
make help
```

### Método 3: Manual

```bash
# Compilação passo a passo
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Verificar resultado
ls -lh main.pdf
pdfinfo main.pdf | grep Pages
```

---

## 📄 Estrutura de Arquivos

```
POR/
├── main.tex              # Arquivo principal do paper
├── references.bib        # Referências bibliográficas (BibTeX)
├── README.md            # Documentação do projeto
├── INSTRUCTIONS.md      # Este arquivo
├── compile.sh           # Script de compilação
├── Makefile             # Makefile alternativo
├── .gitignore           # Arquivos a ignorar no Git
└── main.pdf             # PDF gerado (após compilação)
```

---

## ✏️ Editando o Paper

### Editores Recomendados

1. **Overleaf** (online, gratuito)
   - Upload dos arquivos .tex e .bib
   - Compilação automática
   - Colaboração em tempo real

2. **TeXstudio** (desktop, gratuito)
   ```bash
   sudo apt-get install texstudio
   ```

3. **VS Code** (com extensão LaTeX Workshop)
   ```bash
   # Instalar extensão
   code --install-extension James-Yu.latex-workshop
   ```

4. **Vim/Emacs** (para usuários avançados)

### Principais Seções a Editar

- **main.tex**: Conteúdo principal do paper
  - Linha 32-42: Autores e afiliações
  - Linha 44-52: Abstract
  - Linha 54: Keywords
  - Linha 56+: Seções do paper

- **references.bib**: Adicionar/modificar referências
  - Formato BibTeX padrão
  - Exemplo:
    ```bibtex
    @article{autor2024,
      title={Título do Artigo},
      author={Sobrenome, Nome},
      journal={Nome da Revista},
      year={2024}
    }
    ```

---

## 📊 Verificações de Qualidade

### Número de Páginas

```bash
# Verificar quantas páginas tem o PDF
pdfinfo main.pdf | grep Pages

# Limite: 10 páginas
```

**IMPORTANTE**: O paper deve ter **no máximo 10 páginas**. Se exceder:
1. Reduzir detalhes em seções menos importantes
2. Compactar tabelas e figuras
3. Mover conteúdo para apêndice (se permitido)
4. Usar formatação mais compacta

### Contagem de Palavras

```bash
# Contagem aproximada (requer detex)
detex main.tex | wc -w

# Instalar detex se necessário
sudo apt-get install texlive-extra-utils
```

### Verificação de Erros

```bash
# Compilar e mostrar apenas erros
pdflatex main.tex | grep -i error

# Verificar warnings
pdflatex main.tex | grep -i warning
```

---

## 🎯 Checklist Antes de Submissão

- [ ] Paper compila sem erros
- [ ] Número de páginas ≤ 10
- [ ] Todas as referências estão citadas no texto
- [ ] Todas as citações têm entrada no .bib
- [ ] Figuras e tabelas têm legendas claras
- [ ] Abstract < 300 palavras
- [ ] Keywords definidas (5-7 palavras)
- [ ] Autores e afiliações corretos
- [ ] Formatação segue template da conferência
- [ ] Revisão ortográfica completa
- [ ] PDF visualiza corretamente

---

## 🐛 Solução de Problemas

### Erro: "pdflatex: command not found"

```bash
# Instalar LaTeX
sudo apt-get install texlive-latex-base
```

### Erro: "Package babel Error: Unknown option 'portuguese'"

```bash
# Instalar suporte a português
sudo apt-get install texlive-lang-portuguese
```

### Erro: "! LaTeX Error: File 'IEEEtran.cls' not found"

```bash
# Instalar classes IEEE
sudo apt-get install texlive-publishers
```

### Referências não aparecem

1. Certifique-se de que as citações estão no formato `\cite{chave}`
2. Execute bibtex: `bibtex main`
3. Compile novamente com pdflatex (2 vezes)

### PDF não atualiza

```bash
# Limpar arquivos auxiliares e recompilar
make distclean
make
```

---

## 📚 Recursos Adicionais

### LaTeX
- [Overleaf Documentation](https://www.overleaf.com/learn)
- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX)
- [IEEE Author Center](https://ieeeauthorcenter.ieee.org/)

### BibTeX
- [BibTeX Guide](http://www.bibtex.org/Using/)
- [Google Scholar](https://scholar.google.com/) - Exportar citações em BibTeX
- [dblp](https://dblp.org/) - Referências em ciência da computação

### Templates
- [IEEE Templates](https://www.ieee.org/conferences/publishing/templates.html)
- [ACM Templates](https://www.acm.org/publications/proceedings-template)

---

## 📧 Suporte

Para questões sobre:
- **Conteúdo do paper**: [A definir]
- **Problemas técnicos de compilação**: Abrir issue no GitHub
- **Sugestões de melhoria**: Pull requests são bem-vindos

---

**Última Atualização**: Dezembro 2025
