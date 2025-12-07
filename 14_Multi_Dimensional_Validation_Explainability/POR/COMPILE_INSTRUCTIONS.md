# Instruções de Compilação

Este documento contém instruções detalhadas para compilar o Paper 14.

## Métodos de Compilação

### Método 1: Script Bash (Recomendado)

O método mais simples é usar o script `compile.sh`:

```bash
./compile.sh
```

**Vantagens:**
- Interface amigável com feedback visual
- Verificação automática de erros
- Opção de limpeza de arquivos temporários
- Mostra informações do PDF gerado (tamanho, páginas)

### Método 2: Makefile

Se você preferir usar `make`:

```bash
# Compilar paper
make

# Ver ajuda com todos comandos
make help

# Limpar arquivos temporários
make clean

# Limpar tudo (incluindo PDF)
make cleanall

# Abrir PDF após compilar
make view

# Contar número de páginas
make count
```

**Vantagens:**
- Padrão em projetos LaTeX
- Recompila apenas se arquivos foram modificados
- Comandos adicionais úteis (view, count)

### Método 3: Compilação Manual

Se preferir compilar manualmente:

```bash
# Primeira compilação
pdflatex main.tex

# Processar bibliografia
bibtex main

# Segunda compilação (incorpora referências)
pdflatex main.tex

# Terceira compilação (finaliza cross-references)
pdflatex main.tex
```

**Quando usar:** Debugging de erros específicos, controle fino sobre o processo.

## Requisitos

### Distribuição LaTeX

Você precisa de uma distribuição LaTeX instalada:

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install texlive-scheme-full
```

**macOS:**
```bash
brew install --cask mactex
```

**Windows:**
- Baixe e instale [MiKTeX](https://miktex.org/download)
- Ou [TeX Live](https://www.tug.org/texlive/windows.html)

### Pacotes LaTeX Necessários

O template ACM requer os seguintes pacotes (geralmente incluídos em `texlive-full`):

- `acmart` - Template ACM
- `babel-portuguese` - Suporte a português
- `inputenc` - Codificação UTF-8
- `fontenc` - Codificação de fontes
- `graphicx` - Gráficos
- `booktabs` - Tabelas profissionais
- `amsmath` - Matemática
- `listings` - Código fonte
- `xcolor` - Cores
- `algorithm`, `algpseudocode` - Algoritmos
- `pifont` - Símbolos (checkmarks)

### Ferramentas Auxiliares (Opcionais)

```bash
# Para pdfinfo (contar páginas)
sudo apt-get install poppler-utils

# Para visualização de PDF
sudo apt-get install evince  # ou okular, xpdf, etc.
```

## Verificação de Instalação

Teste se tudo está instalado corretamente:

```bash
# Verificar pdflatex
pdflatex --version

# Verificar bibtex
bibtex --version

# Verificar pdfinfo
pdfinfo -v
```

## Resolução de Problemas

### Erro: "Command not found: pdflatex"

**Solução:** Instale distribuição LaTeX (veja seção Requisitos acima).

### Erro: "File acmart.cls not found"

**Solução:**
```bash
# Atualizar pacotes LaTeX
sudo tlmgr update --self
sudo tlmgr install acmart
```

### Erro: "Package babel Error: Unknown option portuguese"

**Solução:**
```bash
sudo tlmgr install babel-portuges
```

### Warnings do bibtex sobre referências não citadas

**Comportamento normal:** Se você não citar todas as referências no texto, bibtex irá avisar. Você pode ignorar esses warnings.

### PDF não é gerado

1. Verifique `main.log` para erro específico:
```bash
tail -n 50 main.log
```

2. Compile em modo verbose:
```bash
pdflatex main.tex
# (sem redirecionamento de output)
```

3. Verifique se todas seções existem:
```bash
ls sections/*.tex
```

### Erro: "Undefined control sequence"

**Causa comum:** Comando LaTeX não reconhecido ou pacote faltando.

**Solução:**
1. Identifique linha do erro em `main.log`
2. Verifique se pacote necessário está importado
3. Consulte documentação do pacote

## Estrutura de Arquivos

```
POR/
├── main.tex                    # Arquivo principal
├── compile.sh                  # Script de compilação (Método 1)
├── Makefile                    # Makefile (Método 2)
├── .gitignore                  # Ignora arquivos temporários
├── README.md                   # Documentação do paper
├── COMPILE_INSTRUCTIONS.md     # Este arquivo
├── sections/                   # Seções do paper
│   ├── 01_introduction.tex
│   ├── 02_background.tex
│   ├── 03_design.tex
│   ├── 04_implementation.tex
│   ├── 05_evaluation.tex
│   ├── 06_discussion.tex
│   └── 07_conclusion.tex
└── bibliography/
    └── references.bib          # Referências bibliográficas
```

## Arquivos Temporários

Durante a compilação, LaTeX gera arquivos temporários:

- `main.aux` - Informações auxiliares
- `main.bbl` - Bibliografia processada
- `main.blg` - Log do bibtex
- `main.log` - Log completo da compilação
- `main.out` - Hyperlinks (se hyperref usado)

**Limpeza automática:**
```bash
# Com script
./compile.sh  # (pergunta ao final)

# Com make
make clean
```

## Workflow Recomendado

### Para desenvolvimento iterativo:

1. **Primeira compilação completa:**
```bash
./compile.sh
```

2. **Edições subsequentes** (apenas texto, sem novas citações):
```bash
pdflatex main.tex
```

3. **Se adicionar novas citações:**
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Para submissão final:

1. Compilação limpa completa:
```bash
make cleanall
make
```

2. Verificar número de páginas:
```bash
make count
```

3. Revisar PDF:
```bash
make view
```

## Editores LaTeX Recomendados

- **TeXstudio** - IDE completa (Windows/Mac/Linux)
- **Overleaf** - Online, colaborativo
- **VS Code** + LaTeX Workshop - Moderno e extensível
- **Vim** + vimtex - Para usuários avançados
- **Emacs** + AUCTeX - Para usuários avançados

## Integração com Git

O arquivo `.gitignore` já está configurado para:
- ✅ Ignorar arquivos temporários LaTeX
- ✅ Manter `main.pdf` no repositório
- ✅ Ignorar arquivos de sistema operacional
- ✅ Ignorar configurações de editores

## Suporte

**Problemas com LaTeX:**
- [TeX StackExchange](https://tex.stackexchange.com/)
- [Overleaf Documentation](https://www.overleaf.com/learn)

**Problemas com o paper:**
- Verifique `main.log` para erros específicos
- Consulte documentação do template ACM

## Checklist de Compilação Final

Antes de submeter o paper:

- [ ] Compilação limpa sem erros
- [ ] Todas referências citadas no texto
- [ ] Figuras e tabelas numeradas corretamente
- [ ] Seções estão na ordem correta
- [ ] Abstract tem menos de 250 palavras
- [ ] Paper tem no máximo 10 páginas (ACM format)
- [ ] Todos autores e afiliações corretos
- [ ] Email de contato atualizado
- [ ] Bibliografia completa e formatada
- [ ] PDF gerado abre corretamente

---

**Última atualização:** 2025-12-07
**Versão:** 1.0
