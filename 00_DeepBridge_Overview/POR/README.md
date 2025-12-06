# DeepBridge Overview Paper - Versões

Este diretório contém duas versões do artigo "DeepBridge: A Unified Production-Ready Framework for Multi-Dimensional Machine Learning Validation".

## 📁 Estrutura

```
POR/
├── V1/           # Versão completa (153 páginas)
│   ├── main.tex
│   ├── sections/ (11 seções)
│   ├── figures/  (5 figuras TikZ)
│   ├── bibliography/
│   └── elsarticle.cls (formato Elsevier)
│
└── V2/           # Versão condensada (6 páginas)
    ├── main.tex
    ├── sections/ (7 seções)
    ├── figures/  (1 figura TikZ)
    ├── bibliography/
    └── acmart.cls (formato ACM)
```

## 📄 V1 - Versão Completa (153 páginas)

**Formato:** Elsevier (`elsarticle.cls`)  
**Páginas:** 153  
**Tamanho:** 921 KB  
**Status:** ✅ Compilado com sucesso

### Compilação V1

```bash
cd V1
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 📄 V2 - Versão Condensada (6 páginas)

**Formato:** ACM (`acmart.cls`)  
**Páginas:** 6  
**Tamanho:** 482 KB  
**Status:** ✅ Compilado com sucesso

### Compilação V2

```bash
cd V2
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 🎯 Comparação

| Aspecto | V1 (Elsevier) | V2 (ACM) |
|---------|---------------|----------|
| **Páginas** | 153 | 6 |
| **Formato** | elsarticle | acmart |
| **Seções** | 11 principais | 7 condensadas |
| **Figuras** | 5 (TikZ) | 1 (TikZ) |
| **Tamanho PDF** | 921 KB | 482 KB |
