# Plano de Limpeza - POR Directory

## ✅ PODE ELIMINAR (Duplicados em V1/V2)

### Arquivos/Diretórios Principais DUPLICADOS:
- `bibliography/`          → Já em V1/bibliography/ e V2/bibliography/
- `figures/`               → Já em V1/figures/ (5 figuras)
- `sections/`              → Já em V1/sections/ (11 seções)
- `main.tex`               → Já em V1/main.tex
- `main.pdf`               → Já em V1/main.pdf (921KB)
- `main.spl`               → Já em V1/main.spl
- `elsarticle.cls`         → Já em V1/elsarticle.cls
- `elsarticle-*.bst`       → Já em V1/ (3 arquivos)

### Arquivos Temporários de Compilação:
- `main.aux`               → Arquivo temporário LaTeX
- `main.log`               → Log de compilação
- `main.out`               → Hyperref output

### Diretórios VAZIOS:
- `experiments/`           → Vazio (4KB = só diretório)
- `supplementary/`         → Vazio (4KB = só diretório)
- `tables/`                → Vazio (4KB = só diretório)

## ⚠️ AVALIAR ANTES DE ELIMINAR:
- `build/` (972KB)         → Verificar se tem arquivos importantes

## ✅ MANTER:
- `V1/`                    → Versão completa (153 páginas)
- `V2/`                    → Versão condensada (6 páginas)
- `README.md`              → Documentação das versões
- `Makefile`               → Útil para compilação
- `PROPOSTA.md`            → Documentação do projeto
- `STATUS.md`              → Status do desenvolvimento

## Comandos de Limpeza (REVISAR ANTES DE EXECUTAR!)

```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/POR

# 1. BACKUP primeiro!
mkdir -p ../BACKUP_POR_$(date +%Y%m%d)
cp -r . ../BACKUP_POR_$(date +%Y%m%d)/

# 2. Remover duplicados
rm -rf bibliography figures sections
rm -f main.tex main.pdf main.spl
rm -f elsarticle.cls elsarticle-*.bst

# 3. Remover temporários de compilação
rm -f main.aux main.log main.out main.bbl main.blg

# 4. Remover diretórios vazios
rmdir experiments supplementary tables

# 5. Avaliar build/ antes de remover
ls -lah build/
# Se não tiver nada importante:
# rm -rf build/
```

## Resultado Final:

```
POR/
├── V1/                  # Versão completa
├── V2/                  # Versão condensada
├── README.md            # Documentação
├── Makefile             # Compilação
├── PROPOSTA.md          # Proposta do projeto
└── STATUS.md            # Status
```

## Economia de Espaço Estimada:
- Arquivos duplicados: ~1.2 MB
- Temporários: ~178 KB
- Diretórios vazios: 12 KB
- **Total: ~1.4 MB liberados**
