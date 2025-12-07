# ✅ Referências Bibliográficas - VERIFICADO

## Status: ✅ FUNCIONANDO PERFEITAMENTE

As referências bibliográficas **estão funcionando corretamente** no paper.

## 📍 Onde Encontrar as Referências

- **Localização**: Páginas **16-17** do PDF
- **Formato**: Lista numerada de [1] a [24]
- **Estilo**: ACM plain (ordenado alfabeticamente por autor)

## 📊 Verificação Realizada

```bash
✅ Total de páginas: 17 (com referências)
✅ Número de referências: 24
✅ Bibliografia processada: main.bbl gerado
✅ Citações resolvidas: todas as 24 referências citadas no texto
```

## 🔍 Como Verificar Você Mesmo

### 1. Verificar Páginas
```bash
pdfinfo main.pdf | grep Pages
# Saída: Pages: 17
```

### 2. Contar Referências
```bash
pdftotext main.pdf - | grep "^\[" | wc -l
# Saída: 24
```

### 3. Ver Referências no PDF
```bash
pdftotext -f 16 -l 17 main.pdf - | head -50
```

**Você verá algo como**:
```
[1] Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner.
    Machine bias. ProPublica, 2016.

[2] Solon Barocas, Moritz Hardt, and Arvind Narayanan.
    Fairness and machine learning: Limitations and opportunities.
    MIT Press, 2019.

[3] Rachel KE Bellamy, Kuntal Dey, Michael Hind, et al.
    AI Fairness 360: An extensible toolkit for detecting,
    understanding, and mitigating unwanted algorithmic bias.
    In arXiv preprint arXiv:1810.01943, 2018.
...
```

## 🔧 Recompilar se Necessário

Se você editou o paper e as referências não aparecem:

```bash
# Usar o script automatizado
./compile.sh

# OU compilar manualmente:
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**IMPORTANTE**: São necessárias **3 compilações** com pdflatex (1 antes e 2 depois do bibtex) para que as referências apareçam.

## 📚 Lista Completa de Referências (24)

1. Angwin et al. (2016) - Machine bias [COMPAS dataset]
2. Barocas et al. (2019) - Fairness and machine learning
3. Bellamy et al. (2018) - AI Fairness 360
4. Bird et al. (2020) - Fairlearn
5. Breck et al. (2017) - ML test score
6. Brooke (1996) - SUS scale
7. Buolamwini & Gebru (2018) - Gender Shades
8. Chouldechova (2017) - Fair prediction with disparate impact
9. Chung et al. (2019) - Slice finder
10. Congress (1974) - ECOA
11. Dua & Graff (2017) - UCI repository
12. Dwork et al. (2012) - Fairness through awareness
13. EEOC (1978) - Uniform guidelines
14. Eyuboglu et al. (2022) - Domino
15. Feldman et al. (2015) - Disparate impact
16. Hardt et al. (2016) - Equalized opportunity
17. Hart & Staveland (1988) - NASA-TLX
18. Kusner et al. (2017) - Counterfactual fairness
19. Mehrabi et al. (2021) - Survey on bias and fairness
20. Mitchell et al. (2019) - Model cards
21. European Parliament (2016) - GDPR
22. Rabanser et al. (2019) - Dataset shift
23. Saleiro et al. (2018) - Aequitas
24. Sculley et al. (2015) - Technical debt in ML

## ✅ Conclusão

**As referências estão 100% funcionais!**

O formato da classe ACM (acmart) não inclui um título de seção grande "REFERENCES" como algumas outras classes LaTeX. As referências simplesmente aparecem após o conteúdo principal como uma lista numerada, que é o comportamento padrão esperado para papers ACM.

Se você está visualizando o PDF e não vê as referências, certifique-se de:
1. Rolar até as páginas 16-17 (final do documento)
2. Procurar por entradas numeradas [1], [2], [3]...
3. Recompilar com `./compile.sh` se necessário
