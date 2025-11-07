# Article 1: Are Historical Redlining Effects Immutable?

## Evidence of Temporal Variation in Mortgage Credit Access 2018-2024

**Authors:** Gustavo Coelho Haase, Osvaldo Candido da Silva Filho
**Status:** In Preparation
**Target Journal:** Journal of Urban Economics (Tier 1)
**Started:** January 2025

---

## Directory Structure

```
01-Effects_Immutable/
├── main.tex                    # Main LaTeX file
├── sections/                   # Article sections
│   ├── 01-introduction.tex
│   ├── 02-literature.tex
│   ├── 03-data.tex
│   ├── 04-methodology.tex
│   ├── 05-results.tex
│   ├── 06-robustness.tex
│   ├── 07-discussion.tex
│   └── appendix.tex
├── figures/                    # Figures (PDF format)
│   ├── fig01_map_redlining.pdf
│   └── fig02_event_study.pdf
├── tables/                     # Tables (LaTeX format)
│   ├── tab01_descriptive_stats.tex
│   ├── tab02_main_result.tex
│   ├── tab03_temporal_evolution.tex
│   ├── tab04_placebo_test.tex
│   └── tab05_threshold_sensitivity.tex
├── bibliography/               # References
│   └── references.bib
├── supplementary/              # Online appendix
│   └── online_appendix.tex
├── build/                      # Compiled outputs
│   └── main.pdf
├── elsarticle.cls              # Elsevier article class
├── elsarticle-harv.bst         # Harvard bibliography style
├── Makefile                    # Compilation automation
├── README.md                   # This file
└── PLANEJAMENTO_ARTIGO.md      # Detailed planning document
```

---

## Compilation Instructions

### Option 1: Using Makefile (Recommended)
```bash
make          # Compile the article
make clean    # Remove auxiliary files
make view     # Open the compiled PDF
```

### Option 2: Manual Compilation
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Option 3: Overleaf
Upload all files to Overleaf and compile online.

---

## Key Files

### Planning and Organization
- **PLANEJAMENTO_ARTIGO.md**: Comprehensive planning document with:
  - Article structure (6,000-8,000 words target)
  - Section-by-section guidelines
  - Figure/table specifications
  - Timeline and checklist
  - Journal submission strategy

### Main Document
- **main.tex**: Master file that compiles all sections
- **sections/*.tex**: Individual section files (see structure above)

### Data Visualization
- **figures/**:
  - `fig01_map_redlining.pdf`: Spatial distribution of redlined areas
  - `fig02_event_study.pdf`: Temporal evolution (2018-2024)

### Tables
- **tables/**:
  - `tab01`: Descriptive statistics
  - `tab02`: Main result (balanced panel)
  - `tab03`: Year-by-year decomposition
  - `tab04`: Placebo test (parallel trends)
  - `tab05`: Threshold sensitivity

### References
- **bibliography/references.bib**: BibTeX database
  - Source: Dissertation bibliography (filtered for cited references only)
  - Style: Harvard (author-year)

---

## Current Status

### Completed
- [x] Directory structure created
- [x] Template files copied
- [x] Planning document written
- [x] Section files created with TODOs
- [x] Main.tex configured

### In Progress
- [ ] Writing sections (see PLANEJAMENTO_ARTIGO.md for detailed timeline)

### To Do
- [ ] Adapt content from dissertation
- [ ] Create/copy figures
- [ ] Create/copy tables
- [ ] Extract and clean bibliography
- [ ] Write cover letter
- [ ] Prepare supplementary materials

---

## Word Count Target

**Total:** 6,000-8,000 words (excluding references and tables)

**Breakdown:**
- Introduction: ~1,000 words
- Literature: ~1,500 words
- Data + Methodology: ~1,500 words
- Results: ~2,500 words
- Robustness: ~800 words
- Discussion: ~700 words

---

## Submission Checklist

Before submitting to journal:

### Content
- [ ] Abstract (≤200 words)
- [ ] Keywords (5-7)
- [ ] JEL codes
- [ ] All sections complete
- [ ] References formatted correctly
- [ ] Figures high resolution (300 dpi)
- [ ] Tables properly formatted
- [ ] Online appendix prepared

### Formatting
- [ ] Follow journal guidelines
- [ ] Line numbering (if required)
- [ ] Double spacing (if required)
- [ ] Author affiliations correct
- [ ] Acknowledgments section
- [ ] Disclosure statements

### Supporting Materials
- [ ] Cover letter
- [ ] Highlights (3-5 bullet points)
- [ ] Graphical abstract (optional)
- [ ] Replication code (GitHub/OSF)
- [ ] Data availability statement

---

## Target Journals (in order)

1. **Journal of Urban Economics** (Tier 1)
   - Impact Factor: ~3.5
   - Scope: Urban economics, housing, discrimination
   - Typical turnaround: 3-4 months

2. **Real Estate Economics** (Tier 1)
   - Impact Factor: ~3.2
   - Scope: Real estate, housing finance
   - Typical turnaround: 3-6 months

3. **Regional Science and Urban Economics** (Tier 2)
   - Impact Factor: ~2.8
   - Scope: Regional economics, urban issues
   - Typical turnaround: 2-3 months

4. **Journal of Housing Economics** (Tier 2)
   - Impact Factor: ~2.5
   - Scope: Housing markets, policy
   - Typical turnaround: 3-4 months

---

## Contacts

**Corresponding Author:**
Gustavo Coelho Haase
Email: [your-email]

**Co-author:**
Prof. Dr. Osvaldo Candido da Silva Filho
Email: [advisor-email]

---

## Notes

### From Dissertation
This article adapts content from:
- Chapter 1 (Introduction) → Section 1
- Chapter 2 (Literature, Sections 2.1-2.4) → Section 2
- Chapter 3 (Methodology, Sections 3.1-3.3) → Sections 3-4
- Chapter 4 (Results, Sections 4.1-4.3) → Section 5-6
- Chapter 5 (Limitations) → Section 7
- Chapter 6 (Conclusion) → Section 7

### Key Changes from Dissertation
- **Condensed**: 76 pages → 25-30 pages
- **Focused**: Temporal variation (main finding only)
- **Simplified**: Removed excessive technical details
- **Repositioned**: Descriptive contribution, not causal

### Complementary Articles
This is Article 1 of 5-6 planned publications:
- **Article 2:** Selective convergence (heterogeneity)
- **Article 3:** Methodological challenges (panel data)
- **Article 4:** Policy implications
- **Article 5:** Heckman correction (short note)
- **Article 6:** Literature review (optional)

---

**Last Updated:** January 2025
