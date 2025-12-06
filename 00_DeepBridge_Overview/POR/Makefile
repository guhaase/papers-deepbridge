# Makefile para compilar o paper DeepBridge Overview
# Paper 00: DeepBridge - Framework Unificado para Validação de ML em Produção

# Variáveis
MAIN = main
LATEX = pdflatex
BIBTEX = bibtex
BUILD_DIR = build
SECTIONS_DIR = sections
BIBLIOGRAPHY_DIR = bibliography

# Flags do pdflatex
LATEX_FLAGS = -output-directory=$(BUILD_DIR) -interaction=nonstopmode -halt-on-error

# Cores para output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

.PHONY: all build quick view clean cleanall help sections check

# Target padrão
all: build

# Compila o documento completo (PDF)
build:
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Compilando Paper 00: DeepBridge      $(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@mkdir -p $(BUILD_DIR)
	@echo "$(YELLOW)[1/4] Primeira passagem (pdflatex)...$(NC)"
	@$(LATEX) $(LATEX_FLAGS) $(MAIN).tex > /dev/null || (echo "$(RED)Erro na compilação!$(NC)" && exit 1)
	@echo "$(YELLOW)[2/4] Processando bibliografia (bibtex)...$(NC)"
	@cd $(BUILD_DIR) && $(BIBTEX) $(MAIN) > /dev/null || echo "$(YELLOW)Aviso: Bibliografia não encontrada ou vazia$(NC)"
	@echo "$(YELLOW)[3/4] Segunda passagem (pdflatex)...$(NC)"
	@$(LATEX) $(LATEX_FLAGS) $(MAIN).tex > /dev/null
	@echo "$(YELLOW)[4/4] Terceira passagem (pdflatex)...$(NC)"
	@$(LATEX) $(LATEX_FLAGS) $(MAIN).tex > /dev/null
	@echo "$(GREEN)✓ Compilação concluída com sucesso!$(NC)"
	@echo "$(GREEN)PDF gerado: $(BUILD_DIR)/$(MAIN).pdf$(NC)"
	@echo ""

# Compilação rápida (apenas 1 passagem, sem bibtex)
quick:
	@echo "$(BLUE)Compilação rápida (1 passagem)...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@$(LATEX) $(LATEX_FLAGS) $(MAIN).tex > /dev/null
	@echo "$(GREEN)✓ Compilação rápida concluída!$(NC)"
	@echo "$(YELLOW)Nota: Use 'make build' para atualizar bibliografia$(NC)"

# Abre o PDF gerado
view: build
	@echo "$(BLUE)Abrindo PDF...$(NC)"
	@if [ -f "$(BUILD_DIR)/$(MAIN).pdf" ]; then \
		xdg-open $(BUILD_DIR)/$(MAIN).pdf 2>/dev/null || \
		open $(BUILD_DIR)/$(MAIN).pdf 2>/dev/null || \
		echo "$(RED)Erro: Não foi possível abrir o PDF$(NC)"; \
	else \
		echo "$(RED)Erro: PDF não encontrado. Execute 'make build' primeiro.$(NC)"; \
	fi

# Verifica estrutura de arquivos necessários
check:
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Verificando estrutura do projeto     $(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Arquivo principal:$(NC)"
	@if [ -f "$(MAIN).tex" ]; then \
		echo "$(GREEN)✓ $(MAIN).tex encontrado$(NC)"; \
	else \
		echo "$(RED)✗ $(MAIN).tex NÃO encontrado$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Diretório de seções:$(NC)"
	@if [ -d "$(SECTIONS_DIR)" ]; then \
		echo "$(GREEN)✓ $(SECTIONS_DIR)/ encontrado$(NC)"; \
		echo "  Seções existentes:"; \
		@for f in $(SECTIONS_DIR)/*.tex 2>/dev/null; do \
			if [ -f "$$f" ]; then \
				echo "  $(GREEN)✓$(NC) $$(basename $$f)"; \
			fi; \
		done; \
		@echo "  $(YELLOW)Seções faltando:$(NC)"; \
		@for i in 01 02 03 04 05 06 07 08 09 10 11; do \
			if [ ! -f "$(SECTIONS_DIR)/$${i}_*.tex" ]; then \
				echo "  $(RED)✗$(NC) $${i}_*.tex"; \
			fi; \
		done; \
	else \
		echo "$(RED)✗ $(SECTIONS_DIR)/ NÃO encontrado$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Diretório de bibliografia:$(NC)"
	@if [ -d "$(BIBLIOGRAPHY_DIR)" ]; then \
		echo "$(GREEN)✓ $(BIBLIOGRAPHY_DIR)/ encontrado$(NC)"; \
		@if [ -f "$(BIBLIOGRAPHY_DIR)/references.bib" ]; then \
			echo "  $(GREEN)✓ references.bib encontrado$(NC)"; \
		else \
			echo "  $(RED)✗ references.bib NÃO encontrado$(NC)"; \
		fi; \
	else \
		echo "$(RED)✗ $(BIBLIOGRAPHY_DIR)/ NÃO encontrado$(NC)"; \
	fi
	@echo ""

# Cria templates vazios para as seções
sections:
	@echo "$(BLUE)Criando templates de seções...$(NC)"
	@mkdir -p $(SECTIONS_DIR)
	@for i in 01 02 03 04 05 06 07 08 09 10 11; do \
		case $$i in \
			01) name="introduction" ;; \
			02) name="background" ;; \
			03) name="architecture" ;; \
			04) name="validation" ;; \
			05) name="compliance" ;; \
			06) name="hpmkd" ;; \
			07) name="reports" ;; \
			08) name="implementation" ;; \
			09) name="evaluation" ;; \
			10) name="discussion" ;; \
			11) name="conclusion" ;; \
		esac; \
		file="$(SECTIONS_DIR)/$${i}_$${name}.tex"; \
		if [ ! -f "$$file" ]; then \
			echo "% Section $$i: $${name^}" > $$file; \
			echo "% TODO: Write content for this section" >> $$file; \
			echo "" >> $$file; \
			echo "\\section{$${name^}}" >> $$file; \
			echo "\\label{sec:$$name}" >> $$file; \
			echo "" >> $$file; \
			echo "% Content here" >> $$file; \
			echo "$(GREEN)✓ Criado: $$file$(NC)"; \
		else \
			echo "$(YELLOW)- Já existe: $$file$(NC)"; \
		fi; \
	done
	@echo "$(GREEN)✓ Templates de seções criados!$(NC)"

# Limpa arquivos temporários de build
clean:
	@echo "$(YELLOW)Limpando arquivos temporários...$(NC)"
	@rm -f $(BUILD_DIR)/*.aux $(BUILD_DIR)/*.log $(BUILD_DIR)/*.out \
	       $(BUILD_DIR)/*.bbl $(BUILD_DIR)/*.blg $(BUILD_DIR)/*.toc \
	       $(BUILD_DIR)/*.lof $(BUILD_DIR)/*.lot $(BUILD_DIR)/*.fls \
	       $(BUILD_DIR)/*.fdb_latexmk $(BUILD_DIR)/*.synctex.gz
	@echo "$(GREEN)✓ Arquivos temporários removidos$(NC)"

# Limpa TODOS os arquivos de build (incluindo PDF)
cleanall: clean
	@echo "$(YELLOW)Removendo PDF gerado...$(NC)"
	@rm -f $(BUILD_DIR)/$(MAIN).pdf
	@echo "$(GREEN)✓ Todos os arquivos de build removidos$(NC)"

# Mostra ajuda
help:
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)  Makefile - Paper 00: DeepBridge      $(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Comandos disponíveis:$(NC)"
	@echo ""
	@echo "  $(GREEN)make build$(NC)     - Compila o PDF completo (4 passagens)"
	@echo "  $(GREEN)make quick$(NC)     - Compilação rápida (1 passagem, sem bibtex)"
	@echo "  $(GREEN)make view$(NC)      - Compila e abre o PDF"
	@echo "  $(GREEN)make check$(NC)     - Verifica estrutura de arquivos"
	@echo "  $(GREEN)make sections$(NC)  - Cria templates vazios para seções"
	@echo "  $(GREEN)make clean$(NC)     - Remove arquivos temporários"
	@echo "  $(GREEN)make cleanall$(NC)  - Remove TODOS os arquivos de build (incluindo PDF)"
	@echo "  $(GREEN)make help$(NC)      - Mostra esta ajuda"
	@echo ""
	@echo "$(YELLOW)Estrutura de arquivos:$(NC)"
	@echo "  main.tex           - Documento principal"
	@echo "  sections/*.tex     - Seções do paper (11 seções)"
	@echo "  bibliography/*.bib - Referências bibliográficas"
	@echo "  build/             - Arquivos de build (gerados automaticamente)"
	@echo ""
	@echo "$(YELLOW)Workflow típico:$(NC)"
	@echo "  1. $(GREEN)make sections$(NC) - Criar templates de seções"
	@echo "  2. Editar arquivos em sections/"
	@echo "  3. $(GREEN)make build$(NC)    - Compilar PDF"
	@echo "  4. $(GREEN)make view$(NC)     - Visualizar resultado"
	@echo ""
