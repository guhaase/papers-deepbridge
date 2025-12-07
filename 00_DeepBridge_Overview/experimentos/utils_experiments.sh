#!/bin/bash
################################################################################
# Utilitários para Gerenciamento de Experimentos
################################################################################
#
# Funções auxiliares para gerenciar experimentos
#
# Uso: ./utils_experiments.sh [COMANDO]
#
# Comandos:
#   clean           Limpar todos os resultados e logs
#   clean-logs      Limpar apenas logs
#   clean-results   Limpar apenas resultados
#   backup          Fazer backup de resultados
#   restore         Restaurar backup
#   check-space     Verificar espaço em disco
#   kill-all        Matar todos os processos de experimentos
#   list-results    Listar todos os resultados
#   validate        Validar que todos os experimentos estão prontos
#   help            Mostrar esta mensagem
#
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

BASE_DIR="/home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos"
cd "$BASE_DIR"

EXPERIMENTS=(
    "01_benchmarks_tempo"
    "02_estudos_de_caso"
    "03_usabilidade"
    "04_hpmkd"
    "05_conformidade"
    "06_ablation_studies"
)

log() {
    local level=$1
    shift
    local message="$@"

    case $level in
        INFO)  echo -e "${CYAN}[INFO]${NC} $message" ;;
        SUCCESS) echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
        WARNING) echo -e "${YELLOW}[WARNING]${NC} $message" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} $message" ;;
    esac
}

print_header() {
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Clean all results and logs
cmd_clean() {
    print_header "LIMPEZA COMPLETA"

    read -p "⚠️  ATENÇÃO: Isso irá deletar TODOS os resultados e logs. Continuar? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log INFO "Operação cancelada"
        return
    fi

    log INFO "Limpando resultados e logs de todos os experimentos..."

    for exp in "${EXPERIMENTS[@]}"; do
        if [ -d "$exp" ]; then
            log INFO "Limpando $exp..."

            # Clean results
            if [ -d "$exp/results" ]; then
                find "$exp/results" -type f ! -name ".gitkeep" -delete 2>/dev/null || true
                log INFO "  ✓ Resultados limpos"
            fi

            # Clean logs
            if [ -d "$exp/logs" ]; then
                find "$exp/logs" -type f ! -name ".gitkeep" -delete 2>/dev/null || true
                log INFO "  ✓ Logs limpos"
            fi

            # Clean figures
            if [ -d "$exp/figures" ]; then
                find "$exp/figures" -type f ! -name ".gitkeep" -delete 2>/dev/null || true
                log INFO "  ✓ Figuras limpas"
            fi

            # Clean tables
            if [ -d "$exp/tables" ]; then
                find "$exp/tables" -type f ! -name ".gitkeep" -delete 2>/dev/null || true
                log INFO "  ✓ Tabelas limpas"
            fi
        fi
    done

    # Clean master logs
    if [ -d "logs_master" ]; then
        rm -rf logs_master/*
        log INFO "✓ Logs master limpos"
    fi

    log SUCCESS "Limpeza completa!"
}

# Clean only logs
cmd_clean_logs() {
    print_header "LIMPEZA DE LOGS"

    log INFO "Limpando logs de todos os experimentos..."

    for exp in "${EXPERIMENTS[@]}"; do
        if [ -d "$exp/logs" ]; then
            find "$exp/logs" -type f ! -name ".gitkeep" -delete 2>/dev/null || true
            log INFO "✓ $exp/logs limpo"
        fi
    done

    if [ -d "logs_master" ]; then
        rm -rf logs_master/*
        log INFO "✓ logs_master limpo"
    fi

    log SUCCESS "Logs limpos!"
}

# Clean only results
cmd_clean_results() {
    print_header "LIMPEZA DE RESULTADOS"

    read -p "⚠️  Isso irá deletar todos os arquivos de resultados. Continuar? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log INFO "Operação cancelada"
        return
    fi

    log INFO "Limpando resultados de todos os experimentos..."

    for exp in "${EXPERIMENTS[@]}"; do
        if [ -d "$exp/results" ]; then
            find "$exp/results" -type f ! -name ".gitkeep" -delete 2>/dev/null || true
            log INFO "✓ $exp/results limpo"
        fi
    done

    log SUCCESS "Resultados limpos!"
}

# Backup results
cmd_backup() {
    print_header "BACKUP DE RESULTADOS"

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_dir="$BASE_DIR/backups"
    local backup_file="$backup_dir/backup_${timestamp}.tar.gz"

    mkdir -p "$backup_dir"

    log INFO "Criando backup: $backup_file"

    # Create tar with all results, figures, tables, logs
    tar -czf "$backup_file" \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        */results/ */figures/ */tables/ */logs/ logs_master/ 2>/dev/null || true

    local size=$(du -h "$backup_file" | cut -f1)
    log SUCCESS "Backup criado com sucesso!"
    log INFO "Arquivo: $backup_file"
    log INFO "Tamanho: $size"

    # List all backups
    echo ""
    log INFO "Backups disponíveis:"
    ls -lh "$backup_dir"/*.tar.gz 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
}

# Check disk space
cmd_check_space() {
    print_header "VERIFICAÇÃO DE ESPAÇO EM DISCO"

    log INFO "Espaço em disco disponível:"
    df -h "$BASE_DIR" | tail -1 | awk '{print "  Total: " $2 "\n  Usado: " $3 "\n  Disponível: " $4 "\n  Uso: " $5}'

    echo ""
    log INFO "Uso por experimento:"
    for exp in "${EXPERIMENTS[@]}"; do
        if [ -d "$exp" ]; then
            local size=$(du -sh "$exp" 2>/dev/null | cut -f1)
            printf "  %-30s %10s\n" "$exp" "$size"
        fi
    done

    echo ""
    log INFO "Uso por diretório:"
    du -sh */results 2>/dev/null | awk '{print "  " $2 ": " $1}'
    du -sh */logs 2>/dev/null | awk '{print "  " $2 ": " $1}'
    du -sh */figures 2>/dev/null | awk '{print "  " $2 ": " $1}'

    if [ -d "logs_master" ]; then
        local master_size=$(du -sh logs_master 2>/dev/null | cut -f1)
        echo "  logs_master: $master_size"
    fi
}

# Kill all experiment processes
cmd_kill_all() {
    print_header "MATAR TODOS OS PROCESSOS"

    log WARNING "Procurando processos de experimentos..."

    local pids=$(pgrep -f "python3.*run_experiment\|python3.*run_all_cases\|python3.*run_demo" || true)

    if [ -z "$pids" ]; then
        log INFO "Nenhum processo encontrado"
        return
    fi

    echo ""
    log INFO "Processos encontrados:"
    ps aux | grep -E "python3.*(run_experiment|run_all_cases|run_demo)" | grep -v grep

    echo ""
    read -p "Deseja matar estes processos? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log INFO "Operação cancelada"
        return
    fi

    echo "$pids" | while read -r pid; do
        log INFO "Matando processo $pid..."
        kill -9 "$pid" 2>/dev/null || true
    done

    log SUCCESS "Processos finalizados!"
}

# List all results
cmd_list_results() {
    print_header "LISTAGEM DE RESULTADOS"

    for exp in "${EXPERIMENTS[@]}"; do
        if [ -d "$exp/results" ]; then
            local count=$(find "$exp/results" -type f ! -name ".gitkeep" 2>/dev/null | wc -l)

            if [ $count -gt 0 ]; then
                echo -e "${CYAN}$exp${NC} ($count arquivos):"
                find "$exp/results" -type f ! -name ".gitkeep" -exec ls -lh {} \; 2>/dev/null | \
                    awk '{print "  " $9 " (" $5 ")"}'
                echo ""
            else
                echo -e "${YELLOW}$exp${NC}: Nenhum resultado"
                echo ""
            fi
        fi
    done
}

# Validate experiments are ready
cmd_validate() {
    print_header "VALIDAÇÃO DE EXPERIMENTOS"

    local all_valid=true

    for exp in "${EXPERIMENTS[@]}"; do
        echo -e "${CYAN}Validando: $exp${NC}"

        # Check directory exists
        if [ ! -d "$exp" ]; then
            log ERROR "  ✗ Diretório não existe"
            all_valid=false
            continue
        fi

        # Check scripts directory
        if [ ! -d "$exp/scripts" ]; then
            log ERROR "  ✗ Diretório scripts/ não existe"
            all_valid=false
        else
            log SUCCESS "  ✓ Diretório scripts/ OK"
        fi

        # Check for run script
        local has_run_script=false
        for script in "run_experiment.py" "run_all_cases.py" "run_demo.py" "run_hpmkd.py"; do
            if [ -f "$exp/scripts/$script" ]; then
                log SUCCESS "  ✓ Script de execução encontrado: $script"
                has_run_script=true
                break
            fi
        done

        if [ "$has_run_script" = false ]; then
            log WARNING "  ⚠ Nenhum script de execução encontrado"
        fi

        # Check required directories
        for dir in "results" "logs" "figures" "tables"; do
            if [ -d "$exp/$dir" ]; then
                log SUCCESS "  ✓ Diretório $dir/ OK"
            else
                log WARNING "  ⚠ Diretório $dir/ não existe"
            fi
        done

        # Check requirements.txt
        if [ -f "$exp/requirements.txt" ]; then
            log SUCCESS "  ✓ requirements.txt existe"
        else
            log WARNING "  ⚠ requirements.txt não encontrado"
        fi

        echo ""
    done

    if [ "$all_valid" = true ]; then
        log SUCCESS "Todos os experimentos validados!"
    else
        log WARNING "Alguns experimentos têm problemas"
    fi
}

# Show help
cmd_help() {
    grep "^#" "$0" | grep -v "^#!/" | sed 's/^# //'
}

# Main dispatcher
main() {
    local command=${1:-help}

    case $command in
        clean)          cmd_clean ;;
        clean-logs)     cmd_clean_logs ;;
        clean-results)  cmd_clean_results ;;
        backup)         cmd_backup ;;
        check-space)    cmd_check_space ;;
        kill-all)       cmd_kill_all ;;
        list-results)   cmd_list_results ;;
        validate)       cmd_validate ;;
        help|--help|-h) cmd_help ;;
        *)
            log ERROR "Comando desconhecido: $command"
            echo ""
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
