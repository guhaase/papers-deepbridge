#!/bin/bash
################################################################################
# Monitor de Experimentos - Acompanhamento em Tempo Real
################################################################################
#
# Monitora a execução dos experimentos mostrando progresso e logs
#
# Uso: ./monitor_experiments.sh [OPTIONS]
#
# Opções:
#   --follow        Modo contínuo (atualiza a cada 5 segundos)
#   --log FILE      Monitorar um arquivo de log específico
#   --help          Mostrar esta mensagem
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
LOG_DIR="$BASE_DIR/logs_master"

FOLLOW_MODE=false
SPECIFIC_LOG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --follow) FOLLOW_MODE=true; shift ;;
        --log) SPECIFIC_LOG="$2"; shift 2 ;;
        --help)
            grep "^#" "$0" | grep -v "^#!/" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Opção desconhecida: $1${NC}"
            exit 1
            ;;
    esac
done

print_header() {
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════════════${NC}"
}

print_separator() {
    echo -e "${BLUE}───────────────────────────────────────────────────────────────────────────${NC}"
}

show_status() {
    clear
    print_header "MONITOR DE EXPERIMENTOS - $(date '+%Y-%m-%d %H:%M:%S')"

    echo ""
    echo -e "${CYAN}Status dos Experimentos:${NC}"
    echo ""

    # Check each experiment directory for running processes or completed results
    check_experiment "1" "Benchmarks de Tempo" "01_benchmarks_tempo"
    check_experiment "2" "Estudos de Caso" "02_estudos_de_caso"
    check_experiment "3" "Usabilidade" "03_usabilidade"
    check_experiment "5" "Conformidade" "05_conformidade"
    check_experiment "6" "Ablation Studies" "06_ablation_studies"

    echo ""
    print_separator

    # Show latest master log if exists
    if [ -d "$LOG_DIR" ] && [ -n "$(ls -A $LOG_DIR 2>/dev/null)" ]; then
        echo ""
        echo -e "${CYAN}Últimos Logs Master:${NC}"
        echo ""

        local latest_log=$(ls -t "$LOG_DIR"/master_*.log 2>/dev/null | head -1)
        if [ -n "$latest_log" ]; then
            echo -e "${YELLOW}Log: $latest_log${NC}"
            echo ""
            tail -n 15 "$latest_log" 2>/dev/null | while IFS= read -r line; do
                # Colorize by log level
                if [[ $line =~ \[SUCCESS\] ]]; then
                    echo -e "${GREEN}$line${NC}"
                elif [[ $line =~ \[ERROR\] ]]; then
                    echo -e "${RED}$line${NC}"
                elif [[ $line =~ \[WARNING\] ]]; then
                    echo -e "${YELLOW}$line${NC}"
                elif [[ $line =~ \[INFO\] ]]; then
                    echo -e "${CYAN}$line${NC}"
                else
                    echo "$line"
                fi
            done
        fi
    fi

    echo ""
    print_separator

    # Show disk usage
    echo ""
    echo -e "${CYAN}Uso de Disco:${NC}"
    echo ""
    du -sh "$BASE_DIR"/*/ 2>/dev/null | sort -hr | head -10

    echo ""
    print_separator

    # Show running Python processes
    echo ""
    echo -e "${CYAN}Processos Python Ativos:${NC}"
    echo ""
    ps aux | grep -E "python3.*run_|python3.*experiment" | grep -v grep | head -5 || echo "  Nenhum processo ativo"

    echo ""
    print_separator

    if [ "$FOLLOW_MODE" = true ]; then
        echo ""
        echo -e "${YELLOW}Atualizando em 5 segundos... (Ctrl+C para sair)${NC}"
    fi
}

check_experiment() {
    local exp_num=$1
    local exp_name=$2
    local exp_dir=$3

    local full_path="$BASE_DIR/$exp_dir"

    # Check if running
    local is_running=false
    if pgrep -f "python3.*$exp_dir" > /dev/null; then
        is_running=true
    fi

    # Check results
    local has_results=false
    if [ -d "$full_path/results" ] && [ -n "$(ls -A $full_path/results 2>/dev/null)" ]; then
        has_results=true
    fi

    # Check logs
    local latest_log=""
    if [ -d "$full_path/logs" ]; then
        latest_log=$(ls -t "$full_path/logs"/*.log 2>/dev/null | head -1)
    fi

    # Print status
    printf "  %-3s %-35s " "[$exp_num]" "$exp_name"

    if [ "$is_running" = true ]; then
        echo -e "${YELLOW}[EXECUTANDO]${NC}"

        # Show progress if available
        if [ -n "$latest_log" ]; then
            local last_line=$(tail -1 "$latest_log" 2>/dev/null)
            if [ -n "$last_line" ]; then
                echo "       └─ $(echo $last_line | cut -c1-60)..."
            fi
        fi
    elif [ "$has_results" = true ]; then
        echo -e "${GREEN}[COMPLETO]${NC}"

        # Show result count
        local result_count=$(ls "$full_path/results" 2>/dev/null | wc -l)
        echo "       └─ $result_count arquivo(s) de resultado"
    else
        echo -e "${BLUE}[PENDENTE]${NC}"
    fi

    echo ""
}

monitor_specific_log() {
    local log_file="$1"

    if [ ! -f "$log_file" ]; then
        echo -e "${RED}Log file not found: $log_file${NC}"
        exit 1
    fi

    print_header "MONITORANDO: $log_file"

    echo ""
    tail -f "$log_file" | while IFS= read -r line; do
        # Colorize by log level
        if [[ $line =~ SUCCESS ]]; then
            echo -e "${GREEN}$line${NC}"
        elif [[ $line =~ ERROR ]]; then
            echo -e "${RED}$line${NC}"
        elif [[ $line =~ WARNING ]]; then
            echo -e "${YELLOW}$line${NC}"
        elif [[ $line =~ INFO ]]; then
            echo -e "${CYAN}$line${NC}"
        else
            echo "$line"
        fi
    done
}

main() {
    # Monitor specific log
    if [ -n "$SPECIFIC_LOG" ]; then
        monitor_specific_log "$SPECIFIC_LOG"
        exit 0
    fi

    # Show status once or continuously
    if [ "$FOLLOW_MODE" = true ]; then
        while true; do
            show_status
            sleep 5
        done
    else
        show_status
    fi
}

main
