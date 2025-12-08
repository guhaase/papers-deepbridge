#!/bin/bash
################################################################################
# Script Master - Execução de Todos os Experimentos (Dados Reais)
################################################################################
#
# Executa em sequência os experimentos 1, 2, 3, 5, 6 com datasets reais
# Experimento 4 (HPMKD) será executado separadamente em servidor com GPU
#
# Uso: ./run_all_experiments.sh [OPTIONS]
#
# Opções:
#   --skip-exp1     Pular experimento 1 (benchmarks)
#   --skip-exp2     Pular experimento 2 (estudos de caso)
#   --skip-exp3     Pular experimento 3 (usabilidade)
#   --skip-exp5     Pular experimento 5 (conformidade)
#   --skip-exp6     Pular experimento 6 (ablation)
#   --dry-run       Mostrar o que seria executado sem executar
#   --help          Mostrar esta mensagem
#
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos"
cd "$BASE_DIR"

# Timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$BASE_DIR/logs_master"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/master_${TIMESTAMP}.log"
SUMMARY_FILE="$LOG_DIR/summary_${TIMESTAMP}.txt"

# Flags
SKIP_EXP1=false
SKIP_EXP2=false
SKIP_EXP3=false
SKIP_EXP5=false
SKIP_EXP6=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-exp1) SKIP_EXP1=true; shift ;;
        --skip-exp2) SKIP_EXP2=true; shift ;;
        --skip-exp3) SKIP_EXP3=true; shift ;;
        --skip-exp5) SKIP_EXP5=true; shift ;;
        --skip-exp6) SKIP_EXP6=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help)
            grep "^#" "$0" | grep -v "^#!/" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Opção desconhecida: $1${NC}"
            echo "Use --help para ver as opções disponíveis"
            exit 1
            ;;
    esac
done

################################################################################
# Funções auxiliares
################################################################################

log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")

    case $level in
        INFO)  echo -e "${CYAN}[INFO]${NC} $message" ;;
        SUCCESS) echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
        WARNING) echo -e "${YELLOW}[WARNING]${NC} $message" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} $message" ;;
        HEADER) echo -e "${MAGENTA}[HEADER]${NC} $message" ;;
    esac

    echo "[$timestamp] [$level] $message" >> "$MASTER_LOG"
}

print_header() {
    local text="$1"
    local width=80
    local padding=$(printf '%*s' $(((width - ${#text}) / 2)) '')

    echo ""
    echo -e "${MAGENTA}$(printf '=%.0s' $(seq 1 $width))${NC}"
    echo -e "${MAGENTA}${padding}${text}${NC}"
    echo -e "${MAGENTA}$(printf '=%.0s' $(seq 1 $width))${NC}"
    echo ""
}

print_separator() {
    echo -e "${BLUE}$(printf -- '-%.0s' $(seq 1 80))${NC}"
}

run_experiment() {
    local exp_num=$1
    local exp_name=$2
    local exp_dir=$3
    local exp_command=$4

    print_header "EXPERIMENTO $exp_num: $exp_name"

    log INFO "Iniciando experimento $exp_num: $exp_name"
    log INFO "Diretório: $exp_dir"
    log INFO "Comando: $exp_command"

    if [ "$DRY_RUN" = true ]; then
        log INFO "[DRY RUN] Pulando execução"
        return 0
    fi

    # Create experiment log
    local exp_log="$LOG_DIR/exp${exp_num}_${TIMESTAMP}.log"

    # Start time
    local start_time=$(date +%s)

    # Execute command
    cd "$exp_dir"
    log INFO "Executando comando..."

    if eval "$exp_command" > "$exp_log" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local duration_formatted=$(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60)))

        log SUCCESS "Experimento $exp_num concluído com sucesso"
        log INFO "Tempo de execução: $duration_formatted"

        echo "EXP$exp_num|$exp_name|SUCCESS|$duration_formatted" >> "$SUMMARY_FILE"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local duration_formatted=$(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60)))

        log ERROR "Experimento $exp_num falhou!"
        log ERROR "Verifique o log em: $exp_log"
        log INFO "Tempo antes da falha: $duration_formatted"

        echo "EXP$exp_num|$exp_name|FAILED|$duration_formatted" >> "$SUMMARY_FILE"

        # Mostrar últimas linhas do log
        log ERROR "Últimas 20 linhas do log:"
        tail -n 20 "$exp_log" | while IFS= read -r line; do
            echo "  $line" >> "$MASTER_LOG"
        done

        return 1
    fi
}

check_requirements() {
    log INFO "Verificando requisitos..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log ERROR "Python 3 não encontrado!"
        exit 1
    fi

    local python_version=$(python3 --version | cut -d' ' -f2)
    log INFO "Python version: $python_version"

    # Check disk space
    local disk_space=$(df -h "$BASE_DIR" | tail -1 | awk '{print $4}')
    log INFO "Espaço em disco disponível: $disk_space"

    log SUCCESS "Verificações concluídas"
}

generate_summary() {
    print_header "RESUMO DA EXECUÇÃO"

    echo ""
    echo "RESUMO DA EXECUÇÃO - $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
    echo "==========================================================================" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    if [ -f "$SUMMARY_FILE" ]; then
        printf "%-6s %-30s %-10s %-15s\n" "EXP" "NOME" "STATUS" "TEMPO" | tee -a "$MASTER_LOG"
        echo "--------------------------------------------------------------------------" | tee -a "$MASTER_LOG"

        local total_success=0
        local total_failed=0

        while IFS='|' read -r exp name status time; do
            if [ "$status" = "SUCCESS" ]; then
                echo -e "${GREEN}$exp${NC}\t$name\t$status\t$time" | tee -a "$MASTER_LOG"
                ((total_success++))
            else
                echo -e "${RED}$exp${NC}\t$name\t$status\t$time" | tee -a "$MASTER_LOG"
                ((total_failed++))
            fi
        done < "$SUMMARY_FILE"

        echo "" | tee -a "$MASTER_LOG"
        echo "--------------------------------------------------------------------------" | tee -a "$MASTER_LOG"
        echo "Total: $((total_success + total_failed)) experimentos" | tee -a "$MASTER_LOG"
        echo -e "${GREEN}Sucesso: $total_success${NC}" | tee -a "$MASTER_LOG"
        echo -e "${RED}Falhas: $total_failed${NC}" | tee -a "$MASTER_LOG"
    else
        log WARNING "Arquivo de resumo não encontrado"
    fi

    echo "" | tee -a "$MASTER_LOG"
    echo "Logs salvos em: $LOG_DIR" | tee -a "$MASTER_LOG"
    echo "Log master: $MASTER_LOG" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"
}

estimate_time() {
    log INFO "Estimativa de tempo total:"
    echo ""
    echo "  Experimento 1 (Benchmarks):      ~3-4 horas  (10 runs × ~17 min)"
    echo "  Experimento 2 (Estudos de Caso): ~2-3 horas  (3 casos × ~50 min)"
    echo "  Experimento 3 (Usabilidade):     ~5 min      (análise e figuras)"
    echo "  Experimento 5 (Conformidade):    ~5 min      (50 casos + análise)"
    echo "  Experimento 6 (Ablation):        ~2 min      (6 configs × 10 runs + análise)"
    echo ""
    echo "  TOTAL ESTIMADO: ~5-7 horas (executando sequencialmente)"
    echo ""
    echo "  OBS: Experimentos 3, 5 e 6 são rápidos (dados sintéticos)"
    echo "       Experimentos 1 e 2 são mais demorados (dados reais complexos)"
    echo ""
    log WARNING "Recomenda-se executar durante período livre para Exp 1 e 2"
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    # Print banner
    clear
    print_header "EXECUÇÃO DE TODOS OS EXPERIMENTOS - DADOS REAIS"

    log INFO "Iniciando script master de experimentos"
    log INFO "Timestamp: $TIMESTAMP"
    log INFO "Diretório base: $BASE_DIR"
    log INFO "Diretório de logs: $LOG_DIR"

    # System info
    print_separator
    log INFO "Informações do sistema:"
    log INFO "  Hostname: $(hostname)"
    log INFO "  OS: $(uname -s)"
    log INFO "  Kernel: $(uname -r)"
    log INFO "  CPU cores: $(nproc)"
    log INFO "  RAM: $(free -h | grep Mem | awk '{print $2}')"
    print_separator

    # Check requirements
    check_requirements

    # Show estimate
    estimate_time

    # Confirmation
    if [ "$DRY_RUN" = false ]; then
        read -p "Deseja continuar? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log INFO "Execução cancelada pelo usuário"
            exit 0
        fi
    fi

    # Track overall start time
    OVERALL_START=$(date +%s)

    # Execute experiments
    local continue_on_error=true

    # Experiment 1: Benchmarks de Tempo
    if [ "$SKIP_EXP1" = false ]; then
        if ! run_experiment \
            "1" \
            "Benchmarks de Tempo" \
            "$BASE_DIR/01_benchmarks_tempo" \
            "echo 'y' | python3 scripts/run_experiment.py"; then

            if [ "$continue_on_error" = false ]; then
                log ERROR "Parando execução devido a falha no Experimento 1"
                exit 1
            fi
        fi
    else
        log INFO "Pulando Experimento 1 (--skip-exp1)"
        echo "EXP1|Benchmarks de Tempo|SKIPPED|00:00:00" >> "$SUMMARY_FILE"
    fi

    # Experiment 2: Estudos de Caso
    if [ "$SKIP_EXP2" = false ]; then
        if ! run_experiment \
            "2" \
            "Estudos de Caso" \
            "$BASE_DIR/02_estudos_de_caso" \
            "python3 scripts/run_all_cases.py"; then

            if [ "$continue_on_error" = false ]; then
                log ERROR "Parando execução devido a falha no Experimento 2"
                exit 1
            fi
        fi
    else
        log INFO "Pulando Experimento 2 (--skip-exp2)"
        echo "EXP2|Estudos de Caso|SKIPPED|00:00:00" >> "$SUMMARY_FILE"
    fi

    # Experiment 3: Usabilidade
    if [ "$SKIP_EXP3" = false ]; then
        if ! run_experiment \
            "3" \
            "Usabilidade" \
            "$BASE_DIR/03_usabilidade" \
            "python3 scripts/generate_mock_data.py && python3 scripts/analyze_usability.py && python3 scripts/generate_visualizations.py"; then

            if [ "$continue_on_error" = false ]; then
                log ERROR "Parando execução devido a falha no Experimento 3"
                exit 1
            fi
        fi
    else
        log INFO "Pulando Experimento 3 (--skip-exp3)"
        echo "EXP3|Usabilidade|SKIPPED|00:00:00" >> "$SUMMARY_FILE"
    fi

    # Experiment 5: Conformidade Regulatória
    if [ "$SKIP_EXP5" = false ]; then
        if ! run_experiment \
            "5" \
            "Conformidade Regulatória" \
            "$BASE_DIR/05_conformidade" \
            "echo 'y' | python3 scripts/run_experiment.py"; then

            if [ "$continue_on_error" = false ]; then
                log ERROR "Parando execução devido a falha no Experimento 5"
                exit 1
            fi
        fi
    else
        log INFO "Pulando Experimento 5 (--skip-exp5)"
        echo "EXP5|Conformidade|SKIPPED|00:00:00" >> "$SUMMARY_FILE"
    fi

    # Experiment 6: Ablation Studies
    if [ "$SKIP_EXP6" = false ]; then
        if ! run_experiment \
            "6" \
            "Ablation Studies" \
            "$BASE_DIR/06_ablation_studies" \
            "echo 'y' | python3 scripts/run_experiment.py"; then

            if [ "$continue_on_error" = false ]; then
                log ERROR "Parando execução devido a falha no Experimento 6"
                exit 1
            fi
        fi
    else
        log INFO "Pulando Experimento 6 (--skip-exp6)"
        echo "EXP6|Ablation Studies|SKIPPED|00:00:00" >> "$SUMMARY_FILE"
    fi

    # Calculate total time
    OVERALL_END=$(date +%s)
    OVERALL_DURATION=$((OVERALL_END - OVERALL_START))
    OVERALL_FORMATTED=$(printf '%02d:%02d:%02d' $((OVERALL_DURATION/3600)) $((OVERALL_DURATION%3600/60)) $((OVERALL_DURATION%60)))

    # Generate summary
    generate_summary

    log SUCCESS "Execução completa!"
    log INFO "Tempo total: $OVERALL_FORMATTED"

    print_separator
    echo ""
    echo -e "${YELLOW}NOTA:${NC} O Experimento 4 (HPMKD) deve ser executado separadamente em servidor com GPU"
    echo "      Localização: $BASE_DIR/04_hpmkd"
    echo ""
}

# Trap errors
trap 'log ERROR "Script interrompido na linha $LINENO"' ERR

# Run main
main "$@"
