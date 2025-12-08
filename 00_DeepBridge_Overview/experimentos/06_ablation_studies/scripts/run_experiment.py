"""
Run Complete Ablation Studies Experiment

Orchestrates the full experiment:
1. Run ablation studies (all configurations)
2. Analyze and visualize results

Autor: DeepBridge Team
Data: 2025-12-07
"""

import sys
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import setup_logging

BASE_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
LOGS_DIR = BASE_DIR / "logs"


def run_script(script_name, logger):
    """Run a Python script and log output"""
    script_path = SCRIPTS_DIR / script_name

    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {script_name}")
    logger.info(f"{'='*80}\n")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )

        # Log stdout
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"  {line}")

        logger.info(f"\n✓ {script_name} completed successfully\n")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"\n✗ {script_name} failed!")
        logger.error(f"Error: {e}")

        if e.stdout:
            logger.error("STDOUT:")
            for line in e.stdout.splitlines():
                logger.error(f"  {line}")

        if e.stderr:
            logger.error("STDERR:")
            for line in e.stderr.splitlines():
                logger.error(f"  {line}")

        return False


def main():
    """Main execution"""
    logger = setup_logging("run_experiment", LOGS_DIR)

    logger.info("=" * 80)
    logger.info("ABLATION STUDIES EXPERIMENT - COMPLETE RUN")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This experiment will:")
    logger.info("  1. Run ablation studies (6 configurations × 10 runs)")
    logger.info("  2. Analyze results and calculate component contributions")
    logger.info("  3. Generate visualizations (waterfall, stacked bar, boxplot)")
    logger.info("")
    logger.info("Estimated time: ~1-2 minutes")
    logger.info("")
    logger.info("=" * 80)

    # Ask for confirmation
    response = input("\nProceed with full experiment? (y/N): ")
    if response.lower() != 'y':
        logger.info("Experiment cancelled by user")
        return

    # Step 1: Run ablation
    if not run_script("run_ablation.py", logger):
        logger.error("Experiment failed at step 1")
        return

    # Step 2: Analyze and visualize
    if not run_script("analyze_and_visualize.py", logger):
        logger.error("Experiment failed at step 2")
        return

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Output locations:")
    logger.info(f"  - Results:    {BASE_DIR}/results")
    logger.info(f"  - Figures:    {BASE_DIR}/figures")
    logger.info(f"  - Tables:     {BASE_DIR}/tables")
    logger.info(f"  - Logs:       {BASE_DIR}/logs")
    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
