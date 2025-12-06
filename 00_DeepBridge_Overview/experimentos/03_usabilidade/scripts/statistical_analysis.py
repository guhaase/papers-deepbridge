"""
Statistical Analysis of Usability Study

Performs statistical tests:
- One-sample t-test for SUS score vs. global average (68)
- Comparison with baseline (if available)
- Correlations between metrics
- Effect sizes
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.append(str(Path(__file__).parent))
from utils import setup_logging, save_results, load_results

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"


def one_sample_ttest_sus(sus_scores, population_mean=68, logger=None):
    """
    Perform one-sample t-test for SUS scores

    H0: SUS Score = 68 (global average for software)
    H1: SUS Score > 68
    """
    if logger:
        logger.info("Performing one-sample t-test for SUS scores...")

    t_stat, p_value_two_sided = stats.ttest_1samp(sus_scores, population_mean)

    # One-sided p-value (greater than)
    p_value_one_sided = p_value_two_sided / 2 if t_stat > 0 else 1 - p_value_two_sided / 2

    # Effect size (Cohen's d)
    mean_diff = np.mean(sus_scores) - population_mean
    pooled_std = np.std(sus_scores, ddof=1)
    cohens_d = mean_diff / pooled_std

    # 95% Confidence interval
    ci = stats.t.interval(0.95, len(sus_scores) - 1,
                          loc=np.mean(sus_scores),
                          scale=stats.sem(sus_scores))

    result = {
        'test': 'One-sample t-test',
        'null_hypothesis': f'SUS score = {population_mean}',
        'alternative_hypothesis': f'SUS score > {population_mean}',
        'sample_mean': float(np.mean(sus_scores)),
        'sample_std': float(np.std(sus_scores, ddof=1)),
        'population_mean': population_mean,
        't_statistic': float(t_stat),
        'p_value_one_sided': float(p_value_one_sided),
        'p_value_two_sided': float(p_value_two_sided),
        'cohens_d': float(cohens_d),
        'effect_size_interpretation': interpret_cohens_d(cohens_d),
        'ci_95_lower': float(ci[0]),
        'ci_95_upper': float(ci[1]),
        'significant': p_value_one_sided < 0.05,
        'n': len(sus_scores)
    }

    if logger:
        logger.info(f"  Sample Mean: {result['sample_mean']:.2f} ± {result['sample_std']:.2f}")
        logger.info(f"  t-statistic: {result['t_statistic']:.3f}")
        logger.info(f"  p-value (one-sided): {result['p_value_one_sided']:.4f}")
        logger.info(f"  Cohen's d: {result['cohens_d']:.3f} ({result['effect_size_interpretation']})")
        logger.info(f"  95% CI: [{result['ci_95_lower']:.2f}, {result['ci_95_upper']:.2f}]")

        if result['significant']:
            logger.info(f"  ✓ SIGNIFICANT: SUS score is significantly higher than {population_mean}")
        else:
            logger.info(f"  ✗ NOT SIGNIFICANT")

    return result


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def correlation_analysis(df, logger=None):
    """Analyze correlations between metrics"""
    if logger:
        logger.info("Analyzing correlations...")

    # Merge all data
    sus_df = pd.read_csv(RESULTS_DIR / "03_usability_sus_scores.csv")
    tlx_df = pd.read_csv(RESULTS_DIR / "03_usability_nasa_tlx.csv")
    times_df = pd.read_csv(RESULTS_DIR / "03_usability_task_times.csv")
    errors_df = pd.read_csv(RESULTS_DIR / "03_usability_errors.csv")

    merged = sus_df[['participant_id', 'sus_score']].merge(
        tlx_df[['participant_id', 'tlx_overall']], on='participant_id'
    ).merge(
        times_df[['participant_id', 'total_time']], on='participant_id'
    ).merge(
        errors_df[['participant_id', 'total_errors']], on='participant_id'
    )

    # Calculate correlations
    correlations = {}

    pairs = [
        ('sus_score', 'tlx_overall', 'SUS vs. TLX'),
        ('sus_score', 'total_time', 'SUS vs. Time'),
        ('sus_score', 'total_errors', 'SUS vs. Errors'),
        ('tlx_overall', 'total_time', 'TLX vs. Time'),
        ('tlx_overall', 'total_errors', 'TLX vs. Errors'),
        ('total_time', 'total_errors', 'Time vs. Errors')
    ]

    for var1, var2, label in pairs:
        r, p = stats.pearsonr(merged[var1], merged[var2])
        correlations[label] = {
            'r': float(r),
            'p_value': float(p),
            'significant': p < 0.05,
            'interpretation': interpret_correlation(r)
        }

        if logger:
            sig = "*" if p < 0.05 else ""
            logger.info(f"  {label}: r = {r:.3f}, p = {p:.4f}{sig}")

    return correlations


def interpret_correlation(r):
    """Interpret correlation coefficient"""
    abs_r = abs(r)
    if abs_r < 0.1:
        return "negligible"
    elif abs_r < 0.3:
        return "weak"
    elif abs_r < 0.5:
        return "moderate"
    else:
        return "strong"


def normality_tests(data, name, logger=None):
    """Test for normality using Shapiro-Wilk test"""
    if logger:
        logger.info(f"Testing normality for {name}...")

    stat, p = stats.shapiro(data)

    result = {
        'test': 'Shapiro-Wilk',
        'statistic': float(stat),
        'p_value': float(p),
        'normal': p > 0.05,
        'n': len(data)
    }

    if logger:
        logger.info(f"  W = {stat:.4f}, p = {p:.4f}")
        logger.info(f"  {'✓ Normal distribution' if result['normal'] else '✗ Not normal distribution'}")

    return result


def main():
    """Main execution"""
    logger = setup_logging("statistical_analysis", LOGS_DIR)

    logger.info("=" * 80)
    logger.info("STATISTICAL ANALYSIS")
    logger.info("=" * 80)

    # Load data
    sus_df = pd.read_csv(RESULTS_DIR / "03_usability_sus_scores.csv")
    tlx_df = pd.read_csv(RESULTS_DIR / "03_usability_nasa_tlx.csv")
    times_df = pd.read_csv(RESULTS_DIR / "03_usability_task_times.csv")

    sus_scores = sus_df['sus_score'].values
    tlx_scores = tlx_df['tlx_overall'].values
    total_times = times_df['total_time'].dropna().values

    # 1. One-sample t-test for SUS
    logger.info("")
    logger.info("1. ONE-SAMPLE T-TEST: SUS SCORE")
    logger.info("-" * 80)
    sus_ttest = one_sample_ttest_sus(sus_scores, population_mean=68, logger=logger)

    # 2. Normality tests
    logger.info("")
    logger.info("2. NORMALITY TESTS")
    logger.info("-" * 80)
    sus_normality = normality_tests(sus_scores, "SUS Scores", logger)
    tlx_normality = normality_tests(tlx_scores, "NASA TLX Scores", logger)
    time_normality = normality_tests(total_times, "Total Times", logger)

    # 3. Correlation analysis
    logger.info("")
    logger.info("3. CORRELATION ANALYSIS")
    logger.info("-" * 80)
    correlations = correlation_analysis(None, logger)

    # 4. Descriptive statistics
    logger.info("")
    logger.info("4. DESCRIPTIVE STATISTICS")
    logger.info("-" * 80)
    logger.info(f"SUS Score: Mean = {np.mean(sus_scores):.2f}, SD = {np.std(sus_scores, ddof=1):.2f}")
    logger.info(f"NASA TLX: Mean = {np.mean(tlx_scores):.2f}, SD = {np.std(tlx_scores, ddof=1):.2f}")
    logger.info(f"Total Time: Mean = {np.mean(total_times):.2f}, SD = {np.std(total_times, ddof=1):.2f} min")

    # Compile results
    statistical_results = {
        'sus_ttest': sus_ttest,
        'normality_tests': {
            'sus_scores': sus_normality,
            'nasa_tlx_scores': tlx_normality,
            'total_times': time_normality
        },
        'correlations': correlations
    }

    # Save results
    output_file = RESULTS_DIR / "03_usability_statistical_analysis.json"
    save_results(statistical_results, output_file, logger)

    logger.info("")
    logger.info("=" * 80)
    logger.info("STATISTICAL ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
