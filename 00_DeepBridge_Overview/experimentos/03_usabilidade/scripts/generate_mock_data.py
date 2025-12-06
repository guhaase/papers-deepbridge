"""
Generate Mock Usability Study Data

This script generates synthetic data simulating a usability study with 20 participants.
This is useful for:
1. Testing the analysis pipeline
2. Demonstrating expected results
3. Validating visualization and reporting code

In a real study, this would be replaced with actual participant data.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import (
    setup_logging, generate_participant_id,
    calculate_sus_score, calculate_nasa_tlx,
    save_results
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def generate_participant_demographics(n_participants=20, seed=42):
    """Generate participant demographic data"""
    np.random.seed(seed)

    participants = []

    # Distribution: 10 data scientists, 10 ML engineers
    roles = ['Data Scientist'] * 10 + ['ML Engineer'] * 10
    np.random.shuffle(roles)

    # Domain distribution
    domains = (
        ['Fintech'] * 8 +
        ['Healthcare'] * 5 +
        ['Tech'] * 4 +
        ['Retail'] * 3
    )
    np.random.shuffle(domains)

    for i in range(n_participants):
        participant_id = generate_participant_id(i + 1)

        # Experience: 2-10 years, with bias towards middle
        experience = np.random.beta(2, 2) * 8 + 2

        participants.append({
            'participant_id': participant_id,
            'role': roles[i],
            'domain': domains[i],
            'years_experience': round(experience, 1),
            'python_proficiency': np.random.choice(['Intermediate', 'Advanced', 'Expert'],
                                                   p=[0.2, 0.5, 0.3]),
            'ml_deployment_experience': np.random.choice(['Yes', 'No'], p=[0.85, 0.15])
        })

    return pd.DataFrame(participants)


def generate_sus_responses(n_participants=20, target_mean=87.5, seed=42):
    """
    Generate SUS responses

    Target: Mean SUS = 87.5 ± 3.2 (top 10%, "excellent")
    """
    np.random.seed(seed)

    sus_data = []

    for i in range(n_participants):
        participant_id = generate_participant_id(i + 1)

        # Generate responses that will result in target SUS score
        # SUS questions alternate between positive and negative framing

        # Target score for this participant (normal distribution around target_mean)
        target_score = np.random.normal(target_mean, 3.2)
        target_score = np.clip(target_score, 70, 100)  # Keep in realistic range

        # Reverse engineer responses from target score
        # SUS score = sum of contributions * 2.5
        # Contribution = (response - 1) for odd items, (5 - response) for even items
        target_contribution = target_score / 2.5

        # Generate responses that sum to target contribution
        responses = []
        remaining = target_contribution

        for j in range(10):
            if j < 9:
                # Random contribution between 0-4
                contrib = np.random.uniform(0, min(4, remaining))
                remaining -= contrib
            else:
                # Last item: use remaining contribution
                contrib = np.clip(remaining, 0, 4)

            # Convert contribution back to response (1-5 scale)
            if j % 2 == 0:  # Odd items (positive)
                response = int(np.clip(np.round(contrib + 1), 1, 5))
            else:  # Even items (negative)
                response = int(np.clip(np.round(5 - contrib), 1, 5))

            responses.append(response)

        # Calculate actual SUS score
        actual_score = calculate_sus_score(responses)

        sus_data.append({
            'participant_id': participant_id,
            'q1_frequency': responses[0],
            'q2_complex': responses[1],
            'q3_easy': responses[2],
            'q4_support': responses[3],
            'q5_integrated': responses[4],
            'q6_inconsistent': responses[5],
            'q7_learn_quickly': responses[6],
            'q8_complicated': responses[7],
            'q9_confident': responses[8],
            'q10_learn_much': responses[9],
            'sus_score': actual_score
        })

    return pd.DataFrame(sus_data)


def generate_nasa_tlx_responses(n_participants=20, target_mean=28, seed=42):
    """
    Generate NASA TLX responses

    Target: Mean TLX = 28 ± 5.1 (low workload)
    """
    np.random.seed(seed + 1)

    tlx_data = []

    for i in range(n_participants):
        participant_id = generate_participant_id(i + 1)

        # Target overall score for this participant
        target_score = np.random.normal(target_mean, 5.1)
        target_score = np.clip(target_score, 15, 45)  # Keep in realistic range

        # Generate individual dimension scores
        # Some dimensions typically higher/lower for software tasks
        mental_demand = np.random.normal(target_score * 1.2, 8)  # Slightly higher
        physical_demand = np.random.normal(10, 5)  # Very low for software
        temporal_demand = np.random.normal(target_score * 0.9, 8)
        performance = 100 - np.random.normal(target_score * 0.8, 10)  # Inverted (high = good)
        effort = np.random.normal(target_score * 1.1, 8)
        frustration = np.random.normal(target_score * 0.7, 10)

        # Clip to valid range
        dimensions = {
            'mental_demand': np.clip(mental_demand, 0, 100),
            'physical_demand': np.clip(physical_demand, 0, 100),
            'temporal_demand': np.clip(temporal_demand, 0, 100),
            'performance': np.clip(performance, 0, 100),
            'effort': np.clip(effort, 0, 100),
            'frustration': np.clip(frustration, 0, 100)
        }

        overall = calculate_nasa_tlx(dimensions)

        tlx_data.append({
            'participant_id': participant_id,
            **{k: round(v, 1) for k, v in dimensions.items()},
            'tlx_overall': round(overall, 1)
        })

    return pd.DataFrame(tlx_data)


def generate_task_times(n_participants=20, seed=42):
    """
    Generate task completion times

    Expected times:
    - Task 1: 6.5 ± 1.2 min
    - Task 2: 2.8 ± 0.8 min
    - Task 3: 6.2 ± 1.5 min
    - Total: 12 ± 2.5 min
    """
    np.random.seed(seed + 2)

    time_data = []

    # One participant fails (participant 19 out of 20)
    failed_participant = 18  # 0-indexed

    for i in range(n_participants):
        participant_id = generate_participant_id(i + 1)

        if i == failed_participant:
            # This participant fails task 3
            task1_time = np.random.normal(6.5, 1.2)
            task2_time = np.random.normal(2.8, 0.8)
            task3_time = np.nan  # Failed
            completed_all = False
        else:
            # Normal completion
            task1_time = np.random.normal(6.5, 1.2)
            task2_time = np.random.normal(2.8, 0.8)
            task3_time = np.random.normal(6.2, 1.5)
            completed_all = True

        # Clip to reasonable bounds
        task1_time = max(3, task1_time)
        task2_time = max(1, task2_time)
        if not np.isnan(task3_time):
            task3_time = max(3, task3_time)

        time_data.append({
            'participant_id': participant_id,
            'task1_fairness_time': round(task1_time, 2),
            'task2_report_time': round(task2_time, 2),
            'task3_cicd_time': round(task3_time, 2) if not np.isnan(task3_time) else None,
            'total_time': round(task1_time + task2_time + (task3_time if not np.isnan(task3_time) else 0), 2),
            'completed_all_tasks': completed_all,
            'task1_success': True,
            'task2_success': True,
            'task3_success': not np.isnan(task3_time)
        })

    return pd.DataFrame(time_data)


def generate_errors(n_participants=20, seed=42):
    """
    Generate error counts

    Expected: 1.3 ± 0.9 errors per participant
    """
    np.random.seed(seed + 3)

    error_data = []

    for i in range(n_participants):
        participant_id = generate_participant_id(i + 1)

        # Total errors (Poisson distribution)
        n_errors = np.random.poisson(1.3)

        # Distribute among categories
        if n_errors > 0:
            syntax_errors = np.random.binomial(n_errors, 0.2)
            api_errors = np.random.binomial(n_errors - syntax_errors, 0.4)
            conceptual_errors = np.random.binomial(n_errors - syntax_errors - api_errors, 0.5)
            other_errors = n_errors - syntax_errors - api_errors - conceptual_errors
        else:
            syntax_errors = api_errors = conceptual_errors = other_errors = 0

        error_data.append({
            'participant_id': participant_id,
            'syntax_errors': syntax_errors,
            'api_errors': api_errors,
            'conceptual_errors': conceptual_errors,
            'other_errors': other_errors,
            'total_errors': n_errors
        })

    return pd.DataFrame(error_data)


def main():
    """Generate all mock data"""
    logger = setup_logging("generate_mock_data", LOGS_DIR)

    logger.info("=" * 80)
    logger.info("GENERATING MOCK USABILITY STUDY DATA")
    logger.info("=" * 80)

    # Generate data
    logger.info("Generating participant demographics...")
    demographics = generate_participant_demographics()
    demographics.to_csv(DATA_DIR / "participants_demographics.csv", index=False)
    logger.info(f"  Generated {len(demographics)} participants")

    logger.info("Generating SUS responses...")
    sus_responses = generate_sus_responses()
    sus_responses.to_csv(RESULTS_DIR / "03_usability_sus_scores.csv", index=False)
    logger.info(f"  Mean SUS Score: {sus_responses['sus_score'].mean():.2f} ± {sus_responses['sus_score'].std():.2f}")

    logger.info("Generating NASA TLX responses...")
    tlx_responses = generate_nasa_tlx_responses()
    tlx_responses.to_csv(RESULTS_DIR / "03_usability_nasa_tlx.csv", index=False)
    logger.info(f"  Mean TLX Score: {tlx_responses['tlx_overall'].mean():.2f} ± {tlx_responses['tlx_overall'].std():.2f}")

    logger.info("Generating task completion times...")
    task_times = generate_task_times()
    task_times.to_csv(RESULTS_DIR / "03_usability_task_times.csv", index=False)
    logger.info(f"  Mean Total Time: {task_times['total_time'].mean():.2f} ± {task_times['total_time'].std():.2f} min")
    logger.info(f"  Success Rate: {task_times['completed_all_tasks'].sum()}/{len(task_times)} ({task_times['completed_all_tasks'].sum()/len(task_times)*100:.1f}%)")

    logger.info("Generating error counts...")
    errors = generate_errors()
    errors.to_csv(RESULTS_DIR / "03_usability_errors.csv", index=False)
    logger.info(f"  Mean Errors: {errors['total_errors'].mean():.2f} ± {errors['total_errors'].std():.2f}")

    logger.info("=" * 80)
    logger.info("MOCK DATA GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Data saved to: {DATA_DIR}")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
