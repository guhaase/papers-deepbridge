"""
Case Studies Experiment Scripts

This package contains scripts for running 6 case studies across different domains:
1. Credit Scoring
2. Hiring
3. Healthcare
4. Mortgage
5. Insurance
6. Fraud Detection
"""

__version__ = "1.0.0"
__author__ = "DeepBridge Team"

# Make modules available at package level
from . import utils
from . import case_study_credit
from . import case_study_hiring
from . import case_study_healthcare
from . import case_study_mortgage
from . import case_study_insurance
from . import case_study_fraud

__all__ = [
    'utils',
    'case_study_credit',
    'case_study_hiring',
    'case_study_healthcare',
    'case_study_mortgage',
    'case_study_insurance',
    'case_study_fraud',
]
