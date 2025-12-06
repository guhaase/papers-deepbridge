# Usability Study - Task Descriptions

## Overview

You will complete **3 tasks** using DeepBridge. Please work independently and think aloud as you work. There is no time limit, but we ask that you complete the tasks efficiently.

Before starting, you will receive a 10-minute tutorial on DeepBridge basics.

---

## Task 1: Validate Fairness of Credit Model (8 minutes)

### Scenario
You work for a financial institution that has developed a machine learning model to predict credit approval. The model has been trained and is ready for deployment. However, before deploying, you must validate that it is fair with respect to protected attributes (gender and race).

### Your Task
Using DeepBridge, validate the fairness of the provided credit model and identify any violations.

### Materials Provided
- `credit_model.pkl` - Trained XGBoost model
- `credit_test_data.csv` - Test dataset (1,000 samples)
- Column descriptions:
  - Protected attributes: `gender` (0=Male, 1=Female), `race` (0=White, 1=Black, 2=Other)
  - Target: `approved` (0=Denied, 1=Approved)

### Steps
1. Load the model and test data
2. Create a `DBDataset` with the appropriate protected attributes
3. Run a fairness experiment with at least 15 metrics
4. Identify violations of the EEOC 80% rule
5. Interpret the results

### Deliverables
1. Code that runs the fairness validation
2. Identification of which protected attributes have violations
3. Values of the disparate impact (DI) metrics

### Success Criteria
- ✓ Successfully created DBDataset
- ✓ Ran fairness experiment
- ✓ Correctly identified violations (if any)
- ✓ Interpreted disparate impact values

**Start time**: _______
**End time**: _______
**Total time**: _______ minutes

**Errors encountered**: _______

---

## Task 2: Generate Audit-Ready PDF Report (4 minutes)

### Scenario
Your compliance team needs a professional PDF report with all validation results to submit to regulators. The report must include fairness, robustness, and uncertainty metrics, and should be branded with your company logo.

### Your Task
Generate a comprehensive PDF report from the fairness validation you just performed.

### Materials Provided
- Results from Task 1
- `company_logo.png` - Logo for report customization

### Steps
1. Configure report settings (include fairness, robustness, uncertainty)
2. Customize the report with the company logo
3. Generate the PDF report
4. Verify the report includes all expected sections

### Deliverables
1. Code that generates the report
2. PDF file named `credit_model_validation_report.pdf`

### Success Criteria
- ✓ Report generated successfully
- ✓ Report includes fairness section
- ✓ Report includes other validation sections (robustness, uncertainty)
- ✓ Company logo is visible in the report

**Start time**: _______
**End time**: _______
**Total time**: _______ minutes

**Errors encountered**: _______

---

## Task 3: Integrate Validation into CI/CD Pipeline (8 minutes)

### Scenario
Your team uses a CI/CD pipeline for model deployment. You need to integrate DeepBridge validation as an automated check that runs before deployment. If violations are detected, the pipeline should fail.

### Your Task
Create a Python script that can be called from a CI/CD pipeline to validate a model and return an appropriate exit code.

### Requirements
1. The script should accept model and data paths as command-line arguments
2. Run full validation (fairness, robustness, uncertainty)
3. Save the report to a specified directory
4. Return exit code 0 if no violations, exit code 1 if violations detected
5. Print a summary of violations to stdout

### Materials Provided
- Same model and data from Task 1
- Template: `validate_model_template.py`

### Steps
1. Create or modify the validation script
2. Add command-line argument parsing
3. Implement validation logic
4. Implement exit code logic based on violations
5. Test the script from the command line

### Deliverables
1. Python script `validate_model_cicd.py`
2. Demonstration that it works correctly

### Success Criteria
- ✓ Script runs from command line
- ✓ Accepts arguments (model path, data path, output dir)
- ✓ Detects violations correctly
- ✓ Returns correct exit code (0 = pass, 1 = fail)
- ✓ Saves report to specified directory

**Start time**: _______
**End time**: _______
**Total time**: _______ minutes

**Errors encountered**: _______

---

## Overall Summary

**Total time for all tasks**: _______ minutes

**Tasks completed successfully**: _____ / 3

**Overall experience**: (Circle one)
Very Difficult  |  Difficult  |  Neutral  |  Easy  |  Very Easy

**Comments**:
___________________________________________________________________________
___________________________________________________________________________
___________________________________________________________________________

---

## Notes for Facilitator

- Observe participant without intervention
- Note any errors, hesitations, or confusion
- Record completion times for each task
- Note any usability issues or positive comments
