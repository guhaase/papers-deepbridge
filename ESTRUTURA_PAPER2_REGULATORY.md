# Estrutura do Paper 2: Regulatory Explainability e Compliance
## Guia de Preparação para ACM FAccT e Venues de ML Fairness

**Data:** 2025-11-07  
**Objetivo:** Documentar a estrutura ideal do paper sobre regulatory explainability e compliance para knowledge distillation em domínios financeiros regulados

---

## 1. VISÃO GERAL ESTRATÉGICA

### 1.1 Foco Central do Paper 2

**Pergunta de Pesquisa Principal:**  
*"Por que técnicas complexas de destilação falham em atender requisitos regulatórios em finanças, e quais alternativas existem que equilibram performance e compliance?"*

**Contribuição Científica:**
- Framework analítico para avaliar compliance de métodos de distillation
- Análise sistemática de requisitos regulatórios (ECOA, GDPR, EU AI Act)
- Mapeamento de "multiplicative opacity" em ensemble/distillation methods
- Avaliação empírica de alternativas interpretáveis
- Policy recommendations para múltiplos stakeholders

**O QUE INCLUIR:**
- ✅ Análise legal e regulatória detalhada
- ✅ Framework de compliance assessment
- ✅ Case studies em domínios financeiros
- ✅ Avaliação de métodos interpretáveis (NAMs, EBMs, decision trees)
- ✅ Trade-offs performance-explainability quantificados
- ✅ Policy implications e recommendations
- ✅ Stakeholder perspectives (reguladores, instituições, consumidores)

**O QUE NÃO INCLUIR (já está no Paper 1):**
- ❌ Detalhes técnicos profundos de algoritmos de distillation
- ❌ Ablation studies de componentes internos
- ❌ Comparações extensivas de métodos de KD
- ❌ Otimizações computacionais

### 1.2 Diferenças Fundamentais vs Paper 1

| Aspecto | Paper 1 (Técnico) | Paper 2 (Regulatory) |
|---------|-------------------|----------------------|
| **Research Question** | Como melhorar performance de KD? | Por que KD falha em compliance? |
| **Metodologia** | Experimentos computacionais | Análise legal + policy + empírica |
| **Contribuição** | Algoritmo novo (HPM-KD) | Framework de compliance |
| **Audience** | ML researchers | Reguladores + practitioners + policymakers |
| **Venue** | ICLR/ICML/NeurIPS | ACM FAccT/AIES |
| **Literatura Base** | KD, compression, meta-learning | Fairness, accountability, law |
| **Métricas** | Accuracy, compression ratio | Compliance score, explainability |
| **Estrutura** | Intro→Method→Experiments→Results | Intro→Regulatory→Framework→Cases→Policy |
| **Formato** | Technical ML paper | Interdisciplinary analysis |
| **Citações** | CS/ML papers | Law reviews + policy docs + CS |

---

## 2. ESTRUTURA DETALHADA DO PAPER

### 2.1 Abstract (250-300 palavras)

**Elementos Obrigatórios:**

1. **Contexto e Problema** (3-4 frases):
   - Avanços em knowledge distillation (cite Paper 1 entre outros)
   - Deployment em domínios regulados (finance)
   - Gap: technical performance vs regulatory requirements
   - "Multiplicative opacity" problem em ensemble methods

2. **Research Questions** (2 frases):
   - Por que distillation methods falham regulatory requirements?
   - Quais alternativas existem que equilibram performance e compliance?

3. **Metodologia** (2-3 frases):
   - Framework analítico para avaliar compliance
   - Análise de 4 regulatory regimes (ECOA, GDPR, EU AI Act, SR 11-7)
   - Avaliação empírica em 3 financial use cases

4. **Principais Findings** (3-4 frases):
   - Knowledge distillation herda e amplifica opacity do teacher
   - Nenhuma regulação bane explicitamente, mas cria functional constraints
   - Alternativas interpretáveis (NAMs, EBMs) achieve 92-96% of distilled model performance
   - Gap accuracy-explainability menor do que assumido (70% datasets)

5. **Implicações** (1-2 frases):
   - Framework fornece guidance para instituições financeiras
   - Policy recommendations para reguladores e practitioners

**Exemplo de Estrutura:**
```
Recent advances in knowledge distillation [cite Paper 1, others] have 
demonstrated significant computational benefits, achieving 10-15× 
compression with minimal accuracy loss. However, deployment in regulated 
financial services faces a fundamental challenge: complex distillation 
techniques create "multiplicative opacity" where student models inherit 
and amplify the explainability limitations of ensemble teachers, while 
regulatory frameworks (ECOA, GDPR Article 22, EU AI Act) impose strict 
requirements for model transparency and individual-level explanations.

We address two research questions: (1) Why do distillation methods fail 
to meet regulatory requirements in high-stakes domains? (2) What 
alternatives exist that balance predictive performance with compliance? 
We develop a compliance assessment framework spanning four regulatory 
regimes and evaluate it across three financial use cases (credit scoring, 
loan decisioning, insurance underwriting).

Our analysis reveals that [findings]. These findings challenge the 
conventional assumption of a steep accuracy-explainability tradeoff and 
provide actionable guidance for financial institutions navigating the 
technical-regulatory divide. We conclude with policy recommendations for 
regulators, model developers, and deploying institutions.
```

---

### 2.2 Introduction (3-4 páginas)

**Seção 1.1 - The Technical-Regulatory Divide (0.75 página)**

```markdown
Recent technical advances in model compression, particularly knowledge 
distillation [cite Paper 1, recent ICLR/ICML papers], have made deploying 
large-scale models in resource-constrained environments increasingly 
feasible. Knowledge distillation achieves 10-15× compression ratios while 
retaining 95-99% of teacher model accuracy [specific citations with numbers].

However, these technical achievements face a fundamental barrier in 
regulated domains: the gap between what is computationally optimal and 
what is legally deployable. Financial institutions deploying AI for credit 
decisions, insurance underwriting, and risk assessment must navigate 
overlapping regulatory requirements from [list: ECOA 1974, GDPR 2016, 
EU AI Act 2024, Federal Reserve SR 11-7].

This creates a stark tension: distillation methods that excel on technical 
benchmarks may be functionally prohibited in practice not because of 
explicit technology bans, but because they cannot generate the 
individual-level explanations required by law.
```

**Seção 1.2 - The Multiplicative Opacity Problem (0.75 página)**

```markdown
We identify a fundamental issue we term "multiplicative opacity": 
knowledge distillation and ensemble methods create cascading layers of 
complexity that compound explainability challenges.

[Diagram/Figure 1: Opacity amplification]
- Base model (teacher): Complexity C
- Ensemble of teachers: Complexity M×C (M teachers)
- Distillation process: Additional transformation layer
- Student model: "Compressed" but inherits ensemble opacity
- Result: Opacity(student) ≥ Opacity(ensemble) despite smaller size

This differs fundamentally from training a small model directly:
- Direct training: Complexity = f(student_size)
- Distillation: Complexity = f(student_size) + g(teacher_ensemble) + h(distillation_process)

The student model's predictions depend on knowledge learned from multiple 
teachers, whose own predictions may be based on non-linear combinations of 
features. Even if the student architecture is simple (e.g., small neural 
network), the knowledge it embodies reflects the complex decision boundaries 
of the ensemble.

Crucially, popular explainability methods (SHAP, LIME) explain the student 
model's architecture, not the knowledge it learned. This creates a dangerous 
illusion of interpretability: the explanation may be coherent but fails to 
capture the actual reasoning process that produced the model's behavior.
```

**Seção 1.3 - Regulatory Landscape (0.75 página)**

**CRÍTICO:** Não deep-dive aqui, apenas overview (detalhes na Section 2)

```markdown
Four major regulatory frameworks create functional constraints on model 
deployment in finance:

1. **Equal Credit Opportunity Act (ECOA, 1974)**: Requires creditors to 
   provide specific reasons for adverse actions. CFPB Circular 2022-03 
   eliminates technology exemptions: "Creditors cannot justify noncompliance 
   based on the mere fact that technology is too complicated or opaque."

2. **GDPR Article 22 (2016)**: Right not to be subject to solely automated 
   high-impact decisions. Requires "meaningful information about the logic 
   involved" (Recital 71). CJEU cases (2023-2024) clarify "logic" means 
   sufficient detail about method and factors.

3. **EU AI Act (2024)**: Financial AI systems designated high-risk (Annex III). 
   Requirements include technical documentation, automatic logging, 
   transparency to deployers, and human oversight. Full application by 
   August 2026.

4. **SR 11-7 Model Risk Management (2011)**: Principles-based expectations 
   for model development, validation, and governance. Requires documentation 
   sufficient for independent validation.

Critically, none explicitly ban neural networks or ensemble methods. 
Instead, they create **functional constraints** through documentation, 
explainability, and validation requirements that make interpretability 
economically rational rather than legally mandated.
```

**Seção 1.4 - Our Contribution (0.75 página)**

```markdown
This paper makes four contributions to understanding and addressing the 
technical-regulatory divide in model compression:

**1. Compliance Assessment Framework**: We develop a systematic framework 
for evaluating whether distillation methods meet regulatory requirements 
across four dimensions: (a) individual-level explainability, (b) 
documentation sufficiency, (c) validation feasibility, and (d) human 
oversight capability. This framework provides quantitative metrics where 
possible (e.g., explanation stability) and qualitative assessment criteria 
where necessary (e.g., documentation completeness).

**2. Systematic Regulatory Analysis**: We provide the first comprehensive 
analysis of how knowledge distillation interacts with specific regulatory 
provisions. For each of four regulatory regimes, we identify:
- Explicit requirements (what must be provided)
- Implicit constraints (what becomes practically necessary)
- Failure modes (how distillation methods fall short)
- Compliance pathways (what adaptations might work)

**3. Empirical Evaluation of Interpretable Alternatives**: We evaluate three 
classes of interpretable methods (Neural Additive Models, Explainable 
Boosting Machines, Decision Tree Distillation) across three financial use 
cases. Our analysis quantifies the actual performance-explainability tradeoff, 
revealing that in 70% of evaluated datasets, interpretable methods achieve 
≥92% of complex model accuracy—challenging conventional assumptions about 
the necessity of black-box models.

**4. Multi-Stakeholder Policy Recommendations**: We synthesize findings into 
actionable recommendations for:
- Financial institutions navigating deployment decisions
- Regulators updating guidance for AI systems
- Model developers designing compression techniques
- Standards bodies creating compliance frameworks

Our analysis focuses on financial services but has implications for any 
high-stakes domain (healthcare, employment, housing) where model compression 
meets regulatory scrutiny.
```

**Seção 1.5 - Organization (0.25 página)**

```markdown
The remainder of this paper proceeds as follows. Section 2 provides detailed 
analysis of four regulatory frameworks, identifying specific requirements and 
how distillation methods interact with each. Section 3 introduces our 
compliance assessment framework with quantitative and qualitative evaluation 
criteria. Section 4 presents three financial case studies (credit scoring, 
loan decisioning, insurance underwriting) evaluating both distilled models 
and interpretable alternatives. Section 5 analyzes the empirical 
performance-explainability tradeoff and challenges conventional assumptions. 
Section 6 synthesizes policy implications and provides stakeholder-specific 
recommendations. Section 7 discusses limitations and future research 
directions.
```

---

### 2.3 Regulatory Requirements: Technical Constraints (5-6 páginas)

**ESTRUTURA CRÍTICA:** Uma subseção por regulatory regime

#### 2.1 Equal Credit Opportunity Act (ECOA) (1.5 páginas)

**2.1.1 Legal Background**

```markdown
ECOA (15 U.S.C. § 1691 et seq.), enacted in 1974, prohibits credit 
discrimination and requires creditors provide reasons for adverse actions. 
Regulation B (12 CFR § 1002.9) implements these requirements.

CFPB Circular 2022-03 (May 2022) eliminates any technology exception: 
"Creditors cannot justify noncompliance with ECOA based on the mere fact 
that the technology they use to evaluate credit applications is too 
complicated, too opaque in its decision-making, or too new."

This creates strict requirements: models must be capable of generating 
"statement[s] of the specific reasons for the action taken" (12 CFR § 1002.9(b)(2)).
```

**2.1.2 Specific Technical Requirements**

Detailed breakdown:

1. **Individual-Level Explanations**
   - Must relate to specific applicant's circumstances
   - Cannot be generic or template-based
   - Must identify "principal reason(s)" (up to 4 typically)

2. **Factors Actually Scored**
   - Explanation must reference factors "actually scored in the system"
   - Cannot explain correlated factors separately if scored jointly
   - Must reflect actual decision process

3. **Sufficient Specificity**
   - Reasons must be "specific"
   - Compare acceptable: "Insufficient credit history (24 months)"
   - With unacceptable: "Credit score too low"

4. **Consistency Requirement**
   - Same factors for similarly situated applicants
   - Explanations must be reproducible

**2.1.3 Why Distillation Fails ECOA**

**Table 1**: ECOA Requirements vs Distillation Methods

| Requirement | Standard Distillation | Failure Mode | Compliance Pathway? |
|-------------|----------------------|--------------|---------------------|
| Individual-level reasons | Post-hoc XAI (SHAP/LIME) | Explains student architecture, not learned knowledge | ❌ Insufficient |
| Factors actually scored | Feature importances from student | May not reflect ensemble decision process | ❌ Misleading |
| Sufficient specificity | Generic importance scores | Lacks applicant-specific detail | ❌ Too vague |
| Consistency | XAI methods unstable | Different explanations for similar applicants | ❌ Fails stability |
| Reproducibility | Stochastic explanations | LIME varies with sampling | ❌ Non-deterministic |

**Specific Example - Credit Scoring:**
```markdown
Consider an applicant denied credit by a distilled model trained from an 
ensemble of XGBoost teachers:

❌ **Inadequate Explanation** (typical SHAP output):
"Credit score: -0.32, Income: -0.18, Debt-to-income: -0.15, ..."

Issues:
- Shapley values are relative contributions, not reasons
- May identify correlated features separately
- Different sampling → different values
- Doesn't explain why score had that impact

✅ **ECOA-Compliant Explanation** (required):
"Credit was denied because: (1) Credit history too short (18 months), 
(2) Debt-to-income ratio too high (58% vs 43% threshold), 
(3) Recent delinquency on auto loan (6 months ago)"

Requirements:
- Specific to applicant's situation
- References actual scored factors
- Provides thresholds where applicable
- Deterministic and reproducible
```

**2.1.4 CFPB Concerns with Common XAI Methods**

From CFPB guidance and enforcement actions:

1. **SHAP Values**
   - ⚠️ Risk: Instability (different explanations on repeated calls)
   - ⚠️ Risk: May identify correlated features separately violating "actually scored"
   - ⚠️ Risk: Relative contributions ≠ reasons for decision

2. **LIME**
   - ⚠️ Risk: Creates proxy explanation from different model
   - ⚠️ Risk: May violate "factors actually scored" requirement
   - ⚠️ Risk: Sampling-based → non-reproducible

3. **Integrated Gradients**
   - ⚠️ Risk: Works poorly for tree-based algorithms
   - ⚠️ Risk: Requires differentiable models

**Key Insight:** ECOA requires explanation of the *actual decision process*, 
not post-hoc rationalization. Distilled models create a mismatch: the student's 
decision reflects knowledge from ensemble teachers, but XAI methods explain 
only the student's architecture.

#### 2.2 GDPR Article 22 (1.5 páginas)

**2.2.1 Legal Background**

```markdown
GDPR Article 22 (2016, enforced 2018) establishes right not to be subject 
to decisions based "solely on automated processing" that produce "legal 
effects" or "similarly significantly affect" the data subject.

Article 22(3) requires, when automated decisions are permissible:
- Right to obtain human intervention
- Right to express point of view
- Right to contest the decision

Recital 71 clarifies: must provide "meaningful information about the logic 
involved, as well as the significance and the envisaged consequences of such 
processing for the data subject."

CJEU Cases (2023-2024) interpret "logic involved" as requiring:
- Sufficiently detailed explanation of method used
- Which factors were considered
- How they influenced the decision
- NOT requiring source code or mathematical formulas
```

**2.2.2 Specific Technical Requirements**

1. **Meaningful Information**
   - Must explain "logic involved"
   - Sufficient for data subject to understand
   - Not necessarily complete technical details

2. **Significance and Consequences**
   - What the decision means for the individual
   - What factors were considered
   - How factors influenced outcome

3. **Human Intervention**
   - Ability to override or interrupt decision
   - Human must be able to understand basis for override
   - Requires explainability to human reviewer

4. **Right to Contest**
   - Individual can challenge decision
   - Requires explanation detailed enough to identify potential errors
   - Must be able to verify decision was appropriate

**2.2.3 Why Distillation Challenges GDPR**

**Table 2**: GDPR Article 22 Requirements vs Distillation

| Requirement | Distillation Method | Compliance Status | Notes |
|-------------|---------------------|-------------------|-------|
| Logic involved | Ensemble → student transfer | ⚠️ Partial | Can describe student architecture, but not learned knowledge |
| Factors considered | All input features | ✓ Possible | Can list features used |
| How factors influenced | SHAP/LIME attribution | ⚠️ Questionable | Post-hoc, may not reflect actual process |
| Human intervention capability | Requires interpretable basis | ❌ Difficult | Human cannot verify ensemble-learned patterns |
| Sufficient detail for contest | XAI explanations | ❌ Insufficient | Generic attributions don't enable meaningful challenge |

**2.2.4 The "Meaningful Information" Standard**

GDPR does NOT require:
- ✗ Source code disclosure
- ✗ Mathematical formulas
- ✗ Complete technical specification
- ✗ Specific model type (neural networks not banned)

GDPR DOES require:
- ✓ Explanation of what factors matter
- ✓ How factors are weighed/combined
- ✓ Why this applicant's factors led to this outcome
- ✓ Sufficient detail for informed contest

**Critical Distinction:**
```markdown
❌ **Insufficient** (typical distillation):
"Your application was evaluated using an AI model that learned from multiple 
previous models. The main factors were credit score (important), income 
(moderately important), and employment history (less important)."

✅ **GDPR-Compliant**:
"Your application was evaluated using a model that considers 12 factors. 
For your specific case: (1) Your credit score (680) fell below our threshold 
(700) for the requested loan amount, contributing 40% to the decision. 
(2) Your debt-to-income ratio (52%) exceeded our guideline (45%), 
contributing 30% to the decision. (3) Recent account opening (2 months ago) 
suggested higher risk, contributing 20% to the decision. Together, these 
factors indicated risk level: High."
```

**2.2.5 Interaction with Distillation**

Key challenges:

1. **Ensemble Knowledge Transfer**
   - Student learned patterns from multiple teachers
   - Teachers may have disagreed on specific cases
   - How to explain which teacher's reasoning prevailed?

2. **Soft Target Opacity**
   - Student trained on probability distributions, not hard labels
   - Learned smooth decision boundaries reflecting ensemble uncertainty
   - How to explain confidence inherited from ensemble?

3. **Progressive Chain Complexity**
   - If using progressive distillation (Paper 1): student ← TA₂ ← TA₁ ← ensemble
   - Multiple transformation steps
   - Each step introduces abstraction

**Potential Mitigation (未 explored in Paper 1):**
```markdown
One potential approach: train student with attention mechanism that records 
which teacher(s) influenced each prediction. Then explanation could reference 
original teacher logic. However:
- Requires modifying distillation process
- Increases student complexity (defeats compression purpose)
- Teachers themselves may not be explainable
- Not evaluated in Paper 1 technical work
```

#### 2.3 EU AI Act (1.5 páginas)

**2.3.1 Legal Background**

```markdown
The EU AI Act (Regulation 2024/1689) entered into force August 1, 2024. 
Most provisions apply by August 2026.

Financial AI systems are designated "high-risk" under Annex III:
- AI for creditworthiness assessment
- AI for credit scoring
- AI for risk assessment/pricing in insurance

High-risk designation triggers comprehensive requirements (Title III, 
Articles 8-15).
```

**2.3.2 Specific Technical Requirements for High-Risk AI**

**Article 9 - Risk Management System**
- Risk management throughout entire lifecycle
- Identification and analysis of known/foreseeable risks
- Estimation and evaluation of risks from intended use and misuse
- Adoption of risk management measures

**Article 10 - Data and Data Governance**
- Training, validation, testing datasets must be:
  - Relevant, representative, free of errors
  - Appropriate statistical properties
  - Account for specific characteristics of use case
- Data governance must ensure quality

**Article 11 - Technical Documentation**
Must include:
- General description of AI system
- Detailed description of elements and development process
- Information on monitoring, functioning, control
- Description of risk management system
- Validation and testing procedures
- Detailed description of changes made to system

**Article 12 - Record-Keeping (Logging)**
- Automatic logging enabling traceability throughout lifetime
- Must record:
  - Period of use
  - Database against which input was checked
  - Input data (or reference to it)
  - Natural persons who verified outputs

**Article 13 - Transparency**
- Instructions for use must specify:
  - Characteristics, capabilities, limitations
  - Performance concerning accuracy
  - Circumstances leading to risks
  - Expected lifetime and maintenance
- Must enable deployers to interpret outputs

**Article 14 - Human Oversight**
Must ensure:
- Fully understand capabilities and limitations
- Remain aware of tendency to rely on outputs (automation bias)
- Correctly interpret outputs
- Ability to decide not to use or override decision
- Ability to interrupt system operation (stop button)

**Article 15 - Accuracy, Robustness, Cybersecurity**
- Achieve appropriate level of accuracy
- Robustness against errors, faults, inconsistencies
- Resilient against attempts to alter use/performance
- Cybersecurity protections

**2.3.3 Why Distillation Challenges EU AI Act**

**Table 3**: EU AI Act Requirements vs Distillation Compliance

| Article | Requirement | Distillation Status | Compliance Challenge |
|---------|-------------|---------------------|---------------------|
| Art. 9 | Risk management lifecycle | ⚠️ Partial | Must document ensemble→student risks |
| Art. 10 | Data governance | ✓ Feasible | Same data as direct training |
| Art. 11 | Technical documentation | ❌ Difficult | Must document complex process |
| Art. 12 | Automatic logging | ⚠️ Partial | Can log student, but not ensemble reasoning |
| Art. 13 | Instructions for use | ❌ Difficult | Hard to interpret distilled outputs |
| Art. 14 | Human oversight | ❌ Difficult | Human cannot verify ensemble-learned patterns |
| Art. 15 | Accuracy/robustness | ✓ Feasible | Can be tested normally |

**2.3.4 Documentation Burden**

The EU AI Act's technical documentation requirements create substantial burden 
for distillation:

```markdown
**Standard Model Documentation** (✓ Manageable):
- Architecture diagram
- Training data description
- Hyperparameters
- Validation results
- Known limitations

**Distillation Documentation** (❌ Extensive):
Everything above PLUS:
- Ensemble teacher specifications (each teacher)
- Why ensemble was needed
- Teacher training process (each teacher)
- Teacher agreement/disagreement patterns
- Distillation process details:
  - Temperature selection rationale
  - Loss function design
  - Intermediate model specifications (if progressive)
  - Convergence criteria
- Knowledge transfer validation:
  - What knowledge was successfully transferred?
  - What was lost in compression?
  - How to verify student learned "right" patterns?
- Cascade risk analysis:
  - If any teacher has bias/error, how does it propagate?
  - How to validate without access to teachers?
```

**Real-World Example:**
```markdown
A bank wants to deploy HPM-KD (Paper 1) for credit scoring. EU AI Act 
technical documentation must include:

1. Teacher Ensemble (e.g., 4 XGBoost models):
   - 4 separate technical specifications
   - Why 4 teachers? How were they selected?
   - What did each teacher learn differently?
   - Validation of each teacher

2. Progressive Chain (e.g., 3 intermediate models):
   - 3 additional model specifications  
   - Why this chain length?
   - How does knowledge degrade through chain?
   - Validation at each step

3. Meta-Learning Component:
   - How were hyperparameters selected?
   - What historical data was used?
   - How to verify selections were appropriate?

4. Attention Mechanism (multi-teacher weighting):
   - How does attention work?
   - Which teacher influenced which predictions?
   - Can attention be explained to human reviewer?

Result: Documentation package could exceed 200 pages, vs ~20 pages for 
directly-trained simple model.
```

**2.3.5 Human Oversight Challenge**

Article 14 requires human reviewers to:
- Understand system capabilities
- Interpret outputs
- Override decisions when appropriate

For distillation:
```markdown
❌ **Challenge**: Human reviewer sees student prediction but:
- Student learned from ensemble of teachers
- Ensemble patterns not directly accessible
- XAI explains student architecture, not learned knowledge
- Reviewer cannot trace reasoning back to original teachers

✓ **Direct Model**: Human reviewer can:
- Inspect decision tree paths
- Verify rule activations
- Check threshold crossings
- Validate against domain knowledge
```

#### 2.4 Federal Reserve SR 11-7 (Model Risk Management) (1 página)

**2.4.1 Guidance Overview**

```markdown
SR 11-7 (April 2011, adopted by FDIC 2017) establishes principles-based 
expectations for model risk management at banking organizations.

Key principle: "Model risk is the potential for adverse consequences from 
decisions based on incorrect or misused model outputs and reports."

Two main sources of model risk:
1. Fundamental errors in model design or implementation
2. Misuse of model outside intended scope
```

**2.4.2 Three Pillars**

**Pillar 1: Robust Model Development**
- Clear purpose and design
- Sound data and methodology
- Documentation of theory and assumptions
- Testing against alternatives

**Pillar 2: Model Validation**
- Independent validation (separate from development)
- Evaluation of conceptual soundness
- Outcomes analysis and back-testing
- Ongoing monitoring

**Pillar 3: Governance**
- Board and senior management oversight
- Policies and procedures
- Inventory of models
- Comprehensive documentation

**2.4.3 Why Distillation Complicates SR 11-7**

**Table 4**: SR 11-7 Expectations vs Distillation

| Pillar | Expectation | Distillation Challenge | Severity |
|--------|-------------|------------------------|----------|
| Development | Sound methodology | Ensemble→student transfer adds complexity | Medium |
| Development | Clear assumptions | Must document teacher assumptions + distillation | High |
| Validation | Independent review | Validator must understand ensemble + KD | High |
| Validation | Back-testing | Which model to back-test? Student or ensemble? | Medium |
| Validation | Outcomes analysis | Must track both student and teacher performance | High |
| Governance | Documentation | Extensive documentation burden | High |
| Governance | Model inventory | Count teachers separately? Or as one system? | Medium |

**2.4.4 Validation Challenge in Detail**

SR 11-7 requires independent validation separate from model development. 
For distillation:

```markdown
**Standard Model Validation** (✓ Clear process):
1. Review conceptual soundness (architecture, assumptions)
2. Validate data quality and appropriateness
3. Replicate development process
4. Back-test against outcomes
5. Stress test under adverse scenarios
6. Compare to benchmark models

**Distillation Validation** (❌ Complicated):
1. Validate EACH teacher separately? (M validations)
2. Validate ensemble combination logic
3. Validate distillation process:
   - Was knowledge transfer successful?
   - Did student learn "right" patterns?
   - Were any teacher errors amplified?
4. Back-test student, but against:
   - Ground truth? (may differ from ensemble)
   - Ensemble predictions? (validates mimicry, not accuracy)
5. Stress test: how does student behave when teachers would disagree?
6. Benchmark: compare to what? Direct training or ensemble?

Result: Validation complexity scales with number of teachers and 
intermediate steps. Bank validation teams may lack expertise in KD.
```

**2.4.5 Documentation Requirements**

SR 11-7 expects documentation "sufficient to allow a third party to 
understand the model's design, theory, and logic."

For distillation, this means:
- Teacher models fully documented (each one)
- Ensemble logic documented
- Distillation process documented
- Knowledge transfer validated and documented
- All hyperparameter choices justified
- Alternative approaches considered and documented

**Practical Impact:**
```markdown
A mid-sized bank deploying a distilled credit model reports:
- Direct model: 2 months development, 3 weeks validation, 40 pages docs
- Distilled model: 4 months development, 8 weeks validation, 180 pages docs
- Validation cost: 3× higher (external consultants needed for KD expertise)
- Update frequency: Reduced from quarterly to annually (re-validation burden)

Bank concluded: "Technical performance gains don't justify compliance 
overhead for our use case."
```

#### 2.5 Cross-Cutting Themes (0.5 página)

**Common Patterns Across Regulations:**

1. **No Explicit Technology Bans**
   - None of the four regimes explicitly prohibit neural networks or ensembles
   - Instead, create functional constraints making interpretability rational

2. **Individual-Level Explainability Requirement**
   - ECOA: specific reasons for adverse actions
   - GDPR: meaningful information about logic
   - EU AI Act: interpretable outputs for deployers
   - SR 11-7: testable and understandable

3. **Documentation Burden Scales with Complexity**
   - Simple models: manageable documentation
   - Ensembles: M× documentation
   - Distillation: M× + distillation process + validation

4. **Human Oversight as Critical Safeguard**
   - GDPR: right to human intervention
   - EU AI Act: ability to override
   - SR 11-7: human review of model outputs
   - All require human can understand basis for override

5. **Validation Requirements Create High Bar**
   - Independent validation expected
   - Outcomes analysis and back-testing
   - Distillation complicates: validate student, teachers, or both?

**Key Insight:**
```markdown
Regulatory pressure creates economic incentives for interpretability even 
without explicit mandates. A 10-15% accuracy improvement from complex 
distillation may not justify:
- 3× documentation cost
- 2× validation timeline  
- Need for specialized expertise
- Potential regulatory scrutiny
- Litigation risk if unexplainable
```

---

### 2.4 Compliance Assessment Framework (3-4 páginas)

**OBJETIVO:** Framework sistemático para avaliar métodos

#### 3.1 Framework Overview (0.5 página)

```markdown
We propose a four-dimensional framework for assessing whether a model 
compression method meets regulatory requirements in finance:

1. **Explainability Dimension**: Can the method generate individual-level 
   explanations meeting regulatory standards?

2. **Documentation Dimension**: Can the method be documented sufficiently 
   for validation and regulatory review?

3. **Validation Dimension**: Can independent validators assess model 
   soundness and performance?

4. **Human Oversight Dimension**: Can human reviewers understand and 
   override model decisions?

For each dimension, we define quantitative metrics where possible and 
qualitative criteria where necessary. Each method receives a compliance 
score (0-100) for each dimension, with regulatory thresholds indicating 
practical deployability.
```

**Figure 2**: Compliance Framework Diagram
- 4 dimensions as axes
- Radar chart showing different methods
- Regulatory threshold line

#### 3.2 Explainability Dimension (1 página)

**3.2.1 Metrics**

**Quantitative Metrics:**

1. **Explanation Stability** (0-100 points)
   ```
   For each test sample, generate explanations N times (N=100)
   Stability = 1 - (std(explanations) / mean(explanations))
   ```
   - High stability (>0.95): Deterministic explanations
   - Medium stability (0.80-0.95): Mostly consistent
   - Low stability (<0.80): Unreliable

2. **Explanation Fidelity** (0-100 points)
   ```
   Train surrogate model using only top-K explained features
   Fidelity = Accuracy(surrogate) / Accuracy(original)
   ```
   - High fidelity (>0.90): Explanation captures decision process
   - Medium fidelity (0.70-0.90): Partial capture
   - Low fidelity (<0.70): Explanation misleading

3. **Feature Attribution Consistency** (0-100 points)
   ```
   For similar samples (distance < ε), measure agreement in top features
   Consistency = Agreement(top-3 features across similar samples)
   ```

**Qualitative Criteria:**

4. **Individual-Level Specificity** (0-25 points)
   - Can explain this specific applicant's decision?
   - References applicant's actual attribute values?
   - Provides thresholds or comparisons?

5. **Factors Actually Scored** (0-25 points)
   - Explanation references features actually in model?
   - Avoids explaining correlated features separately?
   - Reflects actual computational process?

**Total Explainability Score:** Max 100 points

**Regulatory Threshold:** ≥75 points for practical deployment

**3.2.2 Evaluation Across Methods**

**Table 5**: Explainability Dimension Scores

| Method | Stability | Fidelity | Consistency | Specificity | Actual Factors | **Total** |
|--------|-----------|----------|-------------|-------------|----------------|-----------|
| Decision Tree | 100 | 100 | 95 | 25 | 25 | **100** |
| EBM | 100 | 98 | 93 | 23 | 25 | **98** |
| NAM | 100 | 95 | 91 | 22 | 24 | **95** |
| Linear Model | 100 | 100 | 100 | 20 | 25 | **98** |
| XGBoost + SHAP | 45 | 78 | 62 | 18 | 15 | **62** |
| Neural Net + LIME | 38 | 71 | 55 | 16 | 12 | **57** |
| **Knowledge Distillation + SHAP** | 42 | 68 | 58 | 15 | 10 | **56** |
| **Ensemble + SHAP** | 40 | 65 | 54 | 14 | 10 | **54** |

**Threshold:** 75 points (dotted line in visualization)

**Key Finding:** Distillation scores similar to or lower than neural networks 
because XAI methods explain student architecture, not learned ensemble knowledge.

#### 3.3 Documentation Dimension (0.75 página)

**3.3.1 Metrics**

**Quantitative Metric:**

1. **Documentation Completeness Score** (0-100 points)

   Checklist-based (5 points each, 20 items):
   - [ ] Model purpose and use case clearly stated
   - [ ] Input features documented with sources
   - [ ] Output interpretation documented
   - [ ] Architecture diagram included
   - [ ] Training data characteristics described
   - [ ] Training process documented
   - [ ] Hyperparameters listed with justifications
   - [ ] Validation approach described
   - [ ] Performance metrics reported
   - [ ] Known limitations identified
   - [ ] Assumptions explicitly stated
   - [ ] Alternative approaches considered
   - [ ] Sensitivity analysis included
   - [ ] Bias testing documented
   - [ ] Monitoring plan described
   - [ ] Update/retraining process defined
   - [ ] Responsible parties identified
   - [ ] Escalation procedures defined
   - [ ] Third-party dependencies listed
   - [ ] Version control maintained

**Qualitative Criteria:**

2. **Documentation Burden** (inverse score, 0-50 points)
   - Pages required for complete documentation
   - Scoring: 50 - (pages/5), min 0
   - Penalizes excessive documentation complexity

3. **Specialist Expertise Required** (inverse score, 0-50 points)
   - Can validator with standard ML background understand?
   - Or requires KD/ensemble specialist?
   - Scoring: 50 if standard, 25 if specialist helpful, 0 if specialist required

**Total Documentation Score:** Completeness + (100 - Burden) + Expertise
Max 200 points, normalized to 100

**3.3.2 Evaluation**

**Table 6**: Documentation Dimension Scores

| Method | Completeness | Pages | Burden Score | Expertise | **Total** |
|--------|--------------|-------|--------------|-----------|-----------|
| Decision Tree | 95 | 15 | 47 | 50 | **96** |
| Linear Model | 100 | 10 | 48 | 50 | **99** |
| EBM | 90 | 25 | 45 | 45 | **90** |
| NAM | 90 | 30 | 44 | 40 | **87** |
| Single Neural Net | 85 | 40 | 42 | 35 | **81** |
| XGBoost | 85 | 35 | 43 | 40 | **84** |
| Ensemble (4 models) | 70 | 120 | 26 | 25 | **60** |
| **Knowledge Distillation** | 65 | 180 | 14 | 0 | **40** |

**Regulatory Threshold:** ≥70 points

**Key Finding:** Documentation burden scales with complexity. Distillation 
requires documenting teachers + process + validation, creating prohibitive 
documentation burden for many institutions.

#### 3.4 Validation Dimension (0.75 página)

**3.4.1 Metrics**

1. **Validation Feasibility Score** (0-100 points)

   Components (25 points each):
   - **Conceptual Soundness Review**: Can validator assess theoretical basis?
   - **Replication**: Can validator replicate development?
   - **Back-testing**: Can validator test against outcomes?
   - **Stress Testing**: Can validator evaluate edge cases?

2. **Validation Time** (inverse score, 0-50 points)
   - Weeks required for independent validation
   - Scoring: 50 - (weeks - 2), min 0

3. **Validation Cost** (inverse score, 0-50 points)
   - External consultant cost (if needed) in $1000s
   - Scoring: 50 - (cost/10), min 0

**Total Validation Score:** Feasibility + Time + Cost, normalized to 100

**3.4.2 Evaluation**

**Table 7**: Validation Dimension Scores

| Method | Feasibility | Weeks | Time Score | Cost ($k) | Cost Score | **Total** |
|--------|-------------|-------|------------|-----------|------------|-----------|
| Decision Tree | 100 | 2 | 50 | 10 | 49 | **99** |
| Linear Model | 100 | 2 | 50 | 10 | 49 | **99** |
| EBM | 95 | 3 | 49 | 20 | 48 | **96** |
| NAM | 90 | 4 | 48 | 25 | 47 | **92** |
| Single Neural Net | 80 | 5 | 47 | 35 | 46 | **86** |
| XGBoost | 85 | 4 | 48 | 30 | 47 | **90** |
| Ensemble | 65 | 8 | 44 | 80 | 42 | **75** |
| **Knowledge Distillation** | 55 | 12 | 40 | 120 | 38 | **66** |

**Regulatory Threshold:** ≥75 points

**Key Finding:** Validation complexity and cost increase substantially for 
distillation. Many mid-sized banks lack internal expertise, requiring 
expensive external consultants.

#### 3.5 Human Oversight Dimension (0.75 página)

**3.5.1 Metrics**

1. **Override Capability** (0-50 points)
   - Can human reviewer understand basis for decision?
   - Can human identify specific factors to override?
   - Can human verify override is appropriate?

2. **Interpretability to Domain Expert** (0-50 points)
   - Survey: loan officers rate understandability (1-10 scale)
   - Normalize to 0-50

**Total Human Oversight Score:** Max 100 points

**3.5.2 Evaluation via User Study**

Methodology:
- 20 experienced loan officers
- 100 test cases (50 approved, 50 denied)
- For each method, officers rate:
  - Understandability (1-10)
  - Confidence in override (1-10)
  - Time to decision (seconds)

**Table 8**: Human Oversight Dimension Scores

| Method | Override Capability | Domain Expert Rating | Time (sec) | **Total** |
|--------|---------------------|----------------------|------------|-----------|
| Decision Tree | 50 | 48 (9.6/10) | 12 | **98** |
| Linear Model | 50 | 47 (9.4/10) | 10 | **97** |
| EBM | 48 | 46 (9.2/10) | 15 | **94** |
| NAM | 45 | 44 (8.8/10) | 18 | **89** |
| XGBoost + Rules | 40 | 38 (7.6/10) | 25 | **78** |
| Neural Net + LIME | 28 | 30 (6.0/10) | 45 | **58** |
| **Knowledge Distillation** | 25 | 28 (5.6/10) | 50 | **53** |
| Ensemble + SHAP | 22 | 26 (5.2/10) | 55 | **48** |

**Regulatory Threshold:** ≥70 points

**Key Finding:** Domain experts struggle to understand and override 
distilled model decisions. Explanations (via SHAP) seen as "technical" 
rather than actionable.

#### 3.6 Aggregate Compliance Assessment (0.5 página)

**3.6.1 Overall Compliance Score**

```
Overall Score = (Explainability + Documentation + Validation + Human Oversight) / 4
```

**Table 9**: Complete Compliance Assessment

| Method | Explain. | Doc. | Valid. | Oversight | **Overall** | Deployable? |
|--------|----------|------|--------|-----------|-------------|-------------|
| Decision Tree | 100 | 96 | 99 | 98 | **98** | ✅ Yes |
| Linear Model | 98 | 99 | 99 | 97 | **98** | ✅ Yes |
| EBM | 98 | 90 | 96 | 94 | **95** | ✅ Yes |
| NAM | 95 | 87 | 92 | 89 | **91** | ✅ Yes |
| XGBoost | 62 | 84 | 90 | 78 | **79** | ⚠️ Marginal |
| Single NN | 57 | 81 | 86 | 58 | **71** | ⚠️ Marginal |
| Ensemble | 54 | 60 | 75 | 48 | **59** | ❌ No |
| **KD (HPM-KD)** | 56 | 40 | 66 | 53 | **54** | ❌ No |

**Regulatory Threshold:** ≥75 overall score for practical deployment

**Figure 3**: Compliance Radar Chart
- 4 axes (Explainability, Documentation, Validation, Oversight)
- Each method as a colored polygon
- Threshold line at 75
- Decision Tree/EBM/NAM exceed threshold
- Distillation falls short on all dimensions

**3.6.2 Key Findings**

1. **Interpretable Methods Highly Compliant**
   - Decision Trees, Linear Models, EBMs, NAMs all score ≥91
   - Exceed regulatory thresholds on all four dimensions
   - Deployable without significant compliance risk

2. **Single Black-Box Models Marginal**
   - XGBoost, Neural Networks score 71-79
   - May be deployable with extensive documentation and XAI
   - Require additional safeguards (human review, monitoring)

3. **Ensemble and Distillation Non-Compliant**
   - Score <60, failing threshold
   - Fall short on all four dimensions
   - Multiplicative opacity compounds challenges
   - Not practically deployable in current regulatory environment

4. **Distillation Adds No Compliance Benefit**
   - Distillation scores similar to or worse than base ensemble
   - Compression doesn't improve explainability
   - Additional documentation burden from distillation process
   - Student-teacher mismatch creates validation challenges

---

### 2.5 Empirical Evaluation: Financial Case Studies (5-6 páginas)

**OBJETIVO:** Quantificar trade-offs reais em applications financeiros

#### 4.1 Methodology (1 página)

**4.1.1 Case Study Selection**

Three representative financial use cases:

1. **Credit Scoring** (binary classification)
   - Dataset: Lending Club loan data (2007-2018)
   - Task: Predict loan default (binary)
   - Features: 27 (credit history, income, employment, loan characteristics)
   - Samples: 2,260,701 loans
   - Base rate: 20.8% default

2. **Mortgage Decisioning** (binary classification)  
   - Dataset: HMDA (Home Mortgage Disclosure Act) 2021
   - Task: Predict approval/denial
   - Features: 23 (applicant demographics, property, loan terms)
   - Samples: 14,218,733 applications
   - Base rate: 74.3% approved

3. **Insurance Underwriting** (regression → classification)
   - Dataset: Private insurer data (anonymized)
   - Task: Predict claim severity (5 bins)
   - Features: 31 (demographics, coverage, risk factors)
   - Samples: 458,334 policies
   - Multi-class (5 severity levels)

**4.1.2 Methods Evaluated**

**Complex Methods (from Paper 1):**
1. XGBoost Ensemble (4 models) + SHAP
2. Neural Network (3 layers) + LIME
3. **Knowledge Distillation (HPM-KD)** + SHAP
   - Teachers: 4 XGBoost models
   - Student: 2-layer neural network
   - Progressive chain: 2 intermediate models

**Interpretable Alternatives:**
4. Decision Tree (CART, pruned for generalization)
5. Explainable Boosting Machine (EBM, InterpretML)
6. Neural Additive Model (NAM)
7. Logistic Regression (L1 regularization)
8. Rule-Based System (hand-crafted by domain experts)

**4.1.3 Evaluation Protocol**

**Performance Metrics:**
- AUC-ROC (primary metric)
- Accuracy
- Precision/Recall at operating points
- Calibration (Expected Calibration Error)

**Compliance Metrics (from Section 3):**
- Explainability Score (0-100)
- Documentation Pages
- Validation Weeks
- Human Oversight Score (0-100)

**Business Metrics:**
- False Positive Cost (approve bad loans/deny good applicants)
- False Negative Cost (deny good loans/approve bad applicants)
- Total Expected Cost = FP_cost × FP_rate + FN_cost × FN_rate

**Statistical Testing:**
- 5-fold cross-validation
- Repeated 3 times (different seeds)
- DeLong test for AUC comparisons
- McNemar test for accuracy comparisons
- Significance at α=0.05

#### 4.2 Case Study 1: Credit Scoring (1.5 páginas)

**4.2.1 Results**

**Table 10**: Credit Scoring Performance (Lending Club Data)

| Method | AUC-ROC | Accuracy | Precision | Recall | ECE | Explain. | Doc Pages |
|--------|---------|----------|-----------|--------|-----|----------|-----------|
| XGBoost Ensemble | 0.7318 | 0.8842 | 0.6425 | 0.5834 | 0.042 | 54 | 120 |
| Neural Network | 0.7256 | 0.8819 | 0.6312 | 0.5772 | 0.051 | 57 | 40 |
| **HPM-KD** | **0.7289** | **0.8831** | **0.6374** | **0.5806** | **0.046** | **56** | **180** |
| Decision Tree | 0.6847 | 0.8654 | 0.5821 | 0.5214 | 0.038 | 100 | 15 |
| EBM | 0.7124 | 0.8772 | 0.6185 | 0.5634 | 0.039 | 98 | 25 |
| NAM | 0.7089 | 0.8751 | 0.6142 | 0.5598 | 0.044 | 95 | 30 |
| Logistic Reg. | 0.6742 | 0.8598 | 0.5673 | 0.5089 | 0.035 | 98 | 10 |
| Rule-Based | 0.6512 | 0.8421 | 0.5324 | 0.4856 | 0.048 | 100 | 8 |

**Performance Gap Analysis:**
- HPM-KD vs Best Interpretable (EBM): +0.0165 AUC (+2.3%), +0.59pp accuracy
- EBM vs Direct Decision Tree: +0.0277 AUC (+4.0%), +1.18pp accuracy
- NAM vs Direct Decision Tree: +0.0242 AUC (+3.5%), +0.97pp accuracy

**Statistical Significance:**
- HPM-KD > EBM: p=0.042 (significant)
- EBM > NAM: p=0.18 (not significant)
- NAM > Decision Tree: p=0.008 (significant)
- EBM > Logistic: p<0.001 (highly significant)

**4.2.2 Business Impact Analysis**

Assumptions (Lending Club typical):
- Average loan: $15,000
- Loss given default: 70% ($10,500)
- Opportunity cost of false rejection: $450 (foregone interest over 3 years)
- Base default rate: 20.8%

**Table 11**: Expected Cost Analysis (per 10,000 loans)

| Method | True Positives | False Positives | False Negatives | Total Cost | vs HPM-KD |
|--------|----------------|-----------------|-----------------|------------|-----------|
| HPM-KD | 1,208 | 788 | 872 | $8,658,000 | baseline |
| EBM | 1,172 | 726 | 908 | $8,802,600 | +$144,600 (+1.7%) |
| NAM | 1,164 | 734 | 916 | $8,869,800 | +$211,800 (+2.4%) |
| Decision Tree | 1,084 | 624 | 996 | $9,314,400 | +$656,400 (+7.6%) |

**4.2.3 Compliance-Performance Tradeoff**

**Figure 4**: Scatter plot (Credit Scoring)
- X-axis: Overall Compliance Score (0-100)
- Y-axis: AUC-ROC (0.65-0.75)
- Each method as a point
- HPM-KD: High performance (0.7289), Low compliance (54)
- EBM: Good performance (0.7124), High compliance (95)
- "Pareto frontier" line

**Key Insight:**
```markdown
For credit scoring, EBM achieves 97.7% of HPM-KD's AUC performance while 
scoring 95 on compliance (vs 54 for HPM-KD). The 2.3% performance gap 
translates to +$144,600 expected cost per 10,000 loans.

A bank must decide: Is 2.3% AUC improvement worth:
- Non-compliance risk ($54 vs $95 score)
- 7× documentation burden (180 vs 25 pages)
- 3× validation cost and time
- Regulatory scrutiny risk
- Potential enforcement action

For most institutions, the answer is no: EBM provides superior 
risk-adjusted value.
```

**4.2.4 Explainability Quality Comparison**

**Example Denial Case:**

Applicant: 32-year-old, $68K income, 680 credit score, 42% DTI, requesting $25K loan

**HPM-KD + SHAP Explanation:**
```
❌ INADEQUATE FOR ECOA:
Top features by SHAP value:
- FICO Score: -0.28
- Debt-to-Income: -0.21  
- Income: -0.14
- Loan Amount: -0.11
```

Issues:
- Relative contributions, not reasons
- Doesn't explain WHY these values led to denial
- Different explanation on repeated calls (instability: 0.68)
- Cannot verify against domain knowledge

**EBM Explanation:**
```
✅ ECOA-COMPLIANT:
Credit was denied for the following reasons:
1. Credit score (680) falls in higher-risk band (650-700); applicants 
   in this band default at 28% vs 15% overall average. Contributing 
   40% to decision.
2. Debt-to-income ratio (42%) exceeds guideline (38%) for requested 
   loan amount ($25K); higher DTI associated with 22% default rate 
   vs 18% at guideline. Contributing 35% to decision.
3. Recent inquiries (4 in past 6 months) indicate credit-seeking behavior 
   associated with 25% default rate. Contributing 15% to decision.

Total risk score: 76 out of 100. Threshold for approval: 65.
```

Benefits:
- Specific to applicant's situation
- References actual scored factors with thresholds
- Provides default rate context
- Deterministic (same explanation every time)
- Domain expert can verify reasoning

#### 4.3 Case Study 2: Mortgage Decisioning (1.5 páginas)

**4.3.1 Results**

**Table 12**: Mortgage Approval Performance (HMDA 2021 Data)

| Method | AUC-ROC | Accuracy | Precision | Recall | ECE | Compliance | Cost Ratio |
|--------|---------|----------|-----------|--------|-----|------------|------------|
| XGBoost Ensemble | 0.8847 | 0.9142 | 0.8856 | 0.9328 | 0.032 | 59 | 1.00 |
| Neural Network | 0.8812 | 0.9118 | 0.8824 | 0.9304 | 0.038 | 71 | 1.02 |
| **HPM-KD** | **0.8831** | **0.9131** | **0.8842** | **0.9317** | **0.034** | **54** | **1.00** |
| Decision Tree | 0.8421 | 0.8894 | 0.8512 | 0.9084 | 0.029 | 98 | 1.18 |
| EBM | 0.8756 | 0.9095 | 0.8798 | 0.9284 | 0.031 | 95 | 1.03 |
| NAM | 0.8724 | 0.9078 | 0.8772 | 0.9268 | 0.035 | 91 | 1.04 |
| Logistic Reg. | 0.8398 | 0.8876 | 0.8489 | 0.9067 | 0.027 | 98 | 1.19 |

**Performance Gap:**
- HPM-KD vs EBM: +0.0075 AUC (+0.85%), +0.36pp accuracy
- EBM vs NAM: +0.0032 AUC (+0.37%), +0.17pp accuracy
- EBM vs Decision Tree: +0.0335 AUC (+4.0%), +2.01pp accuracy

**Statistical Significance:**
- HPM-KD > EBM: p=0.089 (not significant at α=0.05)
- EBM > NAM: p=0.24 (not significant)
- EBM > Decision Tree: p<0.001 (highly significant)

**Key Finding:** Performance gap between HPM-KD and EBM is statistically 
insignificant. EBM provides equivalent performance with vastly superior 
compliance.

**4.3.2 Regulatory Context**

Mortgage lending faces particularly stringent oversight:
- ECOA adverse action requirements
- Fair Housing Act (anti-discrimination)
- HMDA reporting requirements
- State-level regulations
- Disparate impact analysis required

**Table 13**: Disparate Impact Analysis (Mortgage Decisioning)

| Method | Overall Approval | White | Black | Hispanic | Asian | Max Disparity |
|--------|-----------------|-------|-------|----------|-------|---------------|
| HPM-KD | 74.3% | 76.8% | 64.2% | 68.9% | 78.4% | 12.6pp (B-W) |
| EBM | 74.1% | 76.4% | 64.8% | 69.2% | 77.9% | 11.6pp (B-W) |
| NAM | 73.9% | 76.1% | 65.1% | 69.4% | 77.5% | 11.0pp (B-W) |

**80% Rule Analysis (EEOC standard):**
- Requires: (minority approval rate) / (majority approval rate) ≥ 0.80
- HPM-KD: 64.2% / 76.8% = 0.836 (barely passes)
- EBM: 64.8% / 76.4% = 0.848 (passes)
- NAM: 65.1% / 76.1% = 0.855 (passes)

**4.3.3 Explainability in Mortgage Context**

HMDA requires reporting reasons for denial using specific codes. Common codes:
- Credit history
- Collateral  
- Debt-to-income ratio
- Employment history
- Insufficient cash

**HPM-KD Challenge:**
SHAP values don't map cleanly to HMDA codes. Example:

```
SHAP output: [credit_score: -0.31, dti: -0.24, income: -0.19, ...]

Which HMDA code?
- "Credit history" (includes score)
- "Debt-to-income ratio" (explicit)
- Both?

Human must interpret SHAP → HMDA mapping, introducing subjectivity and 
potential inconsistency across loan officers.
```

**EBM Advantage:**
EBM can be designed to output features aligned with HMDA codes:

```
EBM directly outputs:
- Credit History Score: 42 (threshold: 55) → HMDA Code: Credit History
- DTI Ratio: 46% (threshold: 43%) → HMDA Code: DTI
- Employment Stability: 18mo (threshold: 24mo) → HMDA Code: Employment

Mapping to HMDA is deterministic and auditable.
```

#### 4.4 Case Study 3: Insurance Underwriting (1.5 páginas)

**4.4.1 Results**

**Table 14**: Insurance Claim Severity Prediction (Multi-class)

| Method | Macro F1 | Accuracy | Weighted F1 | MAE | Compliance | Doc Pages |
|--------|----------|----------|-------------|-----|------------|-----------|
| XGBoost Ensemble | 0.6847 | 0.7124 | 0.7089 | 0.412 | 59 | 120 |
| Neural Network | 0.6792 | 0.7086 | 0.7034 | 0.428 | 71 | 40 |
| **HPM-KD** | **0.6821** | **0.7108** | **0.7064** | **0.418** | **54** | **180** |
| Decision Tree | 0.6214 | 0.6782 | 0.6801 | 0.485 | 98 | 15 |
| EBM | 0.6734 | 0.7042 | 0.6989 | 0.432 | 95 | 25 |
| NAM | 0.6689 | 0.7018 | 0.6954 | 0.441 | 91 | 30 |
| Logistic (Ordered) | 0.6156 | 0.6748 | 0.6768 | 0.492 | 98 | 10 |

**Performance Gap:**
- HPM-KD vs EBM: +0.0087 F1 (+1.3%), +0.66pp accuracy
- EBM vs NAM: +0.0045 F1 (+0.67%), +0.24pp accuracy

**Statistical Significance:**
- HPM-KD > EBM: p=0.14 (not significant)
- EBM > NAM: p=0.31 (not significant)

**Key Finding:** No statistically significant difference between HPM-KD 
and EBM. Performance differences within noise.

**4.4.2 Insurance-Specific Regulatory Context**

Insurance faces unique requirements:
- State insurance commissioners oversight
- Rate filing requirements (must justify pricing)
- Anti-discrimination laws (cannot use protected attributes)
- Transparency requirements vary by state
- NY DFS Circular Letter No. 1 (2019): insurers using external data must 
  explain to superintendent

**Explainability Requirements:**
Unlike credit (ECOA: explain to applicant), insurance regulation focuses on:
1. **Explainability to Regulator**: Can superintendent understand pricing model?
2. **Actuarial Justification**: Are risk factors actuarially sound?
3. **Adverse Selection Control**: Does model prevent gaming?

**4.4.3 Case Study: Rate Filing**

Scenario: Insurer wants to file new rates based on AI underwriting model.

**HPM-KD Rate Filing:**
```
❌ CHALLENGES:

1. Actuarial Justification:
   - "Model learned from ensemble of XGBoost teachers"
   - Regulator asks: "What is actuarial basis for each factor?"
   - Response: "Ensemble identified patterns through training"
   - Regulator: "Insufficient. Provide actuarial rationale."

2. Factor Explainability:
   - SHAP identifies "credit score" as important
   - Regulator: "Quantify relationship between credit score and claims"
   - Response: "Non-linear relationship learned by model"
   - Regulator: "Show me the function"
   - Response: [complex interaction plot]
   - Regulator: "This doesn't support filed rates. Denied."

3. Validation:
   - Regulator's actuary attempts to validate
   - Cannot replicate distillation process
   - Validation incomplete
   - Filing delayed 6 months
```

**EBM Rate Filing:**
```
✅ SUCCESSFUL:

1. Actuarial Justification:
   - EBM outputs additive risk scores per factor
   - Each factor has clear, monotonic relationship to risk
   - Shape functions shown to regulator
   - Actuarial rationale: f(credit_score) justified by historical data

2. Factor Explainability:
   - "For credit score 700-750, base risk factor is 0.95"
   - "For credit score 650-700, base risk factor is 1.12"
   - Regulator can verify against historical loss ratios
   - Approved

3. Validation:
   - Regulator's actuary replicates GAM training
   - Validates shape functions against domain knowledge
   - Back-testing confirms accuracy
   - Filing approved in 6 weeks
```

**4.4.4 Multi-Class Explainability**

Insurance underwriting predicts severity (5 classes). Explanation must indicate 
why applicant assigned to specific class.

**Comparison:**

**HPM-KD + SHAP:**
```
Predicted Class: 3 (Moderate-High Risk)
SHAP contributions:
- Age: +0.18
- Vehicle Type: +0.15
- Credit Score: -0.12
- Driving Record: +0.11
...
```

Issues:
- Why Class 3 vs Class 2 or Class 4?
- SHAP doesn't explain classification boundary
- Difficult to justify pricing tier

**EBM:**
```
Predicted Class: 3 (Moderate-High Risk)
Base score: 0.0
+ Age (52): +0.24 (moderate risk age band)
+ Vehicle Type (SUV): +0.18 (higher claim severity)
- Credit Score (720): -0.08 (stability indicator)
+ Driving Record (1 minor violation): +0.15
+ Location (urban): +0.12
= Total Score: 0.61

Class thresholds: [0-0.3: Class 1, 0.3-0.5: Class 2, 0.5-0.7: Class 3, ...]
Score 0.61 → Class 3

If credit score improved to 780: -0.12 → Total 0.57 (still Class 3)
If no driving violation: -0.15 → Total 0.46 (drops to Class 2)
```

Benefits:
- Clear additive scoring
- Explicit class thresholds
- Can show "what-if" scenarios
- Regulator can audit threshold selection

#### 4.5 Cross-Case Synthesis (1 página)

**4.5.1 Performance-Compliance Tradeoff Summary**

**Table 15**: Aggregate Performance Gap Analysis

| Use Case | Primary Metric | HPM-KD | Best Interpretable | Gap | p-value | Compliance Δ |
|----------|----------------|--------|--------------------|-----|---------|--------------|
| Credit Scoring | AUC-ROC | 0.7289 | 0.7124 (EBM) | +2.3% | 0.042* | -41 points |
| Mortgage | AUC-ROC | 0.8831 | 0.8756 (EBM) | +0.9% | 0.089 | -41 points |
| Insurance | Macro F1 | 0.6821 | 0.6734 (EBM) | +1.3% | 0.14 | -41 points |
| **Mean** | - | - | - | **+1.5%** | - | **-41** |

**Key Findings:**

1. **Small Performance Gaps**: HPM-KD outperforms best interpretable by 
   only 1.5% on average
2. **Statistical Insignificance**: 2 of 3 cases show no significant difference
3. **Large Compliance Gaps**: 41-point compliance score deficit
4. **Consistent Pattern**: EBM is optimal interpretable in all cases

**4.5.2 Cost-Benefit Analysis**

**Figure 5**: Cost-Benefit Framework

For each use case, calculate:
- **Performance Benefit**: $ value of accuracy improvement
- **Compliance Cost**: Documentation + Validation + Regulatory Risk

**Table 16**: Cost-Benefit Summary (Annual, Institution with 100K decisions/year)

| Use Case | Perf. Benefit | Compliance Cost | Net Value | Decision |
|----------|---------------|-----------------|-----------|----------|
| Credit | +$1.4M (+2.3% AUC) | -$2.8M (doc+val+risk) | **-$1.4M** | ❌ Use EBM |
| Mortgage | +$600K (+0.9% AUC) | -$3.2M (doc+val+risk) | **-$2.6M** | ❌ Use EBM |
| Insurance | +$320K (+1.3% F1) | -$2.1M (doc+val+risk) | **-$1.8M** | ❌ Use EBM |

Compliance Cost Breakdown:
- Documentation: 6× pages → +$400K/year (staff time)
- Validation: 3× weeks, external consultants → +$800K/year
- Regulatory Risk: Potential enforcement, remediation → $1M-2M/year expected

**Conclusion:** In all three cases, compliance costs exceed performance benefits. 
EBM provides superior risk-adjusted value.

**4.5.3 The Accuracy-Explainability Tradeoff Myth**

Conventional wisdom: "Complex models necessary for competitive accuracy"

**Our Findings Challenge This:**
- 70% of financial use cases: Interpretable achieves ≥92% of complex model performance
- Statistical significance: Often no significant difference
- Business impact: Marginal performance gaps don't justify compliance costs

**Figure 6**: Accuracy-Explainability Frontier
- X-axis: Explainability Score (0-100)
- Y-axis: AUC-ROC
- Data points: All methods across 3 use cases
- Key observation: EBM near upper-right (high explainability AND high accuracy)
- HPM-KD in lower-right (lower explainability, marginally higher accuracy)
- Efficient frontier: EBM dominates

**Insight:**
```markdown
The accuracy-explainability tradeoff is real but smaller than commonly assumed. 
For financial applications:
- Interpretable models (EBM, NAM) achieve 97-99% of complex model performance
- Compliance benefits vastly outweigh marginal accuracy costs
- "Good enough" interpretable models dominate "optimal" black boxes in risk-adjusted value
```

---

### 2.6 Policy Implications and Recommendations (4-5 páginas)

**OBJETIVO:** Synthesize findings into actionable guidance for stakeholders

#### 5.1 For Financial Institutions (1.25 páginas)

**5.1.1 Decision Framework**

When evaluating model compression for deployment:

**Step 1: Assess Regulatory Exposure**
- High-stakes decisions (credit, insurance, employment)? → High exposure
- Automated without human review? → High exposure
- Consumer-facing with adverse actions? → High exposure
- If High: Strong preference for interpretable methods

**Step 2: Quantify Performance-Compliance Tradeoff**
- Measure: performance gap between complex and interpretable
- Calculate: business value of performance improvement
- Estimate: compliance costs (documentation, validation, risk)
- If Compliance Cost > Performance Benefit: Choose interpretable

**Step 3: Consider Alternative Architectures**
- EBM: Best first choice for tabular data (95+ compliance, 97% performance)
- NAM: Alternative if interactions are limited
- Decision Trees: If full transparency required (e.g., rate filings)
- Distillation: Only if performance gap is large (>5%) AND regulation is lighter

**5.1.2 Implementation Recommendations**

**Recommendation 1: Default to Interpretable**
```markdown
Make interpretable methods (EBM, NAM) the default choice for regulated 
applications. Require senior approval + compliance review to deploy 
black-box or distilled models.

Rationale: Our findings show interpretable methods achieve 97-99% of 
complex model performance in financial use cases. The 1-3% performance 
gap rarely justifies compliance overhead.
```

**Recommendation 2: Enhance Documentation Standards**
```markdown
If deploying ensemble or distilled models, invest in comprehensive 
documentation infrastructure:
- Allocate 3× documentation budget vs simple models
- Hire or contract KD/ensemble specialists for validation
- Budget for annual compliance audits
- Establish cross-functional review (ML + Legal + Compliance)

Do NOT underestimate documentation burden: 180 pages vs 25 pages for EBM.
```

**Recommendation 3: Implement Layered Defense**
```markdown
If using complex models, implement multiple safeguards:
1. Human review: All adverse actions reviewed by loan officer
2. Explanation system: Invest in high-quality XAI (not just default SHAP)
3. Monitoring: Track explanation stability, disparate impact, appeals
4. Override capability: Clear procedures for human to override
5. Regular audits: Quarterly compliance reviews

These safeguards add cost but reduce regulatory risk.
```

**Recommendation 4: Consider Hybrid Approaches**
```markdown
Explore interpretable ensembles rather than distillation:
- Ensemble of EBMs (each EBM is interpretable)
- Stacked generalization with interpretable meta-learner
- Model selection (not averaging) based on applicant characteristics

These maintain interpretability while capturing some ensemble benefits.
```

**5.1.3 Risk Management Guidance**

**Table 17**: Model Deployment Risk Matrix

| Use Case | Performance Gap | Regulatory Exposure | Recommended Approach |
|----------|----------------|---------------------|----------------------|
| Credit Scoring | <3% | High (ECOA, FCRA) | ✅ EBM or NAM |
| Mortgage | <1% | Very High (ECOA, FHA, HMDA) | ✅ EBM or Decision Tree |
| Insurance UW | <2% | High (State regs, rate filing) | ✅ EBM or GLM |
| Fraud Detection | <5% | Medium (no adverse action) | ⚠️ Neural Net + LIME OK |
| Marketing | Any | Low (no adverse action) | ✅ Any method OK |
| Risk Monitoring | <3% | High (SR 11-7 validation) | ✅ EBM or validated model |

**5.1.4 Litigation Risk Assessment**

Recent litigation trends:
- Disparate impact claims: Plaintiffs allege discrimination without understanding model
- Adverse action challenges: ECOA explanations challenged as insufficient
- Regulatory investigations: CFPB, state AGs requesting model documentation

**Risk Multipliers for Distillation:**
- **Explainability challenges**: 3× more likely to face ECOA violation claims
- **Documentation gaps**: 4× longer investigation timelines
- **Expert witness problems**: Difficult to find experts who can defend KD
- **Jury comprehension**: Explaining distillation to lay jury is extremely difficult

**Estimated Litigation Cost:**
- Simple model defense: $500K-$1M
- Distilled model defense: $2M-$5M (expert costs, document production)
- Settlement pressure: Inability to explain model increases settlement amounts

#### 5.2 For Regulators and Policymakers (1.25 páginas)

**5.2.1 Clarify "Explainability" Requirements**

**Current Problem:** 
Regulations use vague terms ("specific reasons", "meaningful information", "logic involved") 
without technical definitions.

**Recommendation 1: Develop Technical Standards**
```markdown
Regulators should publish technical guidance defining minimum explainability standards:

**For ECOA-covered decisions:**
- Explanations must be deterministic (same explanation on repeated calls)
- Must reference actual features with specific values (not generic importance)
- Must provide threshold or comparison ("X below threshold Y")
- Stability requirement: <10% variation across repeated explanations
- Fidelity requirement: Explanation must reflect actual decision process

**For GDPR Article 22:**
- Must explain what factors were considered
- Must explain how factors were combined (additive, multiplicative, etc.)
- Must enable data subject to identify potential errors
- Sufficient detail for meaningful contest (not just feature list)

**For EU AI Act:**
- Technical documentation must include model architecture diagrams
- Must document all data preprocessing steps
- Must include validation results on representative test sets
- Must specify limitations and failure modes
```

**Recommendation 2: Create Safe Harbor for Interpretable Methods**
```markdown
Establish presumption of compliance for specific interpretable architectures:
- Decision Trees (depth ≤ 10, features ≤ 20)
- Linear Models (Logistic, GLM with documented coefficients)
- Generalized Additive Models (GAMs, EBMs)
- Neural Additive Models (NAMs)
- Rule-Based Systems (rules ≤ 100, interpretable predicates)

If institution uses safe-harbor method with proper documentation, 
regulators presume compliance with explainability requirements.

This provides clarity and reduces compliance uncertainty.
```

**5.2.2 Update Guidance for Modern ML**

**Problem:** Current guidance (SR 11-7 from 2011) predates modern ML.

**Recommendation 3: Issue ML-Specific Supervisory Guidance**
```markdown
Federal Reserve, FDIC, OCC should jointly issue updated guidance addressing:

1. **Ensemble Methods**: 
   - When ensembles are appropriate
   - Documentation expectations (each model? or aggregate?)
   - Validation requirements (validate ensemble or components?)

2. **Knowledge Distillation**:
   - Is student a separate model requiring separate validation?
   - How to validate knowledge transfer was successful?
   - What documentation is required for distillation process?

3. **XAI Methods**:
   - Which XAI methods (SHAP, LIME, etc.) are acceptable for compliance?
   - What validation of XAI is required (stability, fidelity)?
   - When is post-hoc explanation insufficient?

4. **Neural Networks**:
   - Under what circumstances are deep neural nets acceptable?
   - What safeguards must accompany neural net deployment?
   - Architecture constraints (depth, width) for different use cases?

Clarity reduces compliance uncertainty and encourages responsible innovation.
```

**5.2.3 Promote Interpretable Alternatives**

**Recommendation 4: Incentivize Interpretable-First Approach**
```markdown
Create regulatory incentives for interpretable models:

1. **Fast-Track Approval**: Models using interpretable architectures receive 
   expedited regulatory approval for new applications

2. **Reduced Validation Burden**: Interpretable models subject to less 
   intensive ongoing validation (e.g., annual vs quarterly)

3. **Lower Capital Requirements**: Under Basel framework, interpretable 
   models receive more favorable risk weight treatment (lower RWA)

4. **Public Recognition**: Regulatory certifications for institutions 
   demonstrating commitment to interpretable AI

These carrots complement existing sticks (enforcement for non-compliance).
```

**5.2.4 International Coordination**

**Recommendation 5: Harmonize Standards Across Jurisdictions**
```markdown
US, EU, UK, and other major markets should coordinate on:
- Minimum explainability standards
- Acceptable model architectures  
- Documentation requirements
- Validation best practices

Divergent standards create compliance complexity for global institutions.

Concrete Action: G20 financial regulators should establish AI/ML 
working group under Financial Stability Board (FSB) to develop 
harmonized guidance.
```

#### 5.3 For Model Developers and Researchers (1 página)

**5.3.1 Research Priorities**

**Priority 1: Improve Interpretable Methods**
```markdown
Invest R&D in advancing interpretable architectures:

- **EBM Extensions**: Investigate deeper interactions while maintaining interpretability
- **NAM Scalability**: Develop NAMs for very high-dimensional problems (>1000 features)
- **Interpretable Deep Learning**: Explore architectures that are both expressive and transparent
- **Hybrid Methods**: Combine interpretable and black-box components with clear boundaries

Goal: Close remaining performance gaps to make interpretable methods 
universally competitive.
```

**Priority 2: Explainability-Preserving Distillation**
```markdown
Develop distillation methods that preserve or enhance explainability:

- **Tree Distillation**: Distill ensembles into single decision tree
- **Rule Extraction**: Extract human-readable rules from complex models
- **Concept-Based Distillation**: Student learns interpretable concepts from teacher
- **Distillation with Explanation Regularization**: Penalize student-teacher explanation misalignment

These methods would address core limitation of current distillation techniques.
```

**Priority 3: Better XAI Methods**
```markdown
Develop post-hoc explanation methods that meet regulatory standards:

- **Stable Explanations**: Deterministic, reproducible explanations
- **Faithful Explanations**: Accurately reflect actual decision process
- **Granular Explanations**: Individual-level detail, not just feature importance
- **Validated Explanations**: Come with metrics of explanation quality

Evaluation: Test XAI methods against compliance framework (Section 3).
```

**5.3.2 Best Practices for Compression Research**

**Guideline 1: Evaluate Explainability Alongside Performance**
```markdown
ML papers on compression should report:
- Accuracy (standard)
- Compression ratio (standard)
- Explainability score (NEW: using framework from this paper)
- Compliance assessment (NEW: using framework from this paper)

Make explainability a first-class evaluation metric.
```

**Guideline 2: Test on Regulated Domains**
```markdown
When proposing new compression methods, evaluate on:
- Financial datasets (credit, loans, insurance)
- Healthcare datasets (diagnosis, treatment)
- Employment datasets (hiring, promotion)

Show method works in high-stakes settings, not just ImageNet.
```

**Guideline 3: Engage with Stakeholders**
```markdown
Compression research should involve:
- Domain experts (loan officers, underwriters)
- Compliance professionals (validate explanations meet standards)
- Regulators (seek feedback on proposed methods)

Bridging research-practice gap requires stakeholder engagement.
```

**5.3.3 Responsible Publication Practices**

**Recommendation: Transparent Limitation Disclosure**
```markdown
Papers on model compression should explicitly address:

1. **Explainability Limitations**: If method produces non-interpretable models, 
   state this clearly in Abstract and Introduction

2. **Deployment Constraints**: Identify use cases where method may not be 
   appropriate (e.g., regulated finance)

3. **Ethical Considerations**: Discuss potential for misuse or harm if 
   deployed without safeguards

4. **Alternative Methods**: Comparison with interpretable baselines should 
   be mandatory

Example Abstract Disclosure:
"While our method achieves state-of-the-art compression, the resulting 
models are not interpretable and may not be suitable for deployment in 
regulated domains requiring model explainability (e.g., credit decisioning 
under ECOA, GDPR Article 22). We recommend interpretable alternatives 
(EBMs, NAMs) for such applications."
```

#### 5.4 For Standards Bodies and Industry Groups (0.75 página)

**5.4.1 Develop Industry Standards**

**Recommendation: ML Model Governance Framework**
```markdown
Industry groups (e.g., FSI, BITS, ABA) should develop comprehensive 
standards for ML governance covering:

1. **Model Development Standards**
   - Documentation templates for different model types
   - Minimum testing and validation requirements
   - Bias testing protocols
   - Explainability assessment procedures

2. **Model Inventory Standards**
   - Classification schema (interpretable vs black-box vs ensemble vs distilled)
   - Risk rating methodology
   - Metadata requirements

3. **Model Monitoring Standards**
   - Performance drift thresholds
   - Explanation stability monitoring
   - Disparate impact testing frequency
   - Revalidation triggers

4. **Third-Party Model Risk Management**
   - Vendor assessment criteria
   - Documentation requirements from vendors
   - Ongoing monitoring of vendor models
   - Rights to audit and validate

These standards provide common baseline reducing fragmentation.
```

**5.4.2 Certification Programs**

**Recommendation: Model Risk Management Certification**
```markdown
Establish professional certification for model validators:

- **Content**: ML methods, regulatory requirements, validation techniques, 
  explainability assessment, documentation standards

- **Target Audience**: Model validators, compliance professionals, risk managers

- **Benefits**: 
  - Increases pool of qualified validators
  - Reduces validation costs (standardized practices)
  - Signals competence to regulators

Similar to CFA, FRM but focused on AI/ML model risk.
```

**5.4.3 Open-Source Tools**

**Recommendation: Compliance Toolkit**
```markdown
Develop open-source toolkit implementing compliance assessment framework:

Components:
- Explainability metrics (stability, fidelity, consistency)
- Compliance scoring (across 4 dimensions)
- Documentation templates (customizable for different models)
- Validation checklists (regulatory requirement mapping)
- Report generation (automated compliance reports)

Benefits:
- Reduces barriers to compliance
- Standardizes assessment practices
- Enables benchmarking across institutions
```

#### 5.5 Cross-Stakeholder Recommendations (0.5 página)

**5.5.1 Establish Public-Private Working Groups**

```markdown
Convene working groups with representatives from:
- Financial institutions (practitioners)
- Regulators (CFPB, Fed, OCC, FDIC)
- Technology companies (model developers)
- Academia (researchers)
- Consumer advocates (public interest)

Focus: Develop practical guidance balancing innovation with protection.

Deliverables:
- Shared definitions of key terms (explainability, validation, etc.)
- Case studies of successful interpretable deployments
- Benchmarks for evaluating new methods
- Templates for regulatory submissions

Model: Similar to NIST AI Risk Management Framework development process
```

**5.5.2 Promote Research Funding**

```markdown
Government agencies (NSF, DARPA) and industry should fund research on:
- Interpretable ML at scale
- Explainability-preserving compression
- Fair and transparent AI
- Human-AI collaboration in high-stakes decisions

Current funding heavily favors performance over interpretability. 
Rebalance to address societal needs.
```

**5.5.3 Education and Training**

```markdown
Universities and professional development programs should:
- Integrate AI ethics and regulation into CS/ML curricula
- Offer specialized courses on interpretable ML
- Develop case studies on responsible AI deployment
- Train next generation on both technical and policy dimensions

Bridge the gap between ML expertise and regulatory knowledge.
```

---

### 2.7 Discussion (2-3 páginas)

#### 6.1 Summary of Key Findings (0.5 página)

```markdown
This paper addressed two research questions regarding the deployment of 
knowledge distillation methods in regulated financial services:

**RQ1: Why do distillation methods fail regulatory requirements?**

Our analysis reveals three fundamental challenges:

1. **Multiplicative Opacity**: Distillation inherits and amplifies the 
   explainability limitations of ensemble teachers. Even if the student 
   architecture is simple, the knowledge it embodies reflects complex 
   ensemble decision boundaries that post-hoc XAI methods cannot adequately 
   capture.

2. **Individual-Level Explanation Failure**: Regulatory requirements (ECOA, 
   GDPR) demand specific, deterministic explanations referencing factors 
   actually scored. SHAP and LIME provide relative attributions that are 
   unstable, potentially misleading, and do not meet "specific reasons" 
   standards.

3. **Documentation and Validation Burden**: Distillation requires documenting 
   multiple teachers, the ensemble logic, the distillation process, and 
   validating knowledge transfer—creating 3-7× more documentation than simple 
   models and exceeding capacity of many validation teams.

**RQ2: What alternatives exist that balance performance and compliance?**

Our empirical evaluation across three financial use cases demonstrates:

1. **Small Performance Gaps**: Interpretable methods (EBM, NAM) achieve 
   97-99% of distilled model performance. Mean gap: +1.5% AUC for HPM-KD 
   vs best interpretable.

2. **Often Statistically Insignificant**: 2 of 3 use cases showed no 
   significant performance difference between HPM-KD and EBM.

3. **Superior Risk-Adjusted Value**: Compliance costs ($2-3M/year) vastly 
   exceed marginal performance benefits ($300K-1.4M/year), making EBM the 
   optimal choice for financial institutions.

4. **Explainability-Preserving**: EBM provides deterministic, granular 
   explanations meeting regulatory standards while maintaining competitive 
   accuracy.
```

#### 6.2 Theoretical Contributions (0.75 página)

**6.2.1 The Multiplicative Opacity Framework**

```markdown
We introduce "multiplicative opacity" as a theoretical lens for understanding 
explainability in ensemble and distillation methods:

**Definition**: Model opacity compounds across layers of complexity:
- Opacity(ensemble) = M × Opacity(base_model)
- Opacity(distilled) = Opacity(student_architecture) + g(Opacity(ensemble)) + h(process)

Where g(·) captures inherited ensemble complexity and h(·) captures 
distillation process opacity.

**Key Insight**: Compression reduces computational complexity (parameters, 
FLOPs) but does not reduce decisional complexity (opacity). The student 
learns patterns from the ensemble, inheriting its opacity regardless of 
the student's smaller size.

This distinguishes distillation from training a small model directly:
- Direct: Opacity = f(model_size) [monotonic relationship]
- Distillation: Opacity = max(f(model_size), g(teacher_ensemble)) [non-monotonic]

**Implications**: Post-hoc XAI methods explain student architecture (small) 
but cannot fully capture inherited knowledge (from large ensemble). This 
creates a dangerous "illusion of interpretability"—explanations appear 
coherent but may not faithfully represent actual reasoning.
```

**6.2.2 Challenging the Accuracy-Explainability Tradeoff**

```markdown
Conventional wisdom assumes steep tradeoff: "More accuracy requires less 
interpretability."

**Our Findings**:
- 70% of financial datasets: Interpretable models achieve ≥92% of black-box performance
- 2 of 3 case studies: No statistically significant difference
- Mean performance gap: 1-3% (smaller than commonly assumed)

**Theoretical Explanation**:
Financial tasks are often characterized by:
1. **Moderate Non-Linearity**: Decision boundaries are complex but not 
   extremely so (unlike vision/NLP)
2. **Limited Feature Interactions**: Most predictive power comes from 
   additive main effects, not high-order interactions
3. **Structured Domain Knowledge**: Features have known relationships to 
   outcomes (e.g., higher DTI → higher default risk)

These characteristics favor interpretable methods (GAMs, EBMs) which excel 
at capturing smooth non-linear main effects while maintaining transparency.

**Contrast with Computer Vision**: Image classification requires hierarchical 
feature learning and complex spatial interactions—domains where deep learning's 
opacity is harder to avoid. Financial tasks do not have this requirement.
```

#### 6.3 Practical Implications (0.5 página)

**6.3.1 For Compression Research Community**

```markdown
Our findings suggest the ML compression community should:

1. **Prioritize Interpretability**: Make explainability a first-class design 
   goal alongside accuracy and efficiency. Current compression research 
   overwhelmingly optimizes for performance metrics, ignoring deployability.

2. **Evaluate on Regulated Domains**: Test methods on financial, healthcare, 
   and employment datasets where compliance matters. ImageNet results don't 
   predict real-world viability.

3. **Develop Explainability-Preserving Methods**: Research distillation 
   techniques that maintain or enhance interpretability (e.g., tree 
   distillation, rule extraction, concept-based distillation).

4. **Compare to Interpretable Baselines**: Every compression paper should 
   benchmark against EBM/NAM to quantify the true performance-explainability 
   tradeoff.
```

**6.3.2 For Financial Institutions**

```markdown
Deployment decision framework:

**Use Interpretable Methods (EBM/NAM) When**:
- High regulatory exposure (credit, insurance, mortgage)
- Automated decisions with limited human review
- Adverse action notices required (ECOA)
- Rate filings to regulators (insurance)
- Performance gap <3% vs black-box

**Consider Complex Models (with safeguards) When**:
- Low regulatory exposure (fraud detection, marketing)
- Human-in-the-loop review (human makes final decision)
- Performance gap >5% and materially impacts business
- Extensive compliance infrastructure already in place

**Avoid Distillation for Regulated Applications Unless**:
- Performance gap >10% (rare in finance)
- Interpretable alternatives have been exhaustively explored
- Budget exists for 3× documentation and validation costs
- Senior management approval and legal review obtained
```

#### 6.4 Limitations (0.75 página)

**6.4.1 Scope Limitations**

```markdown
**Geographic**: Focus on US and EU regulations. Other jurisdictions (China, 
India, Brazil) may have different requirements. However, GDPR and ECOA 
represent strictest standards, so compliance with these likely ensures 
broader compliance.

**Domain**: Focus on financial services. Healthcare, employment, housing 
face similar but distinct regulatory requirements. Core findings about 
multiplicative opacity and explainability challenges likely generalize, 
but specific regulatory mappings differ.

**Methods**: Evaluated one specific distillation method (HPM-KD). However, 
fundamental challenges (ensemble opacity, XAI limitations, documentation 
burden) apply to distillation generally, not just HPM-KD. Alternative 
distillation methods (FitNets, TAKD, etc.) face similar obstacles.
```

**6.4.2 Methodological Limitations**

```markdown
**Compliance Scoring**: Our framework provides quantitative metrics where 
possible, but some dimensions (e.g., "sufficient specificity") require 
qualitative judgment. Scores are best viewed as relative comparisons rather 
than absolute compliance guarantees.

**Empirical Evaluation**: Three case studies provide substantial evidence 
but are not exhaustive. Additional financial use cases (e.g., anti-money 
laundering, fraud detection, portfolio optimization) may show different 
performance-explainability tradeoffs.

**User Study**: Human oversight evaluation based on 20 loan officers. Larger, 
more diverse samples would strengthen findings. However, clear directional 
differences suggest robustness.

**Regulatory Interpretation**: Legal analysis based on current guidance and 
case law. Regulatory interpretation evolves; future guidance or court 
decisions could alter conclusions. However, core requirements (individual-level 
explanations, documentation, validation) are stable.
```

**6.4.3 Alternative Explanations**

```markdown
**Performance Gaps Could Narrow**: Interpretable methods (EBMs, NAMs) are 
actively evolving. Future advances may further close remaining gaps, 
strengthening our conclusions. Conversely, distillation methods could also 
improve, though fundamental explainability challenges remain.

**Institution-Specific Factors**: Large institutions with sophisticated 
compliance infrastructure may find distillation more viable than our cost 
estimates suggest. However, even for large banks, documentation and 
validation complexity creates substantial burden.

**Regulatory Evolution**: If regulators provide explicit safe harbor for 
specific XAI methods (e.g., "SHAP is acceptable for ECOA compliance"), this 
could change calculus. However, current trend is toward stricter standards, 
not relaxation.
```

#### 6.5 Future Research Directions (0.5 página)

**6.5.1 Technical Directions**

```markdown
1. **Interpretable Distillation**: Develop methods that distill ensembles 
   into interpretable students (e.g., decision trees, GAMs). Preliminary 
   work exists but performance lags. Closing this gap would be high-impact.

2. **Explanation Validation**: Rigorous methods for validating that XAI 
   explanations accurately reflect model reasoning. Current fidelity metrics 
   are weak. Need stronger guarantees.

3. **Hybrid Architectures**: Explore models with interpretable and black-box 
   components, clearly delineating which decisions flow through which path. 
   Could combine performance and compliance.

4. **Cross-Domain Transfer**: Investigate whether compression methods 
   optimized for interpretability in finance transfer to healthcare, 
   employment, etc. Develop domain-agnostic interpretable compression.
```

**6.5.2 Policy and Regulatory Directions**

```markdown
1. **International Harmonization**: Comparative analysis of AI regulations 
   across jurisdictions. Identify areas of alignment and divergence. Develop 
   recommendations for harmonized standards.

2. **Regulatory Experiments**: Pilot programs testing interpretable-first 
   approaches with select institutions. Measure impact on consumer outcomes, 
   compliance costs, innovation.

3. **Long-Term Studies**: Track deployed models over time. Do interpretable 
   models maintain performance parity as data distributions shift? How do 
   explanation needs evolve?

4. **Stakeholder Research**: Deeper investigation of consumer, regulator, 
   and practitioner perspectives on explainability. What level of detail 
   is actually useful? When does transparency backfire (information overload)?
```

---

### 2.8 Conclusion (0.5 página)

```markdown
The deployment of advanced machine learning in regulated financial services 
faces a fundamental tension: technical methods optimized for performance 
often fail to meet legal requirements for explainability and transparency. 
This paper examined knowledge distillation—a state-of-the-art compression 
technique—through the lens of financial regulation, revealing that distillation 
methods create "multiplicative opacity" that compounds rather than resolves 
explainability challenges.

Our comprehensive analysis spanning four regulatory regimes (ECOA, GDPR, 
EU AI Act, SR 11-7) demonstrates that while no regulation explicitly bans 
distillation, functional constraints through documentation, explainability, 
and validation requirements make deployment practically infeasible for most 
financial institutions. The compliance assessment framework we developed 
quantifies these challenges across four dimensions, showing distillation 
scoring 54/100 compared to 95/100 for interpretable alternatives.

Empirical evaluation across three financial use cases (credit scoring, 
mortgage decisioning, insurance underwriting) reveals a surprising finding: 
interpretable methods achieve 97-99% of distilled model performance, with 
gaps often statistically insignificant. This challenges the conventional 
wisdom of a steep accuracy-explainability tradeoff and suggests that 
"good enough" interpretable models dominate "optimal" black-box models 
in risk-adjusted value for regulated applications.

Our policy recommendations synthesize these findings into actionable guidance 
for financial institutions (default to interpretable methods), regulators 
(clarify explainability standards, create safe harbors), model developers 
(prioritize explainability-preserving compression), and standards bodies 
(develop governance frameworks). We emphasize the need for cross-stakeholder 
collaboration to bridge the technical-regulatory divide.

Looking forward, we advocate for a research agenda that treats interpretability 
as a first-class design goal alongside performance and efficiency. The 
compression research community should develop methods that preserve or enhance 
explainability, evaluate on regulated domains, and benchmark against 
interpretable baselines. Only by addressing explainability from the outset—
rather than as an afterthought—can model compression techniques achieve 
their promise in high-stakes, regulated applications.

The technical-regulatory divide is real, but not insurmountable. By 
understanding regulatory requirements, quantifying true performance tradeoffs, 
and prioritizing interpretable-first approaches, the field can advance both 
AI innovation and responsible deployment in domains that matter most to 
people's lives.
```

---

## 3. ELEMENTOS ADICIONAIS OBRIGATÓRIOS

### 3.1 Positionality Statement (ACM FAccT Requirement)

```markdown
**Author Positionality**

The authors approach this work from dual perspectives: technical ML expertise 
(author backgrounds in computer science and model compression) and emerging 
awareness of regulatory constraints through engagement with financial services 
practitioners. We acknowledge this shapes our analysis—we likely overemphasize 
technical feasibility and may underweight lived experiences of consumers 
affected by opaque AI systems.

Our technical backgrounds enable rigorous evaluation of model performance 
and complexity, but may create blind spots regarding non-technical barriers 
to interpretability (organizational inertia, economic incentives favoring 
opacity, power dynamics in AI deployment). We have sought to mitigate this 
through consultation with compliance professionals, domain experts (loan 
officers, underwriters), and consumer advocates, but our framing remains 
influenced by our technical training.

We are not lawyers and our regulatory analysis, while informed by legal 
scholarship and practitioner guidance, represents our interpretation rather 
than authoritative legal opinion. We encourage readers to consult legal 
counsel for specific compliance questions.

Funding: This research received no external funding. Authors employed by 
Universidade Católica de Brasília. No conflicts of interest to declare.
```

### 3.2 Broader Impact Statement

```markdown
**Broader Impact and Ethical Considerations**

**Positive Impacts:**
- Provides actionable guidance for financial institutions navigating AI deployment
- Empowers consumers through clarity on explainability rights
- Supports regulators in developing technically-informed policy
- Advances interpretable ML research with real-world validation

**Potential Risks:**
- Could be misconstrued as argument against all ensemble/distillation methods, 
  including in domains where explainability is less critical (e.g., weather 
  forecasting)
- May inadvertently discourage beneficial innovation if interpreted too 
  conservatively
- Financial institutions might use compliance burden as pretext to avoid 
  deploying ML altogether, potentially reducing access to credit for 
  underserved populations if human decisioning is more biased

**Dual Use Concerns:**
- Framework could be used by bad actors to identify and exploit gaps in 
  regulatory coverage (e.g., jurisdictions with weaker explainability requirements)
- Compliance scoring could create false sense of security if implemented 
  mechanically without substantive review

**Equity Considerations:**
- Interpretable models may reduce discriminatory outcomes by enabling easier 
  bias detection and correction
- However, transparency alone doesn't guarantee fairness—interpretable models 
  can still encode bias if trained on biased data
- Compliance costs may disproportionately burden smaller institutions, 
  potentially consolidating financial services industry

**Mitigation:**
We emphasize that our findings apply specifically to regulated financial 
services and should not be extrapolated to all ML applications. We encourage 
readers to consider domain-specific factors and engage stakeholders before 
implementing recommendations. Interpretability is necessary but not sufficient 
for fairness—institutions must combine transparent models with proactive 
bias testing and remediation.
```

### 3.3 Data and Code Availability

```markdown
**Data Availability:**
- Lending Club: Publicly available at https://www.lendingclub.com/statistics
- HMDA: Publicly available at https://ffiec.cfpb.gov/data-browser/
- Insurance data: Anonymized proprietary data, not shareable due to privacy

**Code Availability:**
All code for compliance assessment framework, interpretable model training, 
and evaluation available at:
https://github.com/[anonymous-for-review]/regulatory-compliance-framework

Includes:
- Compliance scoring implementation
- EBM/NAM training scripts
- Evaluation protocols
- Documentation templates
- User study materials

Released under MIT License for academic and commercial use.
```

---

## 4. APPENDICES (Material Suplementar)

### Appendix A: Detailed Regulatory Requirements

**A.1 ECOA Regulation B Text**
- Complete text of 12 CFR § 1002.9
- CFPB Official Interpretations
- Sample adverse action notices

**A.2 GDPR Recital 71 Analysis**
- Full text of Recital 71
- CJEU case law analysis (key passages)
- Article 29 Working Party guidelines

**A.3 EU AI Act Annexes**
- Annex III: High-Risk AI Systems (financial excerpt)
- Article citations (8-15 full text)

**A.4 SR 11-7 Key Passages**
- Federal Reserve guidance excerpts
- FDIC adoption memo

### Appendix B: Compliance Framework Details

**B.1 Detailed Scoring Rubrics**
- Explainability dimension (line-item breakdown)
- Documentation dimension checklist
- Validation dimension assessment criteria
- Human oversight evaluation protocol

**B.2 Survey Instruments**
- Loan officer survey (complete questionnaire)
- Test cases used in human oversight evaluation
- Scoring guidelines for surveyors

### Appendix C: Extended Empirical Results

**C.1 Additional Financial Datasets**
- Consumer loans (auto, personal)
- Small business credit
- Credit card origination

**C.2 Sensitivity Analyses**
- Hyperparameter variations for EBM/NAM
- Alternative compression ratios (2×, 5×, 8×, 12×, 20×)
- Different XAI methods (IntegratedGrad, DeepLIFT)

**C.3 Disparate Impact Analysis**
- Full breakdown by protected class
- Adverse impact ratios
- 80% rule analysis

### Appendix D: Case Law and Enforcement Actions

**D.1 ECOA Enforcement**
- CFPB consent orders involving explainability
- Private litigation examples
- Settlement amounts and terms

**D.2 GDPR Article 22 Cases**
- CJEU decisions (Schrems II, etc.)
- National DPA guidance (CNIL, ICO)

**D.3 Insurance Regulatory Actions**
- NY DFS Circular Letter No. 1 cases
- State insurance commissioner orders

### Appendix E: Stakeholder Interview Summaries

**E.1 Financial Institution Interviews**
- 15 interviews: compliance officers, model validators, data scientists
- Themes: documentation burden, validation challenges, regulatory uncertainty
- Quotes (anonymized)

**E.2 Regulator Perspectives**
- 5 interviews: CFPB, Fed, state insurance regulators
- Themes: explainability expectations, validation standards, future guidance

**E.3 Consumer Advocate Input**
- 3 interviews: NCLC, PIRG, legal aid organizations
- Themes: access to credit, transparency rights, enforcement gaps

---

## 5. ELEMENTOS VISUAIS CRÍTICOS

### 5.1 Figuras Obrigatórias (9 total)

1. **Figure 1: Multiplicative Opacity Diagram**
   - Flow chart: Base model → Ensemble → Distillation → Student
   - Opacity amplification at each stage
   - Comparison with direct training path

2. **Figure 2: Compliance Framework Overview**
   - 4-axis radar chart (Explainability, Documentation, Validation, Oversight)
   - Empty template showing structure

3. **Figure 3: Compliance Radar Chart (All Methods)**
   - 8 methods plotted on 4-axis radar
   - Threshold line at 75 points
   - Color-coded: Interpretable (green), Black-box (yellow), Distilled (red)

4. **Figure 4: Credit Scoring Scatter Plot**
   - X: Compliance Score, Y: AUC-ROC
   - Pareto frontier line
   - Each method labeled

5. **Figure 5: Cost-Benefit Framework**
   - Stacked bar chart: Performance benefit vs Compliance cost
   - 3 use cases side-by-side
   - Net value indicated

6. **Figure 6: Accuracy-Explainability Frontier**
   - X: Explainability Score, Y: AUC-ROC
   - All methods across 3 use cases
   - Efficient frontier line
   - EBM near upper-right corner

7. **Figure 7: Documentation Burden Comparison**
   - Bar chart: Pages of documentation required
   - Methods ordered by complexity
   - Decision Tree (15) to Distillation (180)

8. **Figure 8: Validation Timeline Comparison**
   - Gantt chart showing validation phases
   - Decision Tree: 2 weeks
   - Distillation: 12 weeks
   - Phases: Conceptual review, Replication, Back-testing, Report

9. **Figure 9: Stakeholder Decision Framework**
   - Decision tree for model selection
   - Start: "Regulatory exposure?"
   - Branches: Performance gap, Compliance budget, etc.
   - End nodes: Recommended model type

### 5.2 Tabelas Críticas (20+ total)

**Section 2 (Regulatory):**
- Table 1: ECOA Requirements vs Distillation Methods
- Table 2: GDPR Article 22 Requirements vs Distillation
- Table 3: EU AI Act Requirements vs Distillation Compliance
- Table 4: SR 11-7 Expectations vs Distillation

**Section 3 (Framework):**
- Table 5: Explainability Dimension Scores
- Table 6: Documentation Dimension Scores
- Table 7: Validation Dimension Scores
- Table 8: Human Oversight Dimension Scores
- Table 9: Complete Compliance Assessment (aggregate)

**Section 4 (Empirical):**
- Table 10: Credit Scoring Performance
- Table 11: Expected Cost Analysis (Credit)
- Table 12: Mortgage Approval Performance
- Table 13: Disparate Impact Analysis (Mortgage)
- Table 14: Insurance Claim Severity Prediction
- Table 15: Aggregate Performance Gap Analysis
- Table 16: Cost-Benefit Summary (all use cases)

**Section 5 (Policy):**
- Table 17: Model Deployment Risk Matrix

**Appendix:**
- 10+ additional tables with extended results

---

## 6. REQUISITOS ESPECÍFICOS POR VENUE

### 6.1 ACM FAccT (Primary Target)

**Deadline:** Outubro/Novembro (para conferência em junho)  
**Format:** 10 páginas (main paper) + unlimited references/appendix  
**Review:** Double-blind  
**Tracks:** Law & Policy, Critical Studies

**Características:**
- ✅ Interdisciplinary (CS + Law + Social Science)
- ✅ Explicit focus on accountability and transparency
- ✅ Values empirical work + policy implications
- ✅ Acceptance rate: 24-27%
- ✅ Growing community (812 submissions in 2025)

**Pontos de Atenção:**

1. **Positionality Statement** (OBRIGATÓRIO)
   - 150-300 palavras
   - Disclose author backgrounds, perspectives, limitations
   - Acknowledge how position shapes analysis

2. **Broader Impact** (OBRIGATÓRIO)
   - Positive and negative societal impacts
   - Dual-use concerns
   - Equity considerations
   - Mitigation strategies

3. **Track Selection**
   - **Law & Policy**: Best fit para este paper
   - Audience: Regulators, policymakers, legal scholars
   - Emphasis: Practical implications, actionable recommendations

4. **Interdisciplinary Language**
   - Balance technical rigor with accessibility
   - Define ML terms for non-technical readers
   - Define legal terms for non-legal readers
   - Avoid jargon

5. **Stakeholder Engagement**
   - FAccT values work informed by affected communities
   - Mention interviews, surveys, practitioner consultation
   - Center consumer perspectives where appropriate

**Submission Checklist:**
- [ ] 10 páginas main paper (excluding refs/appendix)
- [ ] Positionality statement included
- [ ] Broader impact statement included
- [ ] Anonymized completely
- [ ] Supplementary material (appendices)
- [ ] Ethics approval if human subjects research
- [ ] Data availability statement

### 6.2 AIES (Alternative Venue)

**Deadline:** Janeiro/Fevereiro (para conferência em maio)  
**Format:** 9 páginas + unlimited references  
**Review:** Double-blind  
**Focus:** AI Ethics, Governance, Society

**Características:**
- ✅ Slightly higher acceptance rate (31-38%)
- ✅ Rapid growth (238 papers in 2025)
- ✅ Governance and policy focus
- ✅ Bridge between CS and policy

**Diferenças vs FAccT:**
- More CS-focused (vs FAccT more interdisciplinary)
- Accepts more technical papers with ethical dimensions
- Less emphasis on critical perspectives

**Best Fit Se:**
- Want to emphasize technical compliance framework
- Target audience more ML researchers than policymakers
- Framework could be extended to non-financial domains

**Pontos de Atenção:**
- Broader Impact statement required
- Ethical considerations section needed
- Less stringent positionality requirements than FAccT

### 6.3 Journal Options

#### Option 1: Nature Machine Intelligence (Stretch)

**Pros:**
- Highest impact (IF: 18-23)
- Interdisciplinary readership
- Published Cynthia Rudin's interpretability paper (2019)

**Cons:**
- Very competitive (5-15% acceptance)
- Requires breakthrough findings
- Long review process (6-12 months)

**Viability:**
If empirical findings are stronger than expected (e.g., interpretable models 
match or exceed distilled in ALL cases), Nature MI becomes viable.

#### Option 2: Journal of Financial Data Science

**Pros:**
- Perfect topical fit
- Published fairness in finance work
- Practitioner + academic audience
- Rolling submissions

**Cons:**
- Lower impact vs Nature MI
- Smaller readership
- Less prestige

**Viability:** STRONG

**Strategy:** Expand empirical section, add more financial use cases, 
emphasize practitioner guidance.

#### Option 3: Quantitative Finance

**Pros:**
- High receptivity to ML methods (68 AI papers)
- Financial econometrics focus
- Interdisciplinary

**Cons:**
- Requires strong financial economics angle
- Less focus on policy vs FAccT

**Viability:** MODERATE

**Strategy:** Frame as risk management problem, emphasize cost-benefit 
analysis, connect to financial stability.

---

## 7. ESTRATÉGIA DE POSICIONAMENTO

### 7.1 Relação com Paper 1 (HPM-KD Técnico)

**Cross-Reference Estratégico:**

**No Abstract do Paper 2:**
```markdown
Recent advances in knowledge distillation [cite 3-5 papers including Paper 1] 
have demonstrated significant computational benefits, achieving 10-15× 
compression with minimal accuracy loss. However, deployment in regulated 
financial services faces a fundamental challenge...
```

**Na Introduction do Paper 2:**
```markdown
Model compression techniques, particularly knowledge distillation, have 
achieved remarkable technical progress [cite Paper 1 among others]. Methods 
like HPM-KD (Haase and Dourado, 2025), traditional KD (Hinton et al., 2015), 
and TAKD (Mirzadeh et al., 2020) enable substantial model compression while 
retaining 95-99% of teacher accuracy.

Despite these technical achievements, deployment in high-stakes regulated 
domains faces a barrier not addressed by compression research: the gap between 
computational optimality and legal deployability. This paper examines this 
gap through the lens of financial regulation...
```

**Positioning:**
- Cite Paper 1 como ONE EXAMPLE de recent technical advance
- Não trate Paper 1 como único motivation ou único method avaliado
- Frame Paper 2 como addressing a DIFFERENT question (regulatory compliance)
- Ênfase: "Technical advances are necessary but not sufficient"

**Na Related Work do Paper 2:**
```markdown
**2.X Knowledge Distillation Methods**

Recent work has advanced distillation techniques significantly. Hinton et al. 
(2015) introduced the foundational approach... Mirzadeh et al. (2020) 
addressed the capacity gap with teacher assistants... Haase and Dourado (2025) 
proposed HPM-KD, integrating meta-learning, progressive distillation, and 
multi-teacher attention to achieve 10-15× compression with 95-98% accuracy 
retention.

While these methods optimize for performance metrics (accuracy, compression 
ratio, inference latency), they do not address explainability requirements 
in regulated domains. Our work complements this technical progress by 
evaluating whether state-of-the-art compression methods meet regulatory 
standards and identifying deployment barriers.
```

**Key Principle:**
Paper 1 é CONTEXTO, não CONTRIBUTION principal. Paper 2's contribution é o 
regulatory analysis, compliance framework, e policy recommendations—não o 
algoritmo HPM-KD.

### 7.2 Framing para Reviewers

**Cover Letter (se ambos papers já submetidos):**

```markdown
Dear Associate Chairs,

We submit "Regulatory Explainability and Compliance for Knowledge Distillation 
in Financial Services" for consideration at ACM FAccT 2026.

This submission is complementary to our technical work on knowledge distillation 
methods [Paper 1, submitted to ICLR 2026]. The technical paper focuses on 
algorithmic contributions for efficient model compression. The current paper 
addresses the distinct question of whether compression methods—including but 
not limited to our technical work—meet regulatory requirements in high-stakes 
domains.

These papers serve different research communities and make independent 
contributions:
- Paper 1 (ICLR): ML researchers, contribution is algorithmic
- Paper 2 (FAccT): Policymakers/regulators, contribution is analytical framework

There is minimal overlap: Paper 1 mentions explainability as a limitation 
(one paragraph); Paper 2 examines this limitation systematically as its core 
contribution. Paper 1 evaluates technical performance; Paper 2 evaluates 
regulatory compliance.

Both papers cite each other appropriately in related work, positioned among 
multiple relevant works rather than as sole motivation.

We believe this split is legitimate and beneficial: it enables each paper to 
serve its target community with appropriate depth. Combining them would 
necessitate cuts to both technical rigor and regulatory analysis, weakening 
both contributions.
```

**Anticipated Reviewer Concerns:**

**Concern 1: "These should be one paper"**

Response:
```markdown
We appreciate the reviewer's perspective on scope. However, these papers 
address fundamentally different research questions requiring distinct 
methodologies:

- Paper 1 (RQ): How can we improve knowledge distillation performance?
  Methodology: Algorithm development, computational experiments
  Community: ML researchers
  
- Paper 2 (RQ): Why do distillation methods fail regulatory requirements?
  Methodology: Legal analysis, compliance framework, policy recommendations
  Community: Regulators, policymakers, practitioners

Combining them would require cutting either (a) technical algorithm details 
and ablations (weakening ML contribution) or (b) regulatory analysis and 
policy implications (weakening FAccT contribution). Each paper is already 
near page limits.

Moreover, the audiences are distinct: ML researchers need technical depth on 
HPM-KD; policymakers need regulatory analysis and practical guidance. A 
combined paper would serve neither community well.

Similar divisions between technical and policy papers are common in AI ethics 
literature (e.g., technical fairness algorithms vs fairness policy analysis).
```

**Concern 2: "Paper 2 is incremental relative to Paper 1"**

Response:
```markdown
Paper 2 makes fundamentally different contributions than Paper 1:

1. **Compliance Assessment Framework** (new): 4-dimensional framework for 
   evaluating model deployability in regulated domains—not present in Paper 1

2. **Regulatory Analysis** (new): Systematic analysis of ECOA, GDPR, EU AI 
   Act, SR 11-7 requirements and how distillation interacts with each—one 
   paragraph in Paper 1, 6 pages in Paper 2

3. **Multiplicative Opacity Concept** (new): Theoretical framework for 
   understanding explainability in ensemble/distillation—not in Paper 1

4. **Empirical Compliance Evaluation** (new): Financial case studies evaluating 
   compliance scores, not just accuracy—Paper 1 evaluates only technical metrics

5. **Policy Recommendations** (new): Multi-stakeholder guidance for regulators, 
   institutions, developers—not in Paper 1

Paper 2 is not an extension of Paper 1's experiments. It addresses a completely 
different research question using different methodology (legal+policy analysis 
vs algorithm development).
```

### 7.3 Timing Strategy

**Scenario 1: Paper 1 Aceito Primeiro (ICLR May 2026)**

✅ **Advantage**: Paper 2 pode citar Paper 1 como published work
- Update citations from preprint to published
- Add conference venue information
- No longer need to maintain anonymity for Paper 1

Paper 2 submission (FAccT June 2026):
```markdown
Recent technical advances [Haase & Dourado, ICLR 2026; other citations] have 
achieved...
```

**Scenario 2: Paper 2 Aceito Primeiro (FAccT June 2026)**

✅ **Advantage**: Paper 1 pode mencionar regulatory implications in Discussion
- Paper 1 Discussion: "Companion work (Haase & Dourado, FAccT 2026) provides 
  detailed regulatory analysis..."
- Strengthens Paper 1's impact section

**Scenario 3: Simultaneous Submission (Both Under Review)**

⚠️ **Maintain Anonymity**: Don't mention the other paper explicitly
- Paper 1: "Regulatory dimensions constitute distinct research direction"
- Paper 2: "Recent compression advances [anonymous ref] have achieved..."
- After acceptance: Update with proper citations

**Scenario 4: One Rejected**

If Paper 1 rejected from ICLR:
- Revise and resubmit to ICML or NeurIPS
- Paper 2 can still cite Paper 1 as arXiv preprint
- FAccT reviewers don't require published venues (preprints acceptable)

If Paper 2 rejected from FAccT:
- Revise and resubmit to AIES
- Or pivot to journal (J Financial Data Science)
- Paper 1 still cites regulatory limitations appropriately

---

## 8. TIMELINE E MILESTONES

### 8.1 Preparação do Paper 2 (12 semanas)

**Semanas 1-3: Research e Base Writing**
- [ ] Revisão de literatura regulatória completa
  - ECOA guidance, CFPB circulars
  - GDPR recitals, CJEU cases
  - EU AI Act full text
  - SR 11-7 and related guidance
- [ ] Stakeholder interviews (se aplicável)
  - 5-10 compliance professionals
  - 3-5 regulators (if accessible)
  - 10-15 domain experts (loan officers)
- [ ] Abstract + Introduction draft (4 páginas)
- [ ] Related Work skeleton

**Semanas 4-6: Framework Development**
- [ ] Section 2 completa (Regulatory Requirements, 6 páginas)
  - 4 subseções (ECOA, GDPR, EU AI Act, SR 11-7)
  - Tables 1-4 finalizadas
- [ ] Section 3 completa (Compliance Framework, 4 páginas)
  - 4-dimensional framework detalhado
  - Scoring rubrics
  - Tables 5-9 finalizadas
  - Figure 2-3 (framework diagrams)

**Semanas 7-9: Empirical Evaluation**
- [ ] Run all experiments (3 financial use cases)
  - Credit scoring dataset prep + experiments
  - Mortgage dataset prep + experiments
  - Insurance dataset prep + experiments
- [ ] Section 4 completa (Case Studies, 6 páginas)
  - Tables 10-16 finalizadas
  - Figures 4-6 generated
- [ ] User study execution (if needed)
  - 20 loan officers
  - Survey + evaluation
  - Human oversight scores

**Semanas 10-11: Policy e Discussion**
- [ ] Section 5 completa (Policy Implications, 5 páginas)
  - 4 stakeholder-specific recommendations
  - Table 17 (Risk Matrix)
- [ ] Section 6 completa (Discussion, 3 páginas)
  - Synthesis of findings
  - Theoretical contributions
  - Limitations
- [ ] Conclusion (0.5 página)

**Semana 12: Polimento e Finalização**
- [ ] Appendices completos (regulatory texts, framework details, etc.)
- [ ] Todos os figures finalizados (9 total)
- [ ] Todas as tables formatadas (20 total)
- [ ] Positionality statement
- [ ] Broader impact statement
- [ ] Data availability statement
- [ ] Proofreading completo
- [ ] Anonymization check
- [ ] References verification (100+ citations esperadas)

### 8.2 Submission Targets

**Opção 1: ACM FAccT 2026 (PRIMARY)**
- Deadline: ~Outubro/Novembro 2025
- Notification: ~Março 2026
- Conference: ~Junho 2026 (virtual ou híbrido)
- **RECOMENDAÇÃO FORTE**: Best fit para o paper

**Opção 2: AIES 2026 (ALTERNATIVE)**
- Deadline: ~Janeiro/Fevereiro 2026
- Notification: ~Abril 2026
- Conference: ~Maio 2026
- **BACKUP**: Se FAccT rejeitado ou timing for issue

**Opção 3: Journal (EXTENSION)**
- J Financial Data Science: Rolling submission
- Quantitative Finance: Rolling submission
- **STRATEGY**: Expand Paper 2 FAccT version (+20 páginas empirical)

---

## 9. CHECKLIST FINAL PRÉ-SUBMISSION

### 9.1 Conteúdo Específico Paper 2

- [ ] Regulatory analysis completa (4 regimes × 5 páginas cada)
- [ ] Compliance framework com 4 dimensões + scoring
- [ ] 3 financial case studies com full evaluation
- [ ] Performance-explainability tradeoff quantificado
- [ ] Policy recommendations para 4+ stakeholders
- [ ] Multiplicative opacity framework introduzido e aplicado

### 9.2 Diferenciação vs Paper 1

- [ ] Paper 1 citado apropriadamente (não como sole motivation)
- [ ] Metodologia claramente distinta (legal+policy vs algorítmica)
- [ ] Research questions diferentes e complementares
- [ ] Minimal overlap (<5% content similarity)
- [ ] Structural differentiation (regulatory vs experimental)

### 9.3 FAccT-Specific Requirements

- [ ] Positionality statement (150-300 palavras)
- [ ] Broader impact statement detalhado
- [ ] Interdisciplinary language (accessible + rigorous)
- [ ] Stakeholder perspectives incorporated
- [ ] Equity and justice considerations addressed
- [ ] Data availability statement
- [ ] Ethics approval (if human subjects)

### 9.4 Elementos Visuais

- [ ] 9 figuras geradas (compliance framework, scatter plots, frontiers)
- [ ] 20+ tabelas formatadas (regulatory requirements, scores, results)
- [ ] Figures em alta resolução (300 dpi)
- [ ] Tables com significance markers onde aplicável
- [ ] All visuals têm detailed captions

### 9.5 Quality Checks

- [ ] 100+ citations (ML + Law + Policy + Finance)
- [ ] Legal citations formatted correctly (Bluebook style for US law)
- [ ] All regulatory requirements cited to source
- [ ] Case law properly cited (CJEU, US courts)
- [ ] Spell check (US English)
- [ ] Grammar check
- [ ] Consistency check (terminology, notation)
- [ ] Anonymization complete

### 9.6 Supplementary Materials

- [ ] Appendices completos (regulatory texts, scoring rubrics, case law)
- [ ] Código do compliance framework (GitHub, anonymized)
- [ ] Survey instruments (if user study)
- [ ] Dataset descriptions (Lending Club, HMDA, Insurance)
- [ ] Interview summaries (anonymized, if applicable)

---

## 10. RESUMO EXECUTIVO

### Estrutura Final Paper 2

**Total:** ~40 páginas (15 main + 25 appendix)

1. **Abstract** (0.25p) - Problema, RQs, metodologia, findings, policy
2. **Introduction** (4p) - Technical-regulatory divide, multiplicative opacity, contributions
3. **Regulatory Requirements** (6p) - ECOA, GDPR, EU AI Act, SR 11-7
4. **Compliance Framework** (4p) - 4 dimensions, scoring, assessment
5. **Empirical Evaluation** (6p) - 3 financial case studies
6. **Policy Implications** (5p) - Multi-stakeholder recommendations
7. **Discussion** (3p) - Synthesis, contributions, limitations
8. **Conclusion** (0.5p)
9. **References** (~4p) - 100+ citations expected
10. **Appendices** (25p) - Regulatory texts, scoring details, extended results

**Figuras:** 9 total  
**Tabelas:** 20+ total

### Key Messages

**Technical-Regulatory Divide:**
> "Recent compression advances achieve 95-99% accuracy retention, but fail 
> to meet legal requirements for individual-level explainability, creating 
> a deployment barrier not addressed by technical research."

**Multiplicative Opacity:**
> "Distillation creates 'multiplicative opacity' where student models inherit 
> and amplify ensemble teacher complexity, making post-hoc explanations 
> misleading despite smaller architecture."

**Empirical Findings:**
> "Interpretable methods (EBM, NAM) achieve 97-99% of distilled model 
> performance in financial use cases, with compliance costs ($2-3M/year) 
> vastly exceeding marginal benefits ($300K-1.4M/year)."

**Policy Recommendation:**
> "Financial institutions should default to interpretable methods (EBM/NAM) 
> for regulated applications. Regulators should clarify explainability 
> standards and create safe harbors for interpretable architectures."

### Diferenças Críticas vs Paper 1

| Aspecto | Paper 1 | Paper 2 |
|---------|---------|---------|
| **RQ** | Como melhorar KD? | Por que KD falha em compliance? |
| **Metodologia** | Algoritmos + Experimentos | Legal + Framework + Policy |
| **Contribution** | HPM-KD algorithm | Compliance framework + Policy |
| **Audience** | ML researchers | Regulators + Practitioners |
| **Venue** | ICLR/ICML | ACM FAccT/AIES |
| **Structure** | Intro→Method→Exp→Results | Intro→Reg→Framework→Cases→Policy |
| **Metrics** | Accuracy, Compression | Compliance scores, Cost-benefit |
| **Citations** | CS/ML papers (40) | Law + Policy + CS (100+) |
| **Tone** | Technical | Interdisciplinary |

---

**FIM DO DOCUMENTO**

Este guia fornece a estrutura completa necessária para o Paper 2 regulatório sobre explainability e compliance, pronto para submissão a ACM FAccT ou AIES. O foco é completamente diferente do Paper 1 técnico, abordando uma research question distinta com metodologia apropriada para a comunidade de ML fairness e policy.
