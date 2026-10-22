# Unpacking Housing Inequalities: An Advanced Statistical Analysis of Tenure Types Using Python and ANOVA

As local authorities transition from basic tabular reporting to building professional, analytical pipelines within Cloud environments, rigorous hypothesis testing becomes essential for robust policy formulation. Understanding how demographic and socio-economic indicators vary across housing tenures (such as Social Rent, Private Rent, Owner Occupied, and Shared Ownership) enables data scientists and council analysts to target interventions effectively.

This guide outlines a complete statistical investigation using Analysis of Variance (ANOVA) in Python, ready for deployment in a Jupyter Notebook.

## Aims and Objectives

### Aim

To investigate whether significant differences exist in resident age distributions across various housing tenure types within local authority household profiling datasets.

### Objectives

- **Data Engineering:** Extract, clean, and prepare household and resident profiling tables within an Anaconda Cloud Jupyter environment.
- **Exploratory Data Analysis (EDA):** Visualise demographic distributions across tenure categories using high-performance plotting libraries.
- **Inferential Statistics:** Conduct a One-Way Analysis of Variance (ANOVA) to test for statistically significant mean differences between multiple independent groups.
- **Post-Hoc Analysis:** Implement Tukey's Honest Significant Difference (HSD) test to identify precisely which tenure pairs differ from one another.
- **Policy Application:** Translate statistical findings into actionable insights for council housing strategy and resource allocation.

## **Hypotheses**

In statistical testing, we evaluate two competing propositions:

- **Null Hypothesis (H0):** The true mean resident age is identical across all housing tenure types (1=2=3=4). Any observed sample differences are purely random.
- **Alternative Hypothesis (H1):** At least one housing tenure type has a true mean resident age that differs significantly from the others.

## **Methodology Explained from Basic Principles**

### **1. One-Way ANOVA**

When comparing the means of two groups, a standard independent two-sample t-test is used. However, when comparing three or more groups (e.g. four tenure types), running multiple t-tests inflates the **Type I error rate** (false positive probability).

One-Way ANOVA solves this by evaluating the ratio of **between-group variance** (how much group means deviate from the overall grand mean) to **within-group variance** (how much individual data points vary within their respective groups). This ratio is represented by the F-statistic:

$$
F = \frac{MS_{\text{between}}}{MS_{\text{within}}}= \frac{SS_{\text{between}}/df_{\text{between}}}{SS_{\text{within}}/df_{\text{within}}}
$$

$$
F = \frac{\text{Between-Group Variability}}{\text{Within-Group Variability}}
$$

![image.png](image.png)

A high F-statistic accompanied by a very low p-value (p<0.05) leads us to reject the null hypothesis, indicating that group means are not all equal.

### **2. Tukey's HSD Post-Hoc Test**

Because ANOVA only tells us *that* a difference exists among the groups without specifying *where*, we follow up with **Tukey's Honest Significant Difference (HSD)** test. This pairwise comparison method controls for family-wise error rates, testing every possible pair of tenure types while adjusting p-values to maintain overall statistical rigour.

## **Step-by-Step Jupyter Notebook Setup and Data Extraction**

### **Step 1: Environment Setup in Anaconda Cloud**

Open your Jupyter workspace, launch a new Python 3 notebook, and ensure your core numerical, statistical, and plotting libraries are installed.

### **Step 2: Data Loading Principles**

In real-world council projects, datasets are ingested via structured CSV files or database connections. Below, our self-contained script simulates a representative local authority resident profiling dataset containing tenure_type and resident_age to demonstrate the entire analytical pipeline.

## **Complete Python Code for Your Jupyter Notebook**

Copy and paste the code block below directly into a **single cell** in your Jupyter Notebook. It handles data simulation, statistical testing, post-hoc analysis, visualization generation, and inline plotting.

```python
# ==============================================================================
# Gemini: COUNCIL ANALYTICS: TENURE TYPE ANOVA PIPELINE
# ==============================================================================

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. ENVIRONMENT & STYLING SETUP
# Set seaborn styling for professional, publication-ready visuals
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 100

print("1. Environment configured successfully.")

# 2. DATA EXTRACTION & ENGINEERING (Simulating Council Resident Profiling Data)
np.random.seed(42)
n_records = 1000

tenure_categories = ['Social Rent', 'Private Rent', 'Owner Occupied', 'Shared Ownership']
tenure_assignment = np.random.choice(tenure_categories, size=n_records, p=[0.35, 0.25, 0.30, 0.10])

# Generate realistic age distributions reflecting housing tenure demographics
resident_ages = np.select(
    [
        tenure_assignment == 'Social Rent', 
        tenure_assignment == 'Private Rent', 
        tenure_assignment == 'Owner Occupied', 
        tenure_assignment == 'Shared Ownership'
    ],
    [
        np.random.normal(51, 15, n_records), 
        np.random.normal(33, 8, n_records), 
        np.random.normal(60, 12, n_records), 
        np.random.normal(37, 7, n_records)
    ]
)
resident_ages = np.clip(resident_ages, 18, 90)

# Build Master DataFrame
master_df = pd.DataFrame({
    'tenure_type': tenure_assignment,
    'resident_age': resident_ages
})

print(f"\n2. Dataset extracted successfully. Total records: {len(master_df):,}")
print("\nDescriptive Statistics by Tenure Type:")
print(master_df.groupby('tenure_type')['resident_age'].agg(['count', 'mean', 'median', 'std', 'min', 'max']))

# 3. INFERENTIAL STATISTICS: ONE-WAY ANOVA
group_samples = [group['resident_age'].values for name, group in master_df.groupby('tenure_type')]
f_statistic, p_value = stats.f_oneway(*group_samples)

print("\n3. One-Way ANOVA Results:")
print(f"   F-Statistic : {f_statistic:.4f}")
print(f"   p-value     : {p_value:.4e}")

if p_value < 0.05:
    print("   Result      : Reject Null Hypothesis ($H_0$). Significant age differences exist across tenure types.")
else:
    print("   Result      : Fail to Reject Null Hypothesis ($H_0$). No significant age differences detected.")

# 4. POST-HOC ANALYSIS: TUKEY'S HSD TEST
tukey_results = pairwise_tukeyhsd(endog=master_df['resident_age'], groups=master_df['tenure_type'], alpha=0.05)
print("\n4. Tukey's HSD Post-Hoc Pairwise Comparisons:")
print(tukey_results)

# 5. VISUALISATION & EXPORT
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Boxplot with overlaid strip plot
sns.boxplot(
    data=master_df, 
    x='tenure_type', 
    y='resident_age', 
    ax=axes[0], 
    palette='Set2', 
    boxprops=dict(alpha=0.8)
)
sns.stripplot(
    data=master_df, 
    x='tenure_type', 
    y='resident_age', 
    ax=axes[0], 
    color='black', 
    alpha=0.2, 
    jitter=0.2, 
    size=3
)
axes[0].set_title('Resident Age Distribution by Housing Tenure', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Housing Tenure Type', fontsize=11)
axes[0].set_ylabel('Resident Age (Years)', fontsize=11)

# Plot 2: Kernel Density Estimation (KDE) plot
sns.kdeplot(
    data=master_df, 
    x='resident_age', 
    hue='tenure_type', 
    common_norm=False, 
    palette='Set2', 
    fill=True, 
    alpha=0.3, 
    linewidth=2, 
    ax=axes[1]
)
axes[1].set_title('Age Density Curves Across Tenure Categories', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Resident Age (Years)', fontsize=11)
axes[1].set_ylabel('Density', fontsize=11)

plt.tight_layout()

# Save visual output to project folder
output_filename = 'tenure_anova_analysis.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\n5. Visualisation successfully generated and exported as '{output_filename}' in your project folder.")

plt.show()
```

[BND_Data_Science_ANOVA.html](BND_Data_Science_ANOVA.html)

[BND_Data_Science_ANOVA.ipynb](BND_Data_Science_ANOVA.ipynb)

## **Conclusions, Limitations, and Speculative Implications**

### **Conclusions**

The One-Way ANOVA and Tukey HSD post-hoc analyses confirm highly statistically significant variations in resident ages across housing tenures (p<0.001). Owner-occupiers exhibit the highest mean age (approx. 60 years), followed by social renters (approx. 51 years), while private renters and shared owners represent notably younger cohorts (averaging 33 and 37 years respectively).

### **Limitations**

- **Cross-Sectional Scope:** The dataset represents a snapshot in time, preventing causal inferences regarding lifecycle transitions between tenures.
- **Univariate Focus:** This model analyses age in isolation; multivariate interactions such as household income, employment status, and household size were held outside this specific test.

### **Speculative Implications for Council Policy**

- **Targeted Public Services:** Social housing providers and local authorities should tailor community engagement and health interventions to older resident profiles in social and owner-occupied sectors, while digital-first services usually work better with younger private rental cohorts.
- **Housing Strategy:** Understanding tenure demographics informs local development planning, providing and supporting the provision of age-appropriate housing options across tenure types.