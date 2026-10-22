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

!image.png

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

- **Housing Strategy:** Understanding tenure demographics informs local development planning, providing and supporting the provision of age-appropriate housing options across tenure types.
