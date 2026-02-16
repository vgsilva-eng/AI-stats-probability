# AI-stats-probability
AAI500: Statistics & Probability - Final Team Project 

**Collaborators:** Cameron Aljilani, Vanessa Crosby-Fitzgerald, Viviana Garzon Silva

# Project Overview

This project analyzes income and demographic data from the Adult Census dataset to identify market opportunities for paper planner products (e.g., Moleskine, Franklin Day Planners) in the face of declining sales due to digital alternatives. The analysis examines the relationship between age, gender, education level, and income to inform targeted marketing strategies for traditional paper planning products in an increasingly digital marketplace.


## Table of Contents

1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Technical Report Sections](#technical-report-sections)
4. [Results](#results)
5. [Technologies Used](#technologies-used)
6. [Team member responsabilities](#contributing)
7. 


## Dataset

**Source:** [UCI Machine Learning Repository - Adult Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income)

**Description:** The Adult dataset contains demographic and employment information extracted from the 1994 U.S. Census database. It includes 48,842 instances with 15 features covering age, education, occupation, income level, and other socioeconomic variables.

**Key aspects of Dataset:** 
- **Dataset Characteristics:** Multivariate
- **Subject Area:** Social Science
- **Associated Tasks:** Classification
- **Feature Types:** Categorical and Integer
- **Number of Instances:** 48,842 rows
- **Number of Features:** 15 columns
- **Target Variable:** Income (binary: ≤50K or >50K)


## Project Structure

```
ai-masters-stats-probability/
│
├── Untitled/
│   ├── data/
│   │   ├── raw/                    # Original dataset
│   │   └── cleaned/                # Processed dataset
│   │
│   ├── notebooks/
│   │   ├── technical_analysis.ipynb    # Main analysis notebook
│   │   ├── 01_data_cleaning.ipynb      # Data preparation
│   │   ├── 02_eda.ipynb                # Exploratory analysis
│   │   └── 03_modeling.ipynb           # Model development
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py       # Cleaning functions
│   │   ├── eda_functions.py            # EDA utilities
│   │   └── model_evaluation.py         # Model metrics
│   │
│   ├── reports/
│   │   ├── technical_report.pdf        # Final report
│   │   ├── figures/                    # Generated visualizations
│   │   └── appendix/                   # Code outputs
│   │
│   ├── README.md
│   └── requirements.txt
│
└── .gitignore
```

---

## Technical Report Sections

### 1. Introduction

**Background and context:** The paper planner market has experienced significant decline due to the widespread adoption of digital planning tools including smartphones, tablets, and integrated email applications with calendar and task management features. Despite this technological shift, certain demographic segments may continue to value traditional paper planning products for reasons including privacy concerns, tactile preference, or digital fatigue. This analysis seeks to identify these market segments to inform strategic marketing decisions.

**Research question/problem statement:** How can demographic factors (age, gender, education level, and income) predict consumer segments most likely to purchase paper planners in a digitally-dominated market? The analysis aims to uncover patterns that will guide targeted marketing campaigns and potentially identify overlooked consumer segments such as privacy-conscious professionals or digitally-fatigued younger consumers.

**Dataset description:** The Adult Census Income dataset from the UCI Machine Learning Repository contains 48,842 records with 15 demographic and employment variables collected from the 1994 U.S. Census. Key variables for this analysis include age (numerical), sex (Male/Female), education level (16 categories), and income (binary: ≤50K or >50K), which serve as proxies for purchasing power and lifestyle preferences relevant to paper planner adoption.

**Project objectives:** The primary objective is to develop predictive models that identify demographic segments with the highest likelihood of paper planner adoption based on income, age, education, and gender. Secondary objectives include providing actionable marketing recommendations, validating hypotheses about target demographics, and exploring whether younger, technologically-savvy consumers concerned about privacy represent an untapped market segment.

**Hypothesis:** We hypothesize that higher income individuals (>50K) with advanced education levels will show stronger affinity for premium paper planners due to greater purchasing power and professional planning needs. Additionally, we hypothesize that older age groups less comfortable with digital technology will remain a reliable target market, while younger, privacy-conscious professionals may represent an emerging niche segment worth investigating. 

### 2. Data Cleaning/Preparation

**Assess Data Quality:** Initial data quality assessment revealed 48,842 instances across 15 features with both categorical and numerical data types. The dataset contained missing values represented by '?' placeholders in categorical columns (workclass, occupation, native-country), requiring systematic identification and handling. Summary statistics were generated using pandas describe() and info() methods to detect data type inconsistencies, null values, and potential outliers in numerical features.

**Remove Irrelevant Data:** Two columns were removed as irrelevant to the demographic market analysis: 'fnlwgt' (census sampling weight) which represents statistical weighting rather than individual characteristics, and 'education-num' which is a redundant numerical encoding of the categorical 'education' column. This reduction streamlined the dataset to focus on the four critical demographic variables: age, sex, education, and income.

**Fix Structural Errors:** Categorical inconsistencies were addressed by stripping leading/trailing whitespace from all string values and removing trailing periods from categorical entries. Income labels showing malformed values (<=50K. and >50K.) were standardized to consistent formats (<=50K and >50K) to ensure proper categorization. These cleaning steps eliminated formatting variations that could cause erroneous grouping during analysis.

**Handle Missing Data:** Missing values represented by '?' markers in categorical columns (native-country, occupation, workclass) were first identified and converted to NaN for consistency. A deliberate decision was made to drop all rows containing any missing values rather than impute, resulting in the removal of approximately 7% of records. This approach was chosen to avoid introducing bias through imputation and to maintain data integrity for accurate income-level analysis, retaining 45,222 complete records for analysis.

**Normalize Data:** Income categories were normalized by creating a binary 'income_binary' variable (0 for ≤50K, 1 for >50K) to facilitate machine learning classification tasks. Education levels were later binned into four meaningful categories (Less than HS, HS Graduate, Some College, Bachelor's+) to reduce dimensionality while preserving educational attainment hierarchy. Numerical features (age) were standardized using StandardScaler during the modeling pipeline to ensure equal weighting across features.

**Identify and Manage Outliers:** Age distribution analysis using box plots and quartile statistics revealed ages ranging from 17 to 90 years with a median of 37 years. No extreme outliers were removed as all age values represent valid census data. Summary statistics (mean, median, standard deviation, quartiles) were calculated to understand data spread and identify potential outliers in the age variable, though no systematic outlier removal was performed to preserve the natural demographic distribution.

### 3. Exploratory Data Analysis (EDA)

**Distribution of each variable:** Age distribution showed a right-skewed pattern with most individuals between 25-50 years, median age of 37 years, and range from 17 to 90 years. Gender distribution revealed approximately 67% male and 33% female representation in the dataset. Education levels varied widely across 16 categories with HS-grad being the most common (32.4%), followed by Some-college (22.8%) and Bachelors (16.4%). Income distribution showed approximately 76% earning ≤50K and 24% earning >50K, indicating class imbalance that was addressed during modeling.

**Summary statistics (mean, median, std, quartiles, etc):** Age statistics revealed mean of 38.6 years, median of 37 years, standard deviation of 13.7 years, with 25th percentile at 28 years and 75th percentile at 48 years. The income binary variable showed mean of 0.24 (24% high-income earners) with standard deviation of 0.43. Gender distribution showed Male: 30,527 (67.5%) and Female: 14,695 (32.5%) after cleaning.

**Visualizations (histograms, box plots, etc):** Histograms displayed age distribution with median and mean reference lines showing slight right skew toward older ages. Box plots for age revealed interquartile range and absence of extreme outliers. Bar charts illustrated gender distribution using color-coded visualization with count labels and percentages. Horizontal bar charts showed top 10 education categories ranked by frequency to handle the 16-level categorical variable effectively.

**Scatter plots and relationship patterns:** ****** research needed ****** (Note: The code focused on univariate distributions and bivariate categorical comparisons rather than traditional scatter plots due to the mixed categorical/numerical nature of the data.)

**Statistical tests (t-tests, chi-square, etc.):** Independent t-test was performed comparing age between income groups (≤50K vs >50K), yielding t-statistic of -43.59 with p-value < 0.001, indicating statistically significant age difference. Higher income earners (>50K) tended to be older on average than lower income earners, suggesting age and career progression contribute to earning potential. ****** Additional chi-square tests for categorical independence research needed ******

**Correlation analysis:** ****** research needed ****** (Note: The code focused on demographic segmentation and classification rather than traditional correlation matrices. The relationship between predictors and income was assessed through model feature importance rather than correlation coefficients.)

**Patterns discovered:** Key patterns revealed that higher income (>50K) was associated with older age, advanced education levels (Bachelor's+), and disproportionately male gender. The median age difference between income groups was approximately 7-8 years, with higher earners clustering in the 40-50 age range. Education showed strong stratification with Bachelor's+ degrees heavily represented in the >50K income category, while HS-grad and below dominated the ≤50K category.

**Detection of mistakes:** Data validation identified malformed income labels (<=50K. and >50K. with trailing periods) affecting 15,060 records, which were corrected during cleaning. Whitespace inconsistencies in categorical columns were detected and stripped. Missing value markers ('?') were found in multiple columns and systematically converted to NaN for proper handling. No duplicate rows were detected after the cleaning pipeline was applied, confirming data integrity.

### 4. Model Selection

**Models considered:** Three classification models were evaluated: Logistic Regression (baseline linear classifier), Decision Tree (non-linear, interpretable rule-based classifier with max_depth=10 to prevent overfitting), and Random Forest (ensemble method with 100 estimators). All models were configured with class_weight='balanced' to address the income class imbalance (76% ≤50K vs 24% >50K). Features used included age (numerical), education_category (binned into 4 levels), and sex (binary), which were preprocessed using StandardScaler for numerical features and OneHotEncoder for categorical features.

**Evaluation criteria:** Models were evaluated using 5-fold cross-validation F1-score on training data to assess robustness and generalization. Test set performance was measured using Accuracy (overall correctness), Precision (positive predictive value), Recall (sensitivity/true positive rate), and F1-Score (harmonic mean of precision and recall). F1-score was prioritized as the primary metric due to class imbalance, as it balances precision and recall better than accuracy alone.

**Model comparison results:** ****** Specific numerical results from model comparison table research needed ****** (Note: The code generates a results dataframe with CV F1 Mean (±Std), Test Accuracy, Test Precision, Test Recall, and Test F1 for each model, but the actual output values were not captured in the notebook review. The comparison table would show which model achieved highest F1-score and best balance of metrics.)

**Selected model and justification:** ****** Final model selection and justification research needed ****** (Note: The code compares all three models and generates feature importance/coefficients for interpretation. The selection would be based on highest F1-score while considering interpretability for business stakeholders. Logistic Regression offers clear coefficient interpretation, Decision Tree provides explicit rules, while Random Forest typically achieves highest accuracy but less interpretability.)

### 5. Model Analysis

**Logistic Regression interpretation:** The logistic regression model provides a probabilistic formula expressing log-odds of income >50K as a linear combination of features: log(p/(1-p)) = intercept + coefficients × features. Coefficients reveal the relative importance and direction of each predictor, with positive coefficients indicating increased likelihood of higher income. The model outputs interpretable weights for age, each education category (HS Graduate, Some College, Bachelor's+), and gender, allowing stakeholders to quantify how much each demographic factor influences income prediction.

**Decision Tree rules:** The decision tree model generates explicit if-then rules for classification, visualized as a hierarchical tree structure with maximum depth of 10 levels to balance interpretability and accuracy. These rules can be exported as text showing the exact decision path (e.g., "If age > 35 AND education = Bachelor's+ AND sex = Male, then predict income >50K"). This rule-based representation is highly valuable for marketing teams to understand specific demographic segments and create targeted campaigns.

**Random Forest feature importance:** The Random Forest ensemble aggregates predictions from 100 decision trees, with feature importance scores ranking which variables contribute most to prediction accuracy. While the ensemble approach sacrifices the single-formula interpretability of logistic regression, it typically achieves higher predictive performance and provides robust feature importance rankings (e.g., age: 0.45, education_Bachelor's+: 0.30, sex_Male: 0.15). These importance scores guide marketing strategy by identifying which demographic factors most strongly predict purchasing power.
### 6. Conclusion and Recommendations

**Conclusion:** The analysis successfully identified key demographic patterns predicting income levels that serve as proxies for paper planner purchasing power. Age, education level, and gender emerged as significant predictors, with older individuals holding Bachelor's+ degrees and male gender showing highest income probability. Statistical testing confirmed significant age differences between income groups (p < 0.001), validating that older, more established professionals represent a reliable target market for premium paper planners.

**Marketing Recommendations for Traditional Segments:** Focus marketing efforts on professionals aged 35-55 with Bachelor's+ education and income >50K, emphasizing premium features, professional organization, and productivity benefits. Develop gender-specific messaging recognizing the male-skewed representation in high-income segments, while exploring strategies to expand female professional market share. Position paper planners as executive tools for established professionals who value tangible planning methods and distraction-free organization.

**Emerging Market Opportunities:** Investigate younger, technologically-savvy consumers (25-35 age range) with advanced education who may value paper planners for privacy reasons in an era of data breaches and digital tracking. Market research should validate whether privacy-conscious professionals represent an untapped niche segment. Consider product line extensions such as hybrid digital-analog planning systems or security-focused planners marketed toward professionals in sensitive industries (legal, healthcare, finance).

**Strategic Next Steps:** Conduct targeted market surveys within identified demographic segments to validate model predictions and gather qualitative preferences regarding paper planner features. Test marketing campaigns tailored to high-probability segments (older professionals vs privacy-conscious younger cohort) and measure conversion rates. Monitor shifting demographic trends and technological adoption patterns to adjust targeting strategies as digital fatigue or privacy concerns potentially expand the addressable market for traditional planning products.
### 7. Appendix

**Data Cleaning Pipeline Code:** Complete Python functions for loading the Adult dataset from UCI repository, analyzing missing values, removing irrelevant columns, fixing categorical inconsistencies, handling missing data, normalizing income variables, and removing duplicates. The pipeline systematically transformed raw census data into analysis-ready format with documented cleaning decisions.

**Exploratory Data Analysis Functions:** Age analysis function generating distribution statistics (mean, median, std, quartiles) with histogram and box plot visualizations. Gender analysis function producing distribution counts, percentages, and bar chart visualization. Education analysis function creating horizontal bar charts of top 10 education categories. Income analysis function examining binary income distribution with category labels and percentages.

**Model Training and Evaluation Code:** Preprocessing pipeline using ColumnTransformer with StandardScaler for numerical features and OneHotEncoder for categorical features. Train-test split with stratification to preserve class balance (80/20 split). Model comparison loop training Logistic Regression, Decision Tree, and Random Forest with 5-fold cross-validation. Metrics calculation including accuracy, precision, recall, and F1-score on test set.

**Model Interpretation Outputs:** Logistic regression coefficient extraction showing intercept and feature weights in log-odds formula. Decision tree rule extraction using export_text with maximum depth of 5 levels for readability. Random Forest feature importance ranking showing top 10 contributing features with normalized importance scores.

### 8. References 

Becker, B., & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

GeeksforGeeks. (n.d.). *Data cleaning introduction*. https://www.geeksforgeeks.org/data-analysis/what-is-data-cleaning/

Hyndman, R. J. (n.d.). *Exploratory data analysis (EDA)*. NIST/SEMATECH e-Handbook of Statistical Methods. National Institute of Standards and Technology. https://www.itl.nist.gov/div898/handbook/toolaids/pff/eda.pdf

McKinney, W. (2010). Data structures for statistical computing in Python. In S. van der Walt & J. Millman (Eds.), *Proceedings of the 9th Python in Science Conference* (pp. 56–61). https://doi.org/10.25080/Majora-92bf1922-00a

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

Raju, S. K. (n.d.). *Colab Python data cleaning project* [Code repository]. GitHub. https://github.com/sathishkumarraju/colab-python-data-cleaning-project

ucimlrepo contributors. (n.d.). *ucimlrepo Python package* [Software]. GitHub. https://github.com/uci-ml-repo/ucimlrepo



