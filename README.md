# 1. Project Overview & Business Problem

## Project Overview

**Instacart Business Analytics** is an end-to-end customer intelligence and decision-support project built using the Instacart Market Basket Analysis dataset.

The project transforms raw grocery transaction data into actionable business insights using:

- Python
- SQL
- DuckDB
- Pandas
- Scikit-learn
- XGBoost
- Tableau

The solution progresses through the complete analytics lifecycle:

**Descriptive → Diagnostic → Predictive → Prescriptive**

The objective is not only to understand historical customer behavior, but also to predict what customers are likely to do next and translate those predictions into actionable business strategies.

---

## Business Problem

A grocery business has access to millions of historical transactions, but historical data alone does not directly answer important business questions such as:

- Which customers demonstrate strong loyalty?
- Which customers may be at risk of becoming inactive?
- Which products are customers likely to purchase next?
- How large is a customer's next basket likely to be?
- How much of the next basket is likely to consist of previously purchased products?
- What action should the business take for each customer?

This project addresses these questions through four major analytical layers.

### 1. Descriptive & Diagnostic Analytics

Understand customer, product, department, basket, and reorder behavior.

### 2. Recommendation Analytics

Predict which products a customer is likely to purchase in their next order.

### 3. Predictive Customer Forecasting

Forecast:

- Future order likelihood
- Expected reorder rate
- Expected basket size

### 4. Prescriptive Analytics

Translate customer predictions and recommendation signals into business actions such as:

- Re-engage Customer
- Retain & Reward
- Cross-sell & Discover
- Promote Recommendations
- Maintain Engagement
- Monitor

---

## Project Objectives

The major objectives are:

- Build a reliable local analytical data layer
- Validate large-scale transactional data
- Analyze customer purchasing behavior
- Identify product and department patterns
- Build a personalized recommendation system
- Forecast future customer behavior
- Create customer-level behavioral segments
- Convert predictions into actionable strategies
- Build interactive Tableau dashboards
- Demonstrate an end-to-end business decision-support workflow

# 2. Dataset, Architecture & Data Engineering

## Dataset

The project uses the publicly available **Instacart Market Basket Analysis** dataset.

The working dataset contains approximately:

| Dataset Component | Volume |
|---|---:|
| Customers | 206,209 |
| Orders | 3,346,083 |
| Order-product records | 33,819,106 |
| Products | 49,685 |

The raw data contains information about:

- Orders
- Products
- Aisles
- Departments
- Customer-product purchases
- Reorder behavior
- Order day and hour
- Days since previous order
- Cart position

---

## Data Architecture

The project follows a local-first analytical architecture:

    Raw CSV Data
         |
         v
    Data Validation & Cleaning
         |
         v
    DuckDB Analytical Layer
         |
         +-------------------+
         |                   |
         v                   v
    SQL Analytics       Python Analytics
         |                   |
         +---------+---------+
                   |
                   v
        Customer & Product Insights
                   |
           +-------+-------+
           |               |
           v               v
    Recommendation     Forecasting
          ML                 ML
           |               |
           +-------+-------+
                   |
                   v
          Prescriptive Strategy
                   |
                   v
              Tableau BI

---

## DuckDB Data Layer

DuckDB is used as the local analytical database because the order-product dataset contains more than **33 million records**.

The main analytical tables are:

    orders
    products
    order_products
    customers
    product_features

Parquet files are also used for efficient storage of large analytical and machine-learning datasets.

---

## Data Validation

The project includes validation checks for:

- Duplicate records
- Missing values
- Referential integrity
- Product uniqueness
- Order uniqueness
- Customer uniqueness
- Valid order-day values
- Valid order-hour values
- Valid reorder flags
- Valid cart positions
- Sequence consistency

Key validation results:

- Duplicate order-product rows: **0**
- Invalid reorder records: **0**
- Invalid order-day values: **0**
- Invalid order-hour values: **0**
- Referential integrity issues: **None identified**

The `days_since_prior_order` field contains legitimate null values for first orders because there is no previous order from which to calculate an interval.

---

## Local-First Design

The current implementation is designed to run entirely locally using:

- Local CSV files
- DuckDB
- Parquet
- Python
- Tableau

No cloud infrastructure is required.

This makes the project reproducible in a local development environment while still supporting large-scale transactional analytics.

# 3. Business Analytics & Key Insights

## Customer Analytics

The project analyzes customer purchasing behavior using SQL and Python.

Key customer-level metrics include:

- Total orders
- Total items purchased
- Average basket size
- Customer-level reorder rate
- Average days between orders
- Purchase frequency
- Behavioral segments

The dataset contains **206,209 customers**.

Average customer-level behavior is approximately:

| Metric | Value |
|---|---:|
| Average Orders | 16.23 |
| Average Items Purchased | 164 |
| Average Basket Size | ~10 items |
| Average Customer Reorder Rate | 44.4% |
| Average Days Between Orders | 15.6 days |

---

## Customer Frequency & Loyalty

Customers were grouped into order-frequency bands:

- 1–5 orders
- 6–10 orders
- 11–20 orders
- 21–30 orders
- 31+ orders

The analysis shows a strong relationship between purchase frequency and repeat behavior.

| Frequency | Avg Basket | Reorder Rate |
|---|---:|---:|
| 1–5 | 9.65 | 25.80% |
| 6–10 | 9.91 | 37.69% |
| 11–20 | 10.08 | 49.97% |
| 21–30 | 10.27 | 59.73% |
| 31+ | 10.33 | 69.51% |

### Business Insight

Customers with higher purchase frequency demonstrate substantially stronger reorder behavior.

For example, customers with **31+ orders have a reorder rate of approximately 69.5%**, compared with **25.8% for customers with 1–5 orders**.

This indicates that purchase frequency is a useful behavioral indicator of customer loyalty.

---

## Product Analytics

Product performance was evaluated using:

- Purchase volume
- Unique customers
- Unique orders
- Product reorder rate
- Average cart position

The dataset contains **49,685 products**.

Examples of high-volume products include:

- Banana
- Bag Organic Bananas
- Organic Strawberries
- Organic Baby Spinach
- Organic Hass Avocado

Banana is one of the strongest examples, with approximately:

- **491K purchases**
- **84.5% product-level reorder rate**

Frequently purchased grocery staples therefore provide strong behavioral signals for recommendation modeling.

---

## Department Analytics

Major departments were analyzed using:

- Purchase volume
- Customer reach
- Reorder behavior
- Product depth
- Customer penetration

High-volume departments include:

- Produce
- Dairy Eggs
- Snacks
- Beverages
- Frozen
- Pantry

Produce and Dairy Eggs demonstrate particularly strong repeat-purchase behavior.

---

## Key Business Insights

### 1. Frequency is strongly associated with loyalty

Higher-frequency customers show substantially higher reorder rates.

### 2. Basket size increases with engagement

Average basket size gradually increases as customer order frequency increases.

### 3. Grocery staples show strong repeat behavior

Products such as bananas, strawberries, spinach, and avocados demonstrate strong reorder signals.

### 4. Historical behavior contains predictive information

Customer purchase history provides useful signals for predicting future ordering behavior, reorder behavior, and basket size.

### 5. Analytics can progress from insight to action

The descriptive analysis provides the behavioral foundation for the recommendation, forecasting, and prescriptive layers developed later in the project.

# 4. Recommendation System

## Business Question

> Can we predict which products a customer is likely to purchase in their next order based on previous shopping behavior?

The recommendation system uses a **candidate-generation + machine-learning ranking** architecture.

    Customer Purchase History
              |
              v
       Candidate Generation
              |
              v
        Feature Engineering
              |
              v
       Logistic Regression
            Baseline
              |
              v
          XGBoost Model
              |
              v
        Product Scoring
              |
              v
          Top-K Ranking
              |
              v
       Personalized Products

---

## Candidate Generation

The final candidate-generation strategy combines three sources.

### Historical Products

Products previously purchased by the customer are included because they provide the strongest personalization signal.

### Global Popular Products

The globally popular products provide additional coverage and help introduce products that may not exist in a customer's purchase history.

### Customer Department Preferences

Products from the customer's strongest departments are included to provide personalized discovery opportunities.

This strategy balances:

- Personalization
- Product discovery
- Popularity
- Candidate coverage

---

## Recommendation Features

The model uses customer-product and product-level behavioral features including:

- Previous purchases
- Previous reorders
- Customer-product reorder rate
- Purchase recency
- Previous order count
- Average basket size
- Customer reorder rate
- Product purchase count
- Product unique customers
- Product reorder rate
- Customer department purchases
- Customer department share
- Product department popularity

These features allow the model to combine individual customer behavior with broader product-level behavior.

---

## Models

### Logistic Regression

Logistic Regression was used as the baseline model because it provides a simple and interpretable benchmark.

### XGBoost

XGBoost was used as the main nonlinear model because it can capture interactions between:

- Customer behavior
- Product behavior
- Reorder history
- Product popularity
- Department preferences

---

## Evaluation

The recommendation system was evaluated using both classification and ranking metrics.

### Classification Metrics

- Precision
- Recall
- F1
- PR-AUC

### Ranking Metrics

- Precision@10
- Recall@10
- MAP@10
- NDCG@10

Ranking metrics are particularly important because the final business output is an ordered list of recommended products.

---

## Final Recommendation Results

The final XGBoost model achieved approximately:

| Metric | XGBoost |
|---|---:|
| PR-AUC | 0.382 |
| Precision | 0.341 |
| Recall | 0.525 |
| F1 | 0.414 |
| Precision@10 | 0.313 |
| Recall@10 | 0.515 |
| MAP@10 | 0.381 |
| NDCG@10 | 0.515 |

XGBoost outperformed the Logistic Regression baseline on the primary ranking metrics.

---

## Leakage Prevention

A key concern in recommendation modeling is allowing future customer behavior to influence historical features.

The evaluation therefore follows customer-level temporal logic rather than relying only on random row splits.

The principle is:

    Historical Customer Behavior
                |
                v
             Training
                |
                v
          Future Behavior
                |
                v
            Evaluation

This provides a more realistic estimate of how the recommendation system would perform when predicting future customer purchases.

---

## Recommendation Output

The final recommendation results contain customer-product predictions and ranking information, including:

- Customer ID
- Recommendation rank
- Product ID
- Product name
- Department
- Predicted probability
- Previous purchases
- Previous reorders
- Customer-product reorder rate
- Product popularity
- Customer behavioral features

The recommendation layer therefore provides the product-level intelligence used later by the prescriptive strategy layer.

# 5. Customer Forecasting

## Business Question

The forecasting layer answers:

> **What is this customer likely to do next?**

Instead of predicting only one outcome, the project forecasts three complementary aspects of future customer behavior:

1. **Future Order Likelihood**
2. **Expected Reorder Rate**
3. **Expected Basket Size**

These predictions are combined into a single customer-level forecast.

---

## Forecasting Dataset

A dedicated forecasting dataset was created containing approximately:

- **3.14 million prediction points**
- **206,209 customers**

The forecasting features capture historical and recent customer behavior, including:

- Previous orders
- Current basket size
- Historical average basket size
- Previous basket size
- Recent basket behavior
- Historical reorder rate
- Recent reorder rate
- Historical average days between orders
- Recent order intervals
- Order day
- Order hour
- Total historical items
- Total historical reordered items
- Behavioral trends and variability

---

## Model 1 — Future Order Likelihood

### Business Question

> **Is the customer likely to place another order within the next 14 days?**

The model predicts the probability that a customer will place another order within the defined 14-day horizon.

XGBoost was used as the final model.

### Evaluation

| Metric | Score |
|---|---:|
| PR-AUC | 0.791 |
| ROC-AUC | 0.770 |
| Precision | 0.787 |
| Recall | 0.540 |
| F1 | 0.641 |

The production forecasting layer generates a future order likelihood for all **206,209 customers**.

---

## Model 2 — Future Reorder Rate

### Business Question

> **When the customer places their next order, what percentage of that basket is likely to consist of previously purchased products?**

The model uses historical reorder behavior and recent customer purchasing patterns.

The production configuration uses an XGBoost model combined with a recent-behavior blend.

The production predictions have an average expected reorder rate of approximately:

**60.9%**

This provides an estimate of how repeat-oriented the customer's next basket may be.

---

## Model 3 — Future Basket Size

### Business Question

> **When the customer places their next order, approximately how many items are they likely to purchase?**

The model predicts future basket size using historical basket behavior and basket dynamics.

The final production model uses an XGBoost regression configuration.

### Evaluation

| Metric | Score |
|---|---:|
| MAE | 3.91 items |
| RMSE | 5.66 items |
| R² | 0.487 |

The production forecast has an average expected basket size of approximately:

**9.7 items**

---

## Customer Forecast Master

The three model outputs are combined into:

    Data/processed/customer_forecast_master.csv

Each customer receives:

- Future order likelihood
- Expected reorder rate
- Expected basket size
- Expected reorder items
- Expected new items
- Historical behavioral features
- Forecast segment

Two additional metrics are derived from the model outputs.

### Expected Reorder Items

    Expected Reorder Items
    =
    Expected Basket Size × Expected Reorder Rate

### Expected New Items

    Expected New Items
    =
    Expected Basket Size − Expected Reorder Items

These values represent **expected basket composition**, not guaranteed future purchases.

---

## Forecast Segmentation

Customers are assigned business-oriented forecast segments based on their predicted future behavior:

- **At-Risk**
- **Regular**
- **Loyal**
- **Potential Loyal**
- **Occasional**

Production distribution:

| Forecast Segment | Customers |
|---|---:|
| At-Risk | 118,393 |
| Regular | 35,820 |
| Loyal | 34,627 |
| Potential Loyal | 14,664 |
| Occasional | 2,705 |

### Important

These segments are **rule-based business interpretations of model outputs**.

They are not separate machine-learning classification predictions.

---

## Business Value

The forecasting layer allows the business to move from historical customer analysis toward forward-looking decision making.

Instead of only asking:

> **"What has this customer done?"**

the business can now ask:

> **"What is this customer likely to do next?"**

The resulting forecasts become important inputs to the final **Prescriptive Strategy** layer.

# 6. Prescriptive Strategy

## Business Question

The prescriptive layer answers:

> **Given the predicted customer behavior and recommendation signals, what should the business do?**

This layer connects machine-learning predictions with business decision making.

The strategy combines:

- Future order likelihood
- Expected reorder rate
- Expected basket size
- Expected new items
- Recommendation availability
- Top recommended product
- Recommended department
- Customer forecast segment

The final customer-level output is:

    Data/processed/prescriptive_strategy_master.csv

The output contains **206,209 customers** and **32 fields**.

---

## From Prediction to Action

The prescriptive workflow is:

    Customer History
           |
           v
    Behavioral Analytics
           |
           v
    Recommendation Model
           |
           v
    Forecasting Models
           |
           v
    Customer-Level Signals
           |
           v
    Business Rules
           |
           v
    Recommended Action
           |
           v
    Action Priority

This converts model outputs into interpretable business actions.

---

## Recommended Business Actions

### Re-engage Customer

Used when the customer's predicted near-term order likelihood indicates a higher risk of inactivity.

Potential business use cases include:

- Re-engagement campaigns
- Reminder notifications
- Personalized offers

---

### Retain & Reward

Used for customers showing strong ordering likelihood and strong repeat-purchase behavior.

Potential business use cases include:

- Loyalty programs
- Personalized rewards
- Retention initiatives

---

### Cross-sell & Discover

Used when customers demonstrate strong repeat behavior while also showing opportunities to discover additional products.

Potential business use cases include:

- Complementary product recommendations
- Category expansion
- Personalized discovery

---

### Promote Recommendations

Used when recommendation signals indicate strong product relevance and an opportunity to promote personalized products.

---

### Maintain Engagement

Used for customers whose behavioral signals indicate relatively stable engagement.

The objective is to maintain the existing relationship rather than apply an aggressive intervention.

---

### Monitor

Used when the available behavioral signals do not strongly justify a more targeted intervention.

---

## Strategy Results

The production strategy produces approximately:

| Recommended Action | Customers |
|---|---:|
| Re-engage Customer | 116,028 |
| Retain & Reward | 34,627 |
| Maintain Engagement | 18,083 |
| Monitor | 13,459 |
| Cross-sell & Discover | 13,414 |
| Promote Recommendations | 10,598 |

### Priority Distribution

| Priority | Customers |
|---|---:|
| High | 150,655 |
| Medium | 27,705 |
| Low | 27,849 |

The large **Re-engage Customer** population is consistent with the large **At-Risk** population identified by the forecasting layer.

---

## Business Opportunity Interpretation

The prescriptive layer can prioritize customers using signals such as:

- Future order likelihood
- Expected reorder behavior
- Expected basket size
- Recommendation strength
- Expected new-item opportunity

These signals help the business determine where different customer strategies may be appropriate.

### Important Limitation

The strategy is a **rule-based business interpretation of model outputs**.

It should not be interpreted as a causal recommendation or as proof that a particular intervention will generate incremental revenue.

Because the dataset does not provide reliable pricing, margins, campaign costs, or inventory information, the project does not claim:

- Revenue uplift
- Profit impact
- ROI
- Incremental sales

# 7. Tableau Business Intelligence & Business Value

## Tableau Business Intelligence

The final Tableau workbook contains **five interactive dashboards**, designed to move from business understanding to customer-level decision making.

---

## Dashboard 1 — Instacart Business Overview

### Business Question

> **How is the business performing?**

Provides a high-level view of:

- Customers
- Orders
- Basket behavior
- Reorder behavior
- Product activity

This dashboard establishes the overall business context before moving into deeper customer and product analysis.

---

## Dashboard 2 — Customer & Product Insights

### Business Question

> **Who are the customers and what products drive repeat behavior?**

Provides insights into:

- Customer purchase frequency
- Basket behavior
- Customer segments
- Product performance
- Department performance
- Reorder behavior

This dashboard helps identify the behavioral patterns behind customer engagement and product repeat purchases.

---

## Dashboard 3 — Recommendation Intelligence

### Business Question

> **What products should be recommended to customers?**

Provides visibility into:

- Personalized product recommendations
- Recommendation behavior
- Product ranking
- Recommendation opportunities

The dashboard connects the recommendation model with a business-facing view of product suggestions.

---

## Dashboard 4 — Forecasting Intelligence

### Business Question

> **What is each customer likely to do next?**

The dashboard combines:

- Future order likelihood
- Expected reorder rate
- Expected basket size
- Expected reorder items
- Expected new items
- Customer forecast segments
- Customer-level forecast details

It allows users to examine both the overall customer population and individual customer forecasts.

---

## Dashboard 5 — Prescriptive Strategy

### Business Question

> **What should the business do about the predicted customer behavior?**

The dashboard combines:

- High-priority customers
- Recommended actions
- Action distribution
- Customer strategy positioning
- Forecast signals
- Recommendation signals
- Selected customer strategy

This represents the final transition from:

**Prediction → Business Action**

---

# Business Value

The overall project demonstrates how transactional data can be transformed into a decision-support system.

The analytical journey is:

    What Happened?
          |
          v
    Descriptive Analytics
          |
          v
    Why Did It Happen?
          |
          v
    Diagnostic Analytics
          |
          v
    What May Happen Next?
          |
          v
    Predictive Analytics
          |
          v
    What Should We Do?
          |
          v
    Prescriptive Analytics
          |
          v
    Business Decision

The project therefore moves beyond traditional reporting by connecting:

**Data → Insights → Predictions → Recommendations → Forecasts → Actions**

This provides a practical framework for customer intelligence and data-driven decision support.

# 8. Project Structure, Setup, Limitations & Future Improvements

## Project Structure

    Instacart Business Analytics/
    │
    ├── Data/
    │   ├── Raw/
    │   │   ├── aisles.csv
    │   │   ├── departments.csv
    │   │   ├── order_products__prior.csv
    │   │   ├── order_products__train.csv
    │   │   ├── orders.csv
    │   │   └── products.csv
    │   │
    │   └── processed/
    │       ├── instacart.duckdb
    │       ├── instacart_master.parquet
    │       ├── ml_final_dataset.parquet
    │       ├── forecasting_dataset.parquet
    │       ├── recommendation_results.csv
    │       ├── recommendation_results.parquet
    │       ├── customer_forecast_master.csv
    │       ├── prescriptive_strategy_master.csv
    │       └── analytical/model outputs
    │
    ├── ML/
    │   ├── 01_build_training_data.py
    │   ├── 17_build_forecasting_dataset.py
    │   ├── 18_train_order_likelihood*.py
    │   ├── 19_train_reorder_rate_forecast*.py
    │   ├── 20_train_basket_size_forecast*.py
    │   ├── 21_build_customer_forecast_master.py
    │   └── 22_build_prescriptive_strategy_master.py
    │
    ├── Python/
    │   └── Business analysis scripts
    │
    ├── SQL/
    │   └── Analytical SQL scripts
    │
    ├── Tableau/
    │   └── data/
    │
    ├── Book_Instacart.twb
    └── README.md

---

## Setup

### Create Virtual Environment

    python -m venv .venv

### Activate Environment

#### macOS / Linux

    source .venv/bin/activate

#### Windows

    .venv\Scripts\activate

---

## Install Dependencies

    pip install pandas duckdb pyarrow scikit-learn scipy joblib xgboost

---

## Run the Forecasting Pipeline

### Build Forecasting Dataset

    python ML/17_build_forecasting_dataset.py

Output:

    Data/processed/forecasting_dataset.parquet

### Build Customer Forecasts

    python ML/21_build_customer_forecast_master.py

Output:

    Data/processed/customer_forecast_master.csv

### Build Prescriptive Strategy

    python ML/22_build_prescriptive_strategy_master.py

Output:

    Data/processed/prescriptive_strategy_master.csv

---

## Tableau

Open the Tableau workbook:

    Book_Instacart.twb

Prepared Tableau data is stored under:

    Tableau/data/

The workbook contains the five completed dashboards:

1. Instacart Business Overview
2. Customer & Product Insights
3. Recommendation Intelligence
4. Forecasting Intelligence
5. Prescriptive Strategy

---

# Limitations

## Dataset Limitations

The dataset is an older public and anonymized dataset and should not be interpreted as current Instacart business performance.

## Revenue Limitations

The dataset does not provide reliable:

- Product prices
- Revenue
- Margins
- Campaign costs

Therefore, the project does not claim revenue uplift, profit impact, or ROI.

## Inventory Limitations

Inventory availability is not modeled.

A production recommendation system should incorporate real-time inventory information.

## Recommendation Coverage

The current personalized recommendation output covers a subset of customers.

Customers without personalized recommendations are not assigned an artificial personalized product.

A production implementation should provide a dedicated cold-start recommendation strategy.

## Forecast Uncertainty

Forecasts represent model estimates and are not guaranteed future outcomes.

## Prescriptive Strategy Limitations

The business actions are rule-based interpretations of model outputs rather than causal recommendations.

The project does not claim that a particular intervention will necessarily produce incremental revenue.

---

# Future Improvements

A production-grade implementation could be extended with:

### Customer Lifetime Value

Integrate revenue and margin information to prioritize customers based on financial value.

### Inventory-Aware Recommendations

Prevent recommendations for products that are unavailable.

### Price and Promotion Modeling

Estimate how discounts and promotions affect purchase probability.

### Uplift Modeling

Predict which customers are most likely to respond positively to a particular intervention.

### A/B Testing

Test recommended business actions and measure actual incremental impact.

### Better Cold-Start Recommendations

Use product similarity, category preferences, and global trends for customers without sufficient purchase history.

### Real-Time Recommendations

Expose recommendations through an API for real-time applications.

### Model Monitoring

Track:

- Prediction drift
- Feature drift
- Recommendation performance
- Forecast accuracy
- Customer behavior changes

### Automated Retraining

Retrain models periodically as new transaction data becomes available.

---

# Technologies Used

## Programming

- Python
- SQL

## Data Processing

- Pandas
- DuckDB
- Parquet
- PyArrow

## Machine Learning

- Scikit-learn
- XGBoost
- Feature Engineering
- Classification
- Regression
- Ranking
- Forecasting

## Business Intelligence

- Tableau

## Analytics

- Customer Analytics
- Product Analytics
- Business Intelligence
- Recommendation Systems
- Predictive Analytics
- Forecasting
- Prescriptive Analytics
- Decision Support

---

# Final Summary

This project demonstrates an end-to-end approach to business analytics by combining large-scale transactional data processing, SQL analytics, customer behavior analysis, recommendation systems, machine learning, forecasting, prescriptive analytics, and business intelligence.

The final solution provides a framework for answering:

> **What happened?**

> **What is likely to happen next?**

> **What should the business do about it?**

The result is a practical customer intelligence and decision-support system that connects:

**Data → Insights → Predictions → Recommendations → Forecasts → Actions**

