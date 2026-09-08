import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "processed"

print("=" * 80)
print("ML → BUSINESS INSIGHTS")
print("=" * 80)


# ============================================================
# 1. MODEL PERFORMANCE
# ============================================================

print("\nCreating model performance summary...")

model_summary = pd.DataFrame({
    "model": [
        "Logistic Regression",
        "XGBoost"
    ],

    "PR-AUC": [
        0.3208,
        0.3820
    ],

    "Precision": [
        0.1476,
        0.3413
    ],

    "Recall": [
        0.8486,
        0.5243
    ],

    "F1": [
        0.2514,
        0.4135
    ]
})

print("\n")
print(model_summary.to_string(index=False))

model_summary.to_csv(
    OUTPUT_DIR / "ml_business_model_summary.csv",
    index=False
)


# ============================================================
# 2. RANKING PERFORMANCE
# ============================================================

print("\nCreating ranking performance summary...")

ranking_summary = pd.DataFrame({

    "model": [
        "Global Popularity",
        "Logistic Regression",
        "XGBoost"
    ],

    "P@5": [
        0.099311,
        0.368296,
        0.399339
    ],

    "R@5": [
        0.072788,
        0.333142,
        0.359738
    ],

    "MAP@5": [
        0.069566,
        0.351626,
        0.398222
    ],

    "NDCG@5": [
        0.119577,
        0.454613,
        0.498822
    ],

    "P@10": [
        0.075431,
        0.294128,
        0.312617
    ],

    "R@10": [
        0.107021,
        0.489498,
        0.514675
    ],

    "MAP@10": [
        0.057902,
        0.340521,
        0.381079
    ],

    "NDCG@10": [
        0.116757,
        0.476705,
        0.514670
    ],

    "P@20": [
        0.054996,
        0.215689,
        0.223571
    ],

    "R@20": [
        0.151195,
        0.661444,
        0.679406
    ],

    "MAP@20": [
        0.058239,
        0.364455,
        0.401551
    ],

    "NDCG@20": [
        0.128420,
        0.531480,
        0.564286
    ]
})

print("\n")
print(ranking_summary.to_string(index=False))

ranking_summary.to_csv(
    OUTPUT_DIR / "ml_business_ranking_summary.csv",
    index=False
)


# ============================================================
# 3. XGBOOST FEATURE IMPORTANCE
# ============================================================

print("\nLoading XGBoost feature importance...")

feature_file = OUTPUT_DIR / "xgboost_feature_importance.csv"

if feature_file.exists():

    feature_importance = pd.read_csv(feature_file)

    print("\nFeature importance columns:")
    print(feature_importance.columns.tolist())

    importance_column = feature_importance.columns[-1]

    feature_importance = feature_importance.sort_values(
        importance_column,
        ascending=False
    )

    print("\nTop XGBoost features:")
    print(
        feature_importance.head(20).to_string(index=False)
    )

    feature_importance.to_csv(
        OUTPUT_DIR / "ml_business_feature_summary.csv",
        index=False
    )

else:

    print(
        "\nWARNING: xgboost_feature_importance.csv not found."
    )


# ============================================================
# 4. BUSINESS INTERPRETATION
# ============================================================

business_feature_map = {

    "previous_purchases":
        "Historical frequency of customer purchases of the product",

    "customer_product_reorder_rate":
        "How consistently the customer repurchased the product",

    "previous_reorders":
        "Historical repeat-purchase behavior for the customer-product relationship",

    "purchase_recency":
        "How recently the customer purchased the product",

    "first_purchase_order":
        "How early the product entered the customer's shopping history",

    "previous_orders":
        "Overall customer shopping-history depth",

    "product_reorder_rate":
        "Overall tendency of the product to be repurchased",

    "product_unique_customers":
        "Breadth of customer adoption for the product",

    "customer_department_share":
        "Importance of the product department within the customer's purchases",

    "avg_basket_size":
        "Typical number of products in a customer's order",

    "avg_days_between_orders":
        "Typical customer ordering interval",

    "avg_cart_position":
        "Typical position of the product within the cart",

    "product_purchase_count":
        "Overall product demand",

    "customer_reorder_rate":
        "Overall tendency of the customer to reorder products",

    "customer_department_purchases":
        "Customer's historical activity within the product department",

    "product_department_popularity":
        "Product popularity relative to its department"
}


if feature_file.exists():

    feature_column = feature_importance.columns[0]
    importance_column = feature_importance.columns[-1]

    business_features = feature_importance.copy()

    business_features["business_interpretation"] = (
        business_features[feature_column]
        .map(business_feature_map)
    )

    business_features.to_csv(
        OUTPUT_DIR / "ml_business_feature_summary.csv",
        index=False
    )


# ============================================================
# 5. XGBOOST VS GLOBAL POPULARITY
# ============================================================

print("\n" + "=" * 80)
print("XGBOOST VS GLOBAL POPULARITY")
print("=" * 80)

popularity = ranking_summary[
    ranking_summary["model"] == "Global Popularity"
].iloc[0]

xgb = ranking_summary[
    ranking_summary["model"] == "XGBoost"
].iloc[0]

comparison_rows = []

for metric in [
    "P@5",
    "R@5",
    "P@10",
    "R@10",
    "P@20",
    "R@20"
]:

    absolute_improvement = (
        xgb[metric] -
        popularity[metric]
    )

    relative_improvement = (
        absolute_improvement /
        popularity[metric]
    ) * 100

    comparison_rows.append({

        "metric": metric,

        "global_popularity": popularity[metric],

        "xgboost": xgb[metric],

        "absolute_improvement":
            absolute_improvement,

        "relative_improvement_percent":
            relative_improvement
    })


baseline_comparison = pd.DataFrame(
    comparison_rows
)

print(
    baseline_comparison.to_string(index=False)
)

baseline_comparison.to_csv(
    OUTPUT_DIR / "ml_business_baseline_comparison.csv",
    index=False
)


# ============================================================
# 6. CANDIDATE STRATEGY CONTEXT
# ============================================================

print("\n" + "=" * 80)
print("CANDIDATE STRATEGY")
print("=" * 80)

candidate_summary = pd.DataFrame({

    "metric": [
        "Candidate strategy",
        "Candidate coverage",
        "Average candidates per customer",
        "Actual target pairs",
        "New-to-customer target pairs"
    ],

    "value": [
        "Historical + Global Top 100 + Top 3 Departments × Top 15 Products",
        "66.19%",
        "172.5",
        "1,384,617",
        "40.14%"
    ]
})

print(
    candidate_summary.to_string(index=False)
)


# ============================================================
# 7. BUSINESS TAKEAWAYS
# ============================================================

print("\n" + "=" * 80)
print("BUSINESS TAKEAWAYS")
print("=" * 80)

takeaways = [

    "1. Personalized recommendation models substantially outperform "
    "global popularity.",

    "2. XGBoost delivers the strongest overall ranking performance "
    "among the evaluated approaches.",

    "3. Customer-product purchase history is the strongest "
    "recommendation signal.",

    "4. Reorder behavior and purchase recency add important "
    "personalization signals.",

    "5. Global popularity is useful for candidate generation and "
    "cold-start fallback, but is not sufficient as the primary "
    "recommendation strategy.",

    "6. The current recommendation approach is strongest when "
    "customers have previously interacted with the product.",

    "7. Candidate coverage is 66.19%, meaning the current candidate "
    "generation strategy does not expose every possible target product "
    "to the ranking model.",

    "8. Approximately 40.14% of target customer-product pairs are "
    "new-to-customer, highlighting an opportunity for a future "
    "product-discovery layer.",

    "9. The recommended business architecture is a personalized "
    "ranking layer with popularity-based fallback and future "
    "new-product discovery."
]

for takeaway in takeaways:
    print("\n" + takeaway)


# ============================================================
# 8. FINAL OUTPUTS
# ============================================================

print("\n" + "=" * 80)
print("STEP 20 COMPLETE")
print("=" * 80)

print("\nGenerated files:")

files = [
    "ml_business_model_summary.csv",
    "ml_business_ranking_summary.csv",
    "ml_business_feature_summary.csv",
    "ml_business_baseline_comparison.csv"
]

for filename in files:

    path = OUTPUT_DIR / filename

    if path.exists():
        print("✓", path)
    else:
        print("⚠", path, "(not generated)")

print("\nDone.")