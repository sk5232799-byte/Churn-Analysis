import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

conn = sqlite3.connect("customer_churn.db")

sql_query = """
        SELECT name 
        FROM sqlite_master 
        WHERE type='table';
        """

tables = pd.read_sql(sql_query, conn)

# create dataframe fro each table

for table_name in tables["name"]:
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    globals()[f"df_{table_name}"] = df
    print(f"Created dataframe: df_{table_name}")

conn.close()


# Data cleaning and preprocessing

df_db_customer.head()  # type: ignore

# rename name column to customer_name

df_db_customer.rename(columns={"name": "customer_name"}, inplace=True)  # type: ignore

# drop column = intrest and pincode

df_db_customer.drop(df_db_customer.columns[-2:], axis=1, inplace=True)

# change datatype = dob

df_db_customer["dob"] = pd.to_datetime(df_db_customer["dob"])

# data standardization = gender

df_db_customer["gender"] = df_db_customer["gender"].replace(
    {"Men": "Male", "Women": "Female"}, inplace=True
)

# fix missing values = country
# print(df_db_customer[df_db_customer['country'].isna()])

state_country_mapping = (
    df_db_customer.dropna(subset=["country"]).set_index("state")["country"].to_dict()
)

df_db_customer["country"] = df_db_customer["country"].fillna(
    df_db_customer["state"].map(state_country_mapping)
)

print(df_db_customer[df_db_customer["country"].isna()])

df_db_customer.info()


date_col = ["subscription_start_date", "renewal_date", "cancellation_date"]

df_db_subscription[date_col] = df_db_subscription[date_col].apply(pd.to_datetime)


print(df_db_subscription.info())

df_db_support.drop(columns=["col_1", "comment"], inplace=True)

df_db_support["complaint_date"] = df_db_support["complaint_date"].apply(pd.to_datetime)

print(df_db_support.info())

# delete duplicates
df_db_support["complaint_count"] = df_db_support.groupby("customerid")[
    "customerid"
].transform("count")

df_db_support = df_db_support.sort_values("complaint_count").drop_duplicates(
    "customerid", keep="last"
)


# Feature Engineering and Data Analysis

# creating a columnn
df_db_subscription["churn_flag"] = np.where(
    df_db_subscription["cancellation_date"].notna(), 1, 0
)

# merge dataframe
df = df_db_subscription.merge(df_db_customer, on="customerid", how="left").merge(
    df_db_support, on="customerid", how="left"
)


# Data Analysis

# churn rate
churn_rate = df["churn_flag"].mean() * 100
print("churn_rate = ", round(churn_rate, 2), "%")

# Reteion rate
retention_rate = 100 - churn_rate
print("retention_rate = ", round(retention_rate, 2), "%")


# churn by plan type
churn_by_plan = (
    df.groupby("plan_type")["churn_flag"]
    .mean()
    .mul(100)
    .round(2)
    .reset_index(name="churn_rate_per")
)
print(churn_by_plan)

# churn by state + sum of revenue & count of users
churn_by_state = (
    df.groupby("state")["churn_flag"]
    .mean()
    .mul(100)
    .round(2)
    .reset_index(name="churn_state_per")
)
print(churn_by_state)

sor = df["monthly_charges"].sum()
print("Total Revenue = ", round(sor, 2))

churn_count = df["churn_flag"].sum()
print("Total Churn = ", churn_count)

# churn by subscription
churn_by_subscription = (
    df.groupby("subscription_type")["churn_flag"]
    .mean()
    .mul(100)
    .round(2)
    .reset_index(name="churn_by_subscription")
)
print(churn_by_subscription)

# ARPU = avg. revenue per user
arpu = df["monthly_charges"].mean()
print("ARPU = ", round(arpu, 2))

# avg. customer tanure
today = pd.Timestamp.today()

df["Tanure_days"] = np.where(
    df["cancellation_date"].notna(),
    (df["cancellation_date"] - df["subscription_start_date"]).dt.days,
    (today - df["subscription_start_date"]).dt.days,
)

avg_tanure = df["Tanure_days"].mean()
print("Avg Tanure days = ", round(avg_tanure), 0)


# revenue at risk
rar = df.loc[df["churn_flag"] == 1, "monthly_charges"].sum()
print("revenue at risk = ", rar)

#  esclation rate
escalation_rate = (df["escalations"] == "Y").mean() * 100
print("Escalation rate = ", round(escalation_rate, 2), "%")

# Avg complaint per user
avg_complaint = df["complaint_count"].sum() / df["customerid"].nunique()
print("Avg complaint per user = ", round(avg_complaint, 2))

# correlation escalation vs churn
df["escalations"] = np.where(df["escalations"] == "Y", 1, 0)
corr_df = df[["escalations", "churn_flag"]].dropna()
correlation = corr_df["escalations"].corr(df["churn_flag"])
print("Correlation between escalation vs churn is = ", round(correlation, 2))

# churn risk : create a column using existing column

conditions = [
    (df["churn_score"] < 50),
    (df["churn_score"] >= 50) & (df["churn_score"] < 70),
    (df["churn_score"] >= 70),
]

choices = ["low", "mid", "high"]

df["churn_risk"] = np.select(conditions, choices, default="unknown")

print(df[["churn_risk", "churn_score"]].head())


# data analysis save as csv
df.to_csv("churn_data.csv", index=False)

# visualization using matplotlib
df_visual = df.copy()

# monthly churn trend (time series kpi)

df_visual["cancellation_month"] = df_visual["cancellation_date"].dt.to_period("M")
churn_trend = (
    df_visual[df_visual["churn_flag"] == 1].groupby("cancellation_month").size()
)

plt.figure(figsize=(8, 3))
plt.plot(
    churn_trend.index.astype(str),
    churn_trend.values,
    color="green",
    marker="o",
    linestyle="dashed",
    linewidth=2,
    markersize=12,
)

plt.title("Monthly churn trend")
plt.xlabel("Month")
plt.ylabel("Churn Customer")
plt.show()

# churn by plan type
churn_plan = df_visual.groupby("plan_type")["churn_flag"].mean()

colors = plt.cm.Set2(np.linspace(0, 1, len(churn_plan)))

plt.figure(figsize=(7, 4))
plt.bar(churn_plan.index, churn_plan.values, color=colors)

plt.title("Churn by plan")
plt.xlabel("plan")
plt.ylabel("Churn Customers")
plt.show()

# churn by state

churn_state = df_visual.groupby("state")["churn_flag"].mean()

colors = plt.cm.Set2(np.linspace(0, 1, len(churn_state)))

plt.figure(figsize=(7, 4))
plt.bar(churn_state.index, churn_state.values, color=colors)

plt.title("Churn by state")
plt.xlabel("state")
plt.ylabel("Churn Customers")
plt.show()


# Visualization by Seaborn

# encoding - convert str to numeric so that we can find corr between feature
df_encoded = df_visual[
    [
        "plan_type",
        "contract_type",
        "churn_score",
        "churn_flag",
        "escalations",
        "churn_risk",
    ]
]

order_mappings = {
    "churn_risk": ["low", "mid", "high"],
    "plan_type": ["Basic", "Standard", "Premium"],
    "contract_type": ["Monthly", "Annual"],
}

for col, order in order_mappings.items():
    df_encoded[col] = pd.Categorical(
        df_encoded[col].astype("category"), categories=order, ordered=True
    ).codes

# Heatmap (correlation matrix)
sns.heatmap(df_encoded.corr(), annot=True)
plt.show()

# Pairplot
sns.pairplot(df_encoded)
plt.show()

# catplt / facegrid plot
sns.catplot(
    data=df_visual, x="plan_type", y="monthly_charges", hue="gender", col="churn_risk"
)
plt.show()


# Pivot table
pivot = pd.pivot_table(df_visual, 
            values= ['monthly_charges', 'customerid', 'churn_flag'], 
            index='plan_type', 
            aggfunc = {
                'monthly_charges' : 'sum',
                  'customerid' : 'nunique', 
                  'churn_flag' : 'mean'
            }
    )
print(pivot)

# working with sql in python (pandas)
# create db in sql
conn = sqlite3.connect('test_database.sqlite')

# # table details
# conn.execute("CREATE TABLE users (first_name TEXT, country TEXT, budget INTEGER)")

# # commit and save
conn.commit()

# insert data
cursor = conn.cursor()
cursor.execute(
        """
            INSERT INTO users VALUES
            ('Sadiq', 'India', 5000),
            ('Rishabh', 'Germany', 2500),
            ('Imaran', 'Spain', 3500)
    """
)

# commit and save
conn.commit()
print("Data Inerted successfully")

# check inserted data in table
conn = sqlite3.connect('test_database.sqlite')
query = """
SELECT * FROM users
"""
df_results = pd.read_sql(query,conn)

print(df_results)

# aggregation

query = """
SELECT country, sum(budget) as total_budget
FROM users
GROUP BY country
"""
df_agg = pd.read_sql(query,conn)
print(df_agg)

conn.close()