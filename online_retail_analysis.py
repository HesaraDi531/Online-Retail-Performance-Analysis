# ==========================================================
# Online Retail Performance Analysis
# Python | Pandas | Visualization | RFM Customer Segmentation
# ==========================================================


# ==============================
# Import Libraries
# ==============================

import os

# Remove KMeans CPU warning
os.environ["LOKY_MAX_CPU_COUNT"] = "8"


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



pd.set_option(
    "display.max_columns",
    None
)


sns.set_style(
    "whitegrid"
)



# ==========================================================
# MAIN FUNCTION
# ==========================================================


def main():

    print("\nStarting Online Retail Analysis")
    print("="*60)



    # ==============================
    # Load Dataset
    # ==============================


    file_path = (
        r"C:\Users\Hesara Dilnath\Downloads\Online Retail.csv"
    )


    df = pd.read_csv(
        file_path,
        encoding="ISO-8859-1"
    )


    print("\nDataset Loaded Successfully")


    print(df.head())



    # ==============================
    # Data Cleaning
    # ==============================


    print("\nOriginal Shape")

    print(df.shape)



    df = df.dropna(
        subset=["Description"]
    )



    # Remove cancelled invoices

    df = df[
        ~df["InvoiceNo"]
        .astype(str)
        .str.startswith("C")
    ]



    # Remove invalid values

    df = df[
        df["Quantity"] > 0
    ]


    df = df[
        df["UnitPrice"] > 0
    ]



    # Remove non-selling items

    remove_items = [

        "POSTAGE",
        "DOTCOM POSTAGE",
        "Manual",
        "BANK CHARGES"

    ]


    df = df[
        ~df["Description"]
        .isin(remove_items)
    ]



    print("\nAfter Cleaning")

    print(df.shape)




    # ==============================
    # Feature Engineering
    # ==============================


    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"]
    )


    df["Revenue"] = (

        df["Quantity"] *
        df["UnitPrice"]

    )



    df["Month"] = (

        df["InvoiceDate"]
        .dt.month

    )


    df["Month_Name"] = (

        df["InvoiceDate"]
        .dt.month_name()

    )


    print(
        "\nFeature Engineering Completed"
    )




    # ==============================
    # Business KPIs
    # ==============================


    revenue = df["Revenue"].sum()


    orders = (
        df["InvoiceNo"]
        .nunique()
    )


    customers = (
        df["CustomerID"]
        .nunique()
    )


    avg_order = (
        revenue/orders
    )



    print("\nBUSINESS KPI")
    print("="*40)


    print(
        f"Total Revenue : £{revenue:,.2f}"
    )


    print(
        f"Orders : {orders}"
    )


    print(
        f"Customers : {customers}"
    )


    print(
        f"Average Order Value : £{avg_order:,.2f}"
    )





    # ==============================
    # Monthly Revenue
    # ==============================


    monthly = (

        df.groupby(
            [
                "Month",
                "Month_Name"
            ]
        )
        ["Revenue"]
        .sum()
        .reset_index()

    )


    monthly = monthly.sort_values(
        "Month"
    )



    print("\nMonthly Revenue")

    print(monthly)



    plt.figure(
        figsize=(10,5)
    )


    sns.barplot(

        data=monthly,

        x="Month_Name",

        y="Revenue"

    )


    plt.title(
        "Monthly Revenue Trend"
    )


    plt.xticks(
        rotation=45
    )


    plt.show()




    # ==============================
    # Product Analysis
    # ==============================


    top_products = (

        df.groupby(
            "Description"
        )
        ["Revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)

    )


    print("\nTop Products")

    print(top_products)



    top_products.sort_values().plot(
        kind="barh",
        figsize=(10,6)
    )


    plt.title(
        "Top Products By Revenue"
    )


    plt.show()




    # ==============================
    # Country Analysis
    # ==============================


    countries = (

        df.groupby(
            "Country"
        )
        ["Revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)

    )


    print("\nTop Countries")

    print(countries)



    countries.plot(
        kind="bar",
        figsize=(10,5)
    )


    plt.title(
        "Revenue By Country"
    )


    plt.xticks(
        rotation=45
    )


    plt.show()




    # ==============================
    # RFM Analysis
    # ==============================


    print(
        "\nStarting RFM Analysis"
    )



    customer_df = df.dropna(
        subset=["CustomerID"]
    )



    reference_date = (

        customer_df["InvoiceDate"]
        .max()

    )



    rfm = (

        customer_df
        .groupby(
            "CustomerID"
        )
        .agg({

            "InvoiceDate":
            lambda x:
            (
                reference_date-x.max()
            ).days,


            "InvoiceNo":
            "nunique",


            "Revenue":
            "sum"

        })

    )



    rfm.columns = [

        "Recency",

        "Frequency",

        "Monetary"

    ]




    # Scale


    scaler = StandardScaler()


    scaled = scaler.fit_transform(
        rfm
    )



    model = KMeans(

        n_clusters=4,

        random_state=42,

        n_init=10

    )


    rfm["Cluster"] = (
        model.fit_predict(
            scaled
        )
    )





    # Business based segments


    def segment(row):

        if row["Monetary"] > 50000:
            return "VIP Customer"

        elif row["Frequency"] > 15:
            return "Loyal Customer"

        elif row["Recency"] > 200:
            return "Inactive Customer"

        else:
            return "Regular Customer"



    rfm["Customer_Segment"] = (

        rfm.apply(
            segment,
            axis=1
        )

    )



    print("\nCustomer Segments")

    print(
        rfm.head()
    )



    print("\nSegment Summary")


    print(

        rfm.groupby(
            "Customer_Segment"
        )
        [
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
        .mean()

    )





    # ==============================
    # Visualization
    # ==============================


    plt.figure(
        figsize=(8,5)
    )


    sns.scatterplot(

        data=rfm,

        x="Frequency",

        y="Monetary",

        hue="Customer_Segment"

    )


    plt.title(
        "Customer Segmentation"
    )


    plt.show()




    # Correlation


    plt.figure(
        figsize=(7,5)
    )


    sns.heatmap(

        rfm[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]

        ].corr(),

        annot=True

    )


    plt.title(
        "RFM Correlation"
    )


    plt.show()




    # ==============================
    # Save Results
    # ==============================


    output = (

        r"D:\Project\online_retail_analysis"

    )


    os.makedirs(
        output,
        exist_ok=True
    )



    df.to_csv(

        os.path.join(
            output,
            "clean_online_retail.csv"
        ),

        index=False

    )



    rfm.to_csv(

        os.path.join(
            output,
            "customer_segments.csv"
        )

    )



    print("\n================================")

    print(
        "Analysis Completed Successfully"
    )


    print(
        "Files Saved:"
    )


    print(
        "clean_online_retail.csv"
    )


    print(
        "customer_segments.csv"
    )


    print("================================")




# ==========================================================
# Prevent Multiple Execution
# ==========================================================


if __name__ == "__main__":

    main()
