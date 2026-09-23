from fraud_risk.data_loader import load_data


def main():
    df = load_data()

    fraud = df[df["fraud_bool"] == 1]
    non_fraud = df[df["fraud_bool"] == 0]

    print("Fraud cases:", len(fraud))
    print("Non-fraud cases:", len(non_fraud))
    print()

    print("Fraud rate:", len(fraud) / len(df) * 100)
    print()

    print("Fraud group:")
    print(fraud.describe(include="all").T)

    print()
    print("Non-fraud group:")
    print(non_fraud.describe(include="all").T)

    print(df["date_of_birth_distinct_emails_4w"].describe())
    print()
    print(df["date_of_birth_distinct_emails_4w"].value_counts().sort_index().head(20))

    print("\nDate-of-birth distinct emails range:")
    print("Minimum:", df["date_of_birth_distinct_emails_4w"].min())
    print("Maximum:", df["date_of_birth_distinct_emails_4w"].max())
    print("Unique values:", df["date_of_birth_distinct_emails_4w"].nunique())

    print("\nDOB distinct emails:")
    print("Minimum:", df["date_of_birth_distinct_emails_4w"].min())
    print("Maximum:", df["date_of_birth_distinct_emails_4w"].max())
    print("Unique:", df["date_of_birth_distinct_emails_4w"].nunique())

if __name__ == "__main__":
    main()