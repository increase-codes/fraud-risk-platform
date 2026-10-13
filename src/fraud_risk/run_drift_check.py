"""Run numeric feature drift checks and save a JSON report."""

import json
from datetime import datetime, timezone
from pathlib import Path

from fraud_risk.data_loader import load_data
from fraud_risk.data_split import split_data
from fraud_risk.drift_detection import detect_numeric_drift

REPORT_FILE = Path("reports") / "drift_report.json"


def main():
    df = load_data()
    train_df, validation_df, test_df = split_data(df)

    excluded = {"fraud_bool", "month"}
    reference = train_df.drop(
        columns=[c for c in excluded if c in train_df.columns]
    )
    validation = validation_df.drop(
        columns=[c for c in excluded if c in validation_df.columns]
    )
    test = test_df.drop(
        columns=[c for c in excluded if c in test_df.columns]
    )

    numeric_columns = reference.select_dtypes(include="number").columns
    reference = reference[numeric_columns]
    validation = validation[numeric_columns]
    test = test[numeric_columns]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_months": sorted(train_df["month"].unique().tolist()),
        "threshold": 0.2,
        "partitions": {},
    }

    for name, current in [
        ("validation_month_3", validation),
        ("test_month_7", test),
    ]:
        results = detect_numeric_drift(reference, current)
        drifted = [
            feature for feature, result in results.items()
            if result["drift_detected"]
        ]

        report["partitions"][name] = {
            "features_checked": len(results),
            "drifted_features": len(drifted),
            "drift_detected": bool(drifted),
            "features": results,
        }

        print(f"\n{name}")
        print(f"Numeric features checked: {len(results)}")
        print(f"Drifted features: {len(drifted)}")
        for feature in drifted:
            print(f"  {feature}: PSI={results[feature]['psi']}")

        if not drifted:
            print("No features exceeded the PSI threshold.")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"\nDrift report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
