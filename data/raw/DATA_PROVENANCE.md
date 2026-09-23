# Dataset Provenance

## Dataset
- **File:** `CCFraud_data.csv`
- **Rows:** 500,000
- **Columns:** 32

## Provenance
A 500,000-row republished copy/subset of the Bank Account Fraud (BAF) dataset family, obtained from a third-party GitHub repository.

The exact upstream source and transformation used to produce this 500,000-row copy cannot currently be established from the GitHub repository alone.

The dataset schema matches the BAF dataset family.

- **Date obtained:** September 11, 2026
- **Source URL:** Not recorded at time of download; could not be recovered from browser history as of September 23, 2026.

## License & Redistribution
Redistribution rights for this specific copy are unverified. The raw CSV is excluded from version control accordingly (see `data/raw/*.csv` in `.gitignore`).

## Integrity Verification

**SHA-256:**
```
16CD960780E25F88682CA86ACC56BE5016360B5468A60E3DD8F68F415716B77D
```

To verify the file has not changed:

```powershell
Get-FileHash -Algorithm SHA256 "data\raw\CCFraud_data.csv"
```

```bash
shasum -a 256 data/raw/CCFraud_data.csv
```

## Important Note
This dataset is synthetic/privacy-preserving data derived from an anonymized real-world bank-account fraud dataset. It should not be described as raw real-world banking transactions.