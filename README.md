# Bank Statement – Customer Name Matcher

Streamlit application that extracts potential customer names from a bank statement PDF, compares them against a customer list, and surfaces the matched customer details.

## Prerequisites

- Python 3.9+
- The following Python packages (install via `pip install -r requirements.txt`):
  - streamlit
  - pandas
  - pymupdf
  - rapidfuzz

## Running the app

```bash
cd "/home/Adi/Projects/Bank-Statement Extractor"
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Usage workflow

1. Upload the bank statement PDF, customer names CSV (with a `CustomerName` column), and customer details CSV (also containing `CustomerName`).
2. Adjust the fuzzy-match score threshold from the sidebar if needed.
3. Click **Extract & Match** to generate matches.
4. Review the extracted names, matched customers with confidence scores, and merged customer details.
5. Download the matched names or detailed results as CSV files.

All processing happens locally within the Streamlit session; no data leaves your machine.
