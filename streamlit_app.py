import io
import re
import string
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - surfaced in UI
    fitz = None

st.set_page_config(
    page_title="Bank Statement – Customer Name Matcher",
    page_icon="📄",
    layout="wide",
)

APP_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = APP_ROOT / "sample_data"
SAMPLE_FILES = {
    "pdf": SAMPLE_DIR / "bank_statement.pdf",
    "customers": SAMPLE_DIR / "customer_names.csv",
    "details": SAMPLE_DIR / "customer_details.csv",
}

_NAME_PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)


@st.cache_data(show_spinner=False)
def load_csv(data: bytes) -> pd.DataFrame:
    """Load a CSV file from raw bytes."""
    return pd.read_csv(io.BytesIO(data))


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF using PyMuPDF."""
    if fitz is None:
        raise ImportError(
            "PyMuPDF (package `pymupdf`) is required to extract text from PDFs."
        )

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        text_chunks = [page.get_text("text") for page in document]
    return "\n".join(text_chunks)


def extract_candidate_names(text: str) -> List[str]:
    """Pull out likely customer names from raw PDF text."""
    if not text:
        return []

    title_case_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")
    all_caps_pattern = re.compile(r"\b([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})\b")

    candidates = set()

    for match in title_case_pattern.findall(text):
        cleaned = match.strip()
        if _is_plausible_name(cleaned):
            candidates.add(cleaned)

    for match in all_caps_pattern.findall(text):
        cleaned = match.title().strip()
        if _is_plausible_name(cleaned):
            candidates.add(cleaned)

    return sorted(candidates)


def _is_plausible_name(name: str) -> bool:
    """Basic heuristics to reduce false positives from PDF text."""
    parts = name.split()
    if len(parts) < 2:
        return False
    if any(len(part) == 1 for part in parts):
        return False
    if any(part.isdigit() for part in parts):
        return False
    return True


def normalize_name(name: str) -> str:
    """Normalize a name for fuzzy comparison."""
    lowered = name.lower().strip()
    normalized = lowered.translate(_NAME_PUNCT_TRANSLATOR)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def match_names(
    pdf_names: List[str],
    customers: pd.DataFrame,
    threshold: int,
) -> pd.DataFrame:
    """Run fuzzy matching between PDF-derived names and customer records."""
    if customers.empty or not pdf_names:
        return pd.DataFrame(columns=["PDF Name", "CustomerName", "Match Score"])

    working = customers.copy()
    working["__normalized"] = working["CustomerName"].astype(str).apply(normalize_name)

    choices = working["__normalized"].tolist()
    matches = []

    for pdf_name in pdf_names:
        normalized_pdf = normalize_name(pdf_name)
        if not normalized_pdf:
            continue

        result = process.extractOne(
            query=normalized_pdf,
            choices=choices,
            scorer=fuzz.WRatio,
        )

        if result is None:
            continue

        matched_value, score, index = result
        if score < threshold:
            continue

        row = working.iloc[index]
        matches.append(
            {
                "PDF Name": pdf_name,
                "CustomerName": row["CustomerName"],
                "Match Score": round(float(score), 2),
            }
        )

    if not matches:
        return pd.DataFrame(columns=["PDF Name", "CustomerName", "Match Score"])

    matched_df = pd.DataFrame(matches)
    matched_df.sort_values(by=["Match Score", "CustomerName"], ascending=[False, True], inplace=True)
    matched_df.drop_duplicates(subset="CustomerName", keep="first", inplace=True)
    matched_df.reset_index(drop=True, inplace=True)
    return matched_df


def merge_with_details(matches: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    """Merge matched names with their additional details."""
    if matches.empty:
        return matches
    merged = matches.merge(details, on="CustomerName", how="left")
    return merged


def main() -> None:
    st.title("Bank Statement – Customer Name Matcher")
    st.markdown(
        """
        Upload a bank statement PDF and customer datasets to automatically identify overlapping
        names. Matched entries include confidence scores and customer details that you can review
        or export.
        """
    )

    st.sidebar.header("Matching Settings")
    use_samples = st.sidebar.toggle(
        "Use sample files",
        value=False,
        help="Automatically load the demo PDF and CSVs shipped with the app for a quick test run.",
    )
    threshold = st.sidebar.slider(
        "Minimum match score",
        min_value=60,
        max_value=100,
        value=85,
        step=1,
        help="Lower the threshold to surface more matches, raise it to focus on high-confidence pairs.",
    )
    
    show_file_details = st.sidebar.toggle(
        "Show file details",
        value=False,
        help="Display a preview of the loaded files (sample or uploaded).",
    )
    
    st.sidebar.info(
        "Processing happens locally inside this Streamlit session. No files leave your machine."
    )

    st.subheader("1. Upload files")
    pdf_file = st.file_uploader(
        "Bank Statement (PDF)",
        type=["pdf"],
        accept_multiple_files=False,
        disabled=use_samples,
    )

    col1, col2 = st.columns(2)
    with col1:
        customer_file = st.file_uploader(
            "Customer Names (CSV)",
            type=["csv"],
            accept_multiple_files=False,
            key="customer-csv",
            help="Must include a 'CustomerName' column.",
            disabled=use_samples,
        )
    with col2:
        details_file = st.file_uploader(
            "Customer Details (CSV)",
            type=["csv"],
            accept_multiple_files=False,
            key="details-csv",
            help="Must include a 'CustomerName' column plus any detail columns you want returned.",
            disabled=use_samples,
        )

    pdf_bytes = customer_bytes = details_bytes = None

    if use_samples:
        missing = [label for label, path in SAMPLE_FILES.items() if not path.exists()]
        if missing:
            missing_labels = ", ".join(missing)
            st.error(
                f"Sample files missing: {missing_labels}. Upload your own files or restore the sample_data folder."
            )
        else:
            pdf_bytes = SAMPLE_FILES["pdf"].read_bytes()
            customer_bytes = SAMPLE_FILES["customers"].read_bytes()
            details_bytes = SAMPLE_FILES["details"].read_bytes()
            st.info(
                "Using bundled demo files from the `sample_data` directory. Toggle off to upload your own."
            )

    if show_file_details:
        st.subheader("📋 File Details Preview")
        try:
            if use_samples:
                if all([SAMPLE_FILES["customers"].exists(), SAMPLE_FILES["details"].exists()]):
                    customers_df = pd.read_csv(SAMPLE_FILES["customers"])
                    details_df = pd.read_csv(SAMPLE_FILES["details"])
                    
                    st.write("**Sample Customer Names:**")
                    st.dataframe(customers_df, width='content', hide_index=True)
                    
                    st.write("**Sample Customer Details:**")
                    st.dataframe(details_df, width='content', hide_index=True)
            else:
                if customer_file is not None and details_file is not None:
                    customers_df = pd.read_csv(customer_file)
                    details_df = pd.read_csv(details_file)
                    
                    st.write("**Uploaded Customer Names:**")
                    st.dataframe(customers_df, width='content', hide_index=True)
                    
                    st.write("**Uploaded Customer Details:**")
                    st.dataframe(details_df, width='content', hide_index=True)
                else:
                    st.info("Please upload CSV files to preview their contents.")
        except Exception as exc:
            st.error(f"Unable to preview files: {exc}")

    st.markdown("---")
    if st.button("Extract & Match", type="primary", width='stretch'):
        if use_samples:
            if not all([pdf_bytes, customer_bytes, details_bytes]):
                st.error("Sample files could not be loaded. Please upload your own inputs instead.")
                return
        else:
            if not all([pdf_file, customer_file, details_file]):
                st.warning("Please upload all three files before processing.")
                return
            pdf_bytes = pdf_file.getvalue()
            customer_bytes = customer_file.getvalue()
            details_bytes = details_file.getvalue()

        if not all([pdf_bytes, customer_bytes, details_bytes]):
            st.warning("Please upload all three files before processing.")
            return

        with st.spinner("Extracting text and running matches..."):
            try:
                statement_text = extract_pdf_text(pdf_bytes)
            except Exception as exc:  # surfaced to user for debugging
                st.error(f"Unable to extract text from the PDF: {exc}")
                return

            candidate_names = extract_candidate_names(statement_text)

            try:
                customers_df = load_csv(customer_bytes)
            except Exception as exc:
                st.error(f"Unable to read the customer names CSV: {exc}")
                return

            if "CustomerName" not in customers_df.columns:
                st.error("The customer names CSV must contain a 'CustomerName' column.")
                return

            try:
                details_df = load_csv(details_bytes)
            except Exception as exc:
                st.error(f"Unable to read the details CSV: {exc}")
                return

            if "CustomerName" not in details_df.columns:
                st.error("The details CSV must contain a 'CustomerName' column.")
                return

            matches_df = match_names(candidate_names, customers_df, threshold)
            merged_df = merge_with_details(matches_df, details_df)

        st.subheader("2. Extracted names")
        st.caption("Names detected in the PDF after basic cleaning.")
        if candidate_names:
            extracted_preview = pd.DataFrame({"Extracted Names": candidate_names})
            st.dataframe(extracted_preview, width='stretch', hide_index=True)
        else:
            st.info("No plausible customer names were found in the uploaded PDF.")

        st.subheader("3. Matched customers")
        if matches_df.empty:
            st.warning(
                "No matches cleared the selected score threshold. Try lowering the score or reviewing the input data."
            )
            return

        st.success(f"Matched {len(matches_df)} customer(s) with a minimum score of {threshold}%.")
        st.dataframe(matches_df, width='stretch')

        matches_csv = matches_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download matched names CSV",
            data=matches_csv,
            file_name="matched_names.csv",
            mime="text/csv",
        )

        st.subheader("4. Matched customer details")
        if merged_df.empty:
            st.info("No additional details were found for the matched customers.")
        else:
            st.dataframe(merged_df, width='stretch')
            details_csv = merged_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download matched details CSV",
                data=details_csv,
                file_name="matched_customer_details.csv",
                mime="text/csv",
            )

        with st.expander("PDF text preview"):
            snippet = statement_text[:5000]
            suffix = "..." if len(statement_text) > 5000 else ""
            st.text(snippet + suffix)


if __name__ == "__main__":
    main()
