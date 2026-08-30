# Student Internship Reflection Analysis Pipeline

A Python-based pipeline for analysing student internship reflections submitted as `.docx` and `.pdf` files.

The pipeline uses **Ollama-hosted large language models (LLMs)** to identify and evaluate seven reflective components from student writing. Extracted phrases are verified against the original text, scored according to predefined criteria, and aggregated into datasets for statistical analysis and visualisation.

The pipeline produces:

* Extracted reflection text
* Component-level reflective phrases
* `Present` / `Weak` / `Missing` scores
* LLM verification logs
* Summary statistics
* Student × week heatmap data
* Reflection word-count data
* Heatmap visualisations

---

## Table of Contents

* [Project Overview](#project-overview)
* [Pipeline Workflow](#pipeline-workflow)
* [Project Structure](#project-structure)
* [Requirements](#requirements)
* [Installation](#installation)
* [Data Organisation](#data-organisation)
* [Quick Start](#quick-start)
* [Detailed Workflow](#detailed-workflow)
* [Configuration](#configuration)
* [Output Files](#output-files)
* [Scoring](#scoring)
* [Verification and Quality Control](#verification-and-quality-control)
* [Troubleshooting](#troubleshooting)
* [Customisation](#customisation)

---

# Project Overview

This repository analyses student internship reflections across multiple weeks.

At a high level, the pipeline:

1. Extracts text from student `.docx` and `.pdf` submissions.
2. Stores the extracted reflections in a structured JSON file.
3. Uses an Ollama LLM to identify phrases representing seven reflective components.
4. Verifies that extracted phrases actually occur in the original reflection.
5. Optionally uses a second LLM as a judge to validate component assignments.
6. Scores each reflective component as `Present`, `Weak`, or `Missing`.
7. Aggregates the results into summary statistics.
8. Generates CSV data for heatmaps.
9. Generates word-count datasets for box plots.
10. Renders a final heatmap visualisation.

---

# Pipeline Workflow

```text
Student .docx / .pdf files
            │
            ▼
┌─────────────────────────┐
│ extract_reflections.py  │
│ Extract raw text        │
└────────────┬────────────┘
             │
             ▼
      reflections.json
             │
             ▼
┌─────────────────────────┐
│       pipeline.py       │
│                         │
│ • LLM phrase extraction │
│ • Phrase verification   │
│ • Optional LLM judge    │
│ • Component scoring     │
│ • Statistics            │
└────────────┬────────────┘
             │
             ├──────────────► phrases_output.json
             │
             ├──────────────► summary_stats.json
             │
             ├──────────────► fabrication_log.json
             │
             ├──────────────► soft_match_log.json
             │
             └──────────────► judge_log.json
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      ┌─────────────┐        ┌─────────────┐
      │ makecsv.py  │        │ boxplot.py  │
      └──────┬──────┘        └──────┬──────┘
             │                       │
             ▼                       ▼
    heatmap_data.csv       reflection_word_counts
                                     │
                                     ▼
                              ┌─────────────┐
                              │ heatmap.py  │
                              └──────┬──────┘
                                     │
                                     ▼
                       figure6_heatmap_sorted.png
```

---

# Project Structure

A typical repository should look like this:

```text
project/
├── extract_reflections.py
├── pipeline.py
├── makecsv.py
├── boxplot.py
├── heatmap.py
│
├── JFC_reflection part 2 2026 Jul/
│   ├── submissions/
│   │   └── *.docx / *.pdf
│   │
│   └── submissions (2)/
│       └── *.docx / *.pdf
│
├── reflections.json
├── heatmap_data.csv
├── reflection_word_counts.xlsx
├── reflection_word_counts.csv
│
└── v5_output/
    ├── phrases_output.json
    ├── summary_stats.json
    ├── fabrication_log.json
    ├── soft_match_log.json
    └── judge_log.json
```

> **Note:** The exact folder-to-week mapping is controlled by `FOLDER_MAP` in `extract_reflections.py`.

---

# Requirements

## Python

Python **3.8 or later** is required.

## Ollama

The pipeline requires a locally running Ollama installation.

The default configuration expects the following models:

```text
qwen2.5:7b
qwen3:14b
```

The first model is used for phrase extraction, while the second is optionally used as a verification/judge model.

---

# Installation

Install the required Python packages:

```bash
pip install python-docx pypdf pandas matplotlib openpyxl ollama
```

You can verify the installation with:

```bash
python --version
```

and:

```bash
ollama --version
```

---

# Data Organisation

Place the reflection data folder in the same parent directory as the Python scripts.

The default folder is:

```text
JFC_reflection part 2 2026 Jul/
```

The expected structure is:

```text
JFC_reflection part 2 2026 Jul/
├── submissions/
│   └── Week 5 reflection files
│
└── submissions (2)/
    └── Week 3 reflection files
```

Both `.docx` and `.pdf` files are supported.

## Important

The folder names are mapped to week labels through the `FOLDER_MAP` dictionary in:

```text
extract_reflections.py
```

If your folder structure changes, update `FOLDER_MAP` accordingly.

---

# Quick Start

Once the repository and data are configured, run the following commands.

## 1. Start Ollama

```bash
ollama serve
```

In another terminal, make sure the required models are available:

```bash
ollama pull qwen2.5:7b
ollama pull qwen3:14b
```

## 2. Extract Reflections

```bash
python extract_reflections.py
```

This generates:

```text
reflections.json
```

## 3. Run the Analysis Pipeline

```bash
python pipeline.py
```

This generates the main analysis outputs in:

```text
v5_output/
```

## 4. Generate Heatmap Data

```bash
python makecsv.py
```

This generates:

```text
heatmap_data.csv
```

## 5. Generate Word-Count Data

```bash
python boxplot.py
```

This generates:

```text
reflection_word_counts.xlsx
reflection_word_counts.csv
```

## 6. Generate the Heatmap

```bash
python heatmap.py
```

This generates:

```text
figure6_heatmap_sorted.png
```

---

# Detailed Workflow

## Step 1 — Extract Raw Reflections

Run:

```bash
python extract_reflections.py
```

The script scans the configured reflection directories and extracts text from all supported files.

The output is:

```text
reflections.json
```

Each reflection includes metadata such as:

* Student ID
* Week
* Filename
* Raw reflection text

---

## Step 2 — Run the LLM Analysis

Run:

```bash
python pipeline.py
```

The pipeline reads `reflections.json` and processes each relevant reflection.

For each reflection, it:

1. Sends the reflection to the configured Ollama model.
2. Extracts phrases associated with seven reflective components.
3. Checks whether each phrase appears verbatim in the original text.
4. Records phrases that fail verification.
5. Optionally sends extracted phrases to a second LLM judge.
6. Assigns component scores.
7. Calculates relevant statistics.
8. Saves the results to `v5_output/`.

### Pipeline Outputs

```text
v5_output/
├── phrases_output.json
├── summary_stats.json
├── fabrication_log.json
├── soft_match_log.json
└── judge_log.json
```

---

# Processing Specific Weeks

The current `pipeline.py` configuration is designed as a targeted re-run for **Weeks 4–7**.

It expects an existing:

```text
v5_output/phrases_output.json
```

If you want to process the entire dataset from scratch, modify `TARGET_WEEKS` near the top of `pipeline.py`.

For example:

```python
TARGET_WEEKS = {
    "Week 1 Reflection",
    "Week 2 Reflection",
    "Week 3 Reflection",
    "Week 4 Reflection",
    "Week 5 Reflection",
    "Week 6 Reflection",
    "Week 7 Reflection",
    "Week 8 Reflection",
}
```

Alternatively, remove or disable the filtering logic in `main()` that restricts processing to `TARGET_WEEKS`.

---

# Step 3 — Generate Heatmap Data

Run:

```bash
python makecsv.py
```

The script reads:

```text
v5_output/phrases_output.json
```

and generates:

```text
heatmap_data.csv
```

The resulting dataset is structured as a **student × week matrix**.

Example:

| Student     | Week 1  | Week 2  | Week 3  | Week 4  |
| ----------- | ------- | ------- | ------- | ------- |
| Student 001 | Present | Weak    | Missing | Present |
| Student 002 | Weak    | Present | Present | Missing |
| Student 003 | Missing | Weak    | Present | Present |

The values represent the emotional-response component score.

Possible values include:

* `Present`
* `Weak`
* `Missing`
* `No submission`

---

# Step 4 — Generate Word-Count Data

Run:

```bash
python boxplot.py
```

The script generates:

```text
reflection_word_counts.xlsx
reflection_word_counts.csv
```

The Excel workbook contains two sheets:

### Raw Word Counts

Contains the original word-count data for each reflection.

### Box-Plot Data

Contains data formatted for subsequent box-plot analysis.

The CSV provides a simpler representation containing:

```text
Week
Word Count
```

---

# Step 5 — Generate the Heatmap

Run:

```bash
python heatmap.py
```

The script reads:

```text
heatmap_data.csv
```

and generates:

```text
figure6_heatmap_sorted.png
```

The resulting figure provides a visual representation of emotional-response scores across students and weeks.

---

# Configuration

The primary configuration options are located near the top of `pipeline.py`.

| Variable                      | Default / Purpose                                                            |
| ----------------------------- | ---------------------------------------------------------------------------- |
| `MODEL`                       | `"qwen2.5:7b"` — primary phrase-extraction model                             |
| `JUDGE_MODEL`                 | `"qwen3:14b"` — optional verification model                                  |
| `ENABLE_LLM_JUDGE`            | Enables or disables the second-pass LLM judge                                |
| `ENABLE_LENGTH_NORMALIZATION` | Calculates `phrases_per_100_words` when enabled                              |
| `ENABLE_HIERARCHY_GATE`       | Checks whether higher-order components occur without foundational components |
| `MAX_RETRIES`                 | Controls the number of retries for failed LLM requests                       |
| `DELAY_SECS`                  | Controls the delay between LLM requests                                      |
| `TARGET_WEEKS`                | Controls which weeks are processed                                           |

---

## Changing the Extraction Model

To use a different Ollama model, change:

```python
MODEL = "qwen2.5:7b"
```

For example:

```python
MODEL = "your-model-name"
```

Make sure the model has already been pulled:

```bash
ollama pull your-model-name
```

---

## Disabling the LLM Judge

The second-pass judge can be disabled to reduce processing time:

```python
ENABLE_LLM_JUDGE = False
```

This is useful when:

* Processing a large number of reflections
* Testing the pipeline
* Running on hardware with limited resources
* The additional verification step is not required

---

# Output Files

| File                             | Description                                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `reflections.json`               | Raw reflection text and associated metadata                                                             |
| `v5_output/phrases_output.json`  | Main analysis dataset containing extracted phrases, scores, topics, word counts, and gating information |
| `v5_output/summary_stats.json`   | Aggregate and per-week statistics                                                                       |
| `v5_output/fabrication_log.json` | Extracted phrases that could not be found verbatim in the original text                                 |
| `v5_output/soft_match_log.json`  | Phrases that matched only after normalisation                                                           |
| `v5_output/judge_log.json`       | Phrases or assignments rejected by the LLM judge                                                        |
| `heatmap_data.csv`               | Student × week emotional-response matrix                                                                |
| `reflection_word_counts.xlsx`    | Excel workbook containing word-count and box-plot datasets                                              |
| `reflection_word_counts.csv`     | CSV version of the word-count data                                                                      |
| `figure6_heatmap_sorted.png`     | Final heatmap visualisation                                                                             |

---

# Scoring

Each reflective component is assigned one of three scores:

| Score       | Meaning                                            |
| ----------- | -------------------------------------------------- |
| **Present** | At least 2 qualifying units/phrases are identified |
| **Weak**    | 1 qualifying unit/phrase is identified             |
| **Missing** | No qualifying units/phrases are identified         |

The scoring thresholds are implemented in:

```python
score_from_phrases
```

in `pipeline.py`.

To change the thresholds, modify this function.

---

# Verification and Quality Control

A key feature of the pipeline is that extracted phrases are checked against the original reflection.

## Verbatim Verification

The pipeline checks whether each phrase returned by the LLM exists as a substring of the source text.

This helps prevent the model from introducing phrases that were not actually written by the student.

Phrases that cannot be verified are recorded in:

```text
v5_output/fabrication_log.json
```

---

## Soft Matching

Some phrases may differ from the original text only because of formatting differences, such as:

* Unicode characters
* Capitalisation
* Quotation marks
* Normalisation differences

These cases are recorded in:

```text
v5_output/soft_match_log.json
```

---

## LLM Judge

An optional second LLM can be used to independently evaluate extracted component assignments.

The judge model is configured through:

```python
JUDGE_MODEL = "qwen3:14b"
```

Rejected phrases or assignments are recorded in:

```text
v5_output/judge_log.json
```

---

# Troubleshooting

## Ollama Connection Refused

If the pipeline cannot connect to Ollama, ensure the service is running:

```bash
ollama serve
```

Also check that the configured models exist:

```bash
ollama list
```

---

## Model Not Found

If you receive a model-not-found error, pull the required model:

```bash
ollama pull qwen2.5:7b
```

or:

```bash
ollama pull qwen3:14b
```

If you use a different model, update the corresponding variables in `pipeline.py`.

---

## Missing Python Package

If you encounter an import error such as:

```text
ModuleNotFoundError
```

install the required dependencies:

```bash
pip install python-docx pypdf pandas matplotlib openpyxl ollama
```

---

## Empty Reflection Text

If reflections are empty or unexpectedly short:

1. Check that the source files are valid.
2. Confirm that they are `.docx` or `.pdf`.
3. Check the folder structure.
4. Inspect `extract_reflections.py`.
5. Check whether the document uses tables or other non-standard formatting.

Some table-based Word documents may require changes to the extraction logic.

---

## Large Number of Fabrications

A high number of entries in:

```text
fabrication_log.json
```

may indicate that the LLM is paraphrasing rather than returning exact phrases from the reflection.

The verification system is designed to detect and remove these phrases.

Consider:

* Strengthening the extraction prompt
* Using a more capable model
* Reducing the number of phrases requested
* Reviewing the `EXTRACTION_PROMPT` configuration

---

## Slow Processing

LLM processing can be computationally intensive.

If the pipeline is too slow, consider:

* Using a smaller model
* Disabling the LLM judge
* Reducing `MAX_RETRIES`
* Increasing `DELAY_SECS`
* Processing a smaller set of weeks using `TARGET_WEEKS`

For example:

```python
ENABLE_LLM_JUDGE = False
```

---

# Customisation

## Change Week Mapping

Modify:

```text
FOLDER_MAP
```

in:

```text
extract_reflections.py
```

This controls how input folders are translated into week labels.

---

## Change Reflective Components

The reflective component definitions are stored in:

```text
COMPONENT_DEFINITIONS
```

in `pipeline.py`.

Modify these definitions to change the criteria used for identifying reflective components.

---

## Change the Extraction Prompt

The main LLM extraction instructions are defined in:

```text
EXTRACTION_PROMPT
```

in `pipeline.py`.

This can be modified to adjust:

* Extraction behaviour
* Component interpretation
* Phrase-selection criteria
* Output requirements
* Verification expectations

---

## Change Scoring Thresholds

Modify:

```text
score_from_phrases
```

in `pipeline.py`.

The default thresholds are:

```text
2+ units → Present
1 unit   → Weak
0 units  → Missing
```

---

# Reproducibility

For consistent results, record the following when running the pipeline:

* Python version
* Ollama version
* Extraction model
* Judge model
* Model versions
* Configuration flags
* `TARGET_WEEKS`
* Source dataset/version

LLM-based extraction can vary between model versions and configurations, so documenting the environment is recommended when results are being used for research or reporting.

---

# Typical End-to-End Run

For a complete run from raw submissions to final visualisation:

```bash
# Start Ollama
ollama serve

# Pull models
ollama pull qwen2.5:7b
ollama pull qwen3:14b

# Extract reflection text
python extract_reflections.py

# Run LLM analysis
python pipeline.py

# Generate heatmap dataset
python makecsv.py

# Generate word-count data
python boxplot.py

# Generate heatmap
python heatmap.py
```

The final outputs will include:

```text
v5_output/
├── phrases_output.json
├── summary_stats.json
├── fabrication_log.json
├── soft_match_log.json
└── judge_log.json

heatmap_data.csv
reflection_word_counts.xlsx
reflection_word_counts.csv
figure6_heatmap_sorted.png
```

---

# Summary

This pipeline provides an end-to-end workflow for transforming raw student internship reflections into structured, analysable data.

The main stages are:

**Extract → Analyse → Verify → Score → Aggregate → Visualise**

The combination of LLM-based extraction and source-text verification is intended to make the analysis scalable while reducing the risk of unsupported or fabricated evidence being included in the results.
