---
title: Insurance Policy Analyst
emoji: 🦀
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.8.0
app_file: app.py
pinned: false
---git a

# 🚗 Insurance Claims Analyst Agent

An AI-powered insurance claims analysis workflow that parses claim data, validates policy status, retrieves relevant policy language, and generates a structured recommendation with confidence scoring.

Built with **Python**, **smolagents**, **ChromaDB**, **Pydantic**, and **GPT-4.1**.

---

## Overview

This project analyzes auto insurance claims using a structured multi-step workflow instead of relying on a single prompt.

The system:

- Parses incoming claim data into a validated schema
- Verifies policy standing and deductible using a CSV source of truth
- Retrieves relevant policy sections from a vector database built from a policy PDF
- Generates a recommendation summary with a confidence score and explanation
- Finalizes payout calculation in code using deterministic business logic

The goal was to build an agent workflow that is more **grounded, auditable, and reliable** than a free-form chat response.

---

## Why This Project Matters

A lot of AI systems can produce an answer, but fewer are built in a way that makes them trustworthy in higher-stakes workflows.

This project was designed to separate:

- **Model-assisted reasoning** for recommendation and explanation
- **Deterministic code** for validation and payout calculation

That makes the workflow easier to inspect, debug, and trust.

---

## Features

- **Structured claim parsing** with Pydantic schemas
- **Policy validation** against `coverage_data.csv`
- **Deductible pulled from source-of-truth data**
- **Policy retrieval** using ChromaDB and embeddings
- **Confidence-scored recommendations** with readable confidence reasoning
- **Deterministic payout calculation** handled in code
- **Repair-cost sanity check** via web search
- **Dashboard-style Gradio UI**
- **JSON-based final decision output**
- **Grounded, multi-step workflow** instead of one-shot prompting

---

## Workflow

The agent follows a strict sequence:

1. **Parse Claim JSON** — `parse_claim()`
   Validates the incoming claim structure with a Pydantic schema.

2. **Validate Policy** — `is_valid_query()`
   Confirms the policy exists, checks for outstanding dues, and retrieves the deductible from `coverage_data.csv`.

3. **Generate Policy Queries** — `generate_policy_queries()`
   Uses the model to generate short keyword-based search terms for retrieving relevant policy sections.

4. **Retrieve Policy Text** — `retrieve_policy_text()`
   Searches the ChromaDB vector store for the most relevant policy chunks.

5. **Estimate Repair Costs** — `WebSearchTool()`
   Helps sanity-check whether the claimed repair amount is reasonable.

6. **Generate Recommendation** — `generate_recommendation()`
   Produces a structured recommendation with:
   - Policy section
   - Recommendation summary
   - Confidence score
   - Confidence reason

7. **Finalize Decision** — `finalize_decision()`
   Applies deterministic payout logic in code using validated deductible data and estimated repair cost.

---

## Example Output

```json
{
  "claim_number": "CLM-10045",
  "covered": true,
  "deductible": 500.0,
  "recommended_payout": 3200.0,
  "confidence_score": 0.87,
  "confidence_reason": "The claim is clearly described, supported by relevant policy language, and the validation data is complete.",
  "notes": "Collision damage appears covered under the relevant section. Final payout calculated as estimated repair cost ($3,700.00) minus deductible ($500.00)."
}
```

---

## Architecture

```
Claim Input
   ↓
parse_claim()
   ↓
is_valid_query()
   ↓
generate_policy_queries()
   ↓
retrieve_policy_text()
   ↓
WebSearchTool()
   ↓
generate_recommendation()
   ↓
finalize_decision()
   ↓
Gradio Dashboard Output
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core language |
| **smolagents** | Tool-calling agent orchestration |
| **OpenAI / GPT-4.1** | Reasoning and recommendation generation |
| **Pydantic** | Structured schemas and validation |
| **SentenceTransformers** | Embeddings for policy retrieval |
| **ChromaDB** | Vector storage for indexed policy text |
| **PyPDF2** | Policy PDF extraction |
| **Gradio** | Guided web interface and dashboard |
| **Web search tool** | Repair-cost sanity checks |

---

## File Requirements

This project expects the following local files:

| File | Description |
|---|---|
| `policy.pdf` | The policy document used to build the retrieval knowledge base |
| `coverage_data.csv` | Contains policy validation data: policy number, premium dues status, deductible |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/insurance-claims-analyst-agent
cd insurance-claims-analyst-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add required files

Place these in the project root:

- `policy.pdf`
- `coverage_data.csv`

### 4. Run the app locally

```bash
python app.py
```

---

## Deployment

This project is designed to be shared as an interactive Gradio app. The easiest way to deploy the live version is through **Hugging Face Spaces** — source code lives in GitHub while the working demo is hosted on Spaces for easier testing, sharing, and portfolio use.

### Deploy to Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Choose the **Gradio SDK**
3. Upload the project files or connect the GitHub repository
4. Add any required environment variables or secrets
5. Launch the Space

---

## Using the App

**Users provide:**

- OpenAI API key
- Optional base URL
- Claim number
- Policy number
- Claimant name
- Date of loss
- Estimated repair cost
- Vehicle details
- Loss description

**The app returns:**

- Claim status
- Recommended payout
- Deductible
- Coverage summary
- Confidence score
- Confidence reasoning

---

## Design Decisions

### 1. Deterministic payout logic

The model does not control the final payout amount. Final payout is calculated in code to reduce hallucination risk and keep critical business logic deterministic.

### 2. Grounded policy reasoning

The recommendation is based on retrieved policy text rather than general model memory.

### 3. Structured outputs

Schemas help ensure consistency and make the system easier to validate and debug.

### 4. Confidence as a usability feature

The confidence score is not treated as absolute truth, but as a useful signal for human review.

---

## Limitations

- The system depends on the quality of the uploaded policy PDF
- Repair-cost estimation uses web search and is only a sanity check
- Confidence scoring is model-generated and should be interpreted cautiously
- This is a prototype for claims analysis support, not a replacement for licensed claim adjudication

---

## Future Improvements

- Add formal evals for claim scenarios and failure cases
- Add better observability for tool-calling and workflow state
- Support multiple policy PDFs and policy types
- Add human-in-the-loop review queue
- Improve retrieval chunking and ranking
- Add persistent database storage for claims history
- Expand dashboard analytics for reviewer workflows

---

## License

MIT — free to use, modify, and build on.