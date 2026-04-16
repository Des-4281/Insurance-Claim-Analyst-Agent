import os
import json
import re
import datetime
import gradio as gr
import chromadb
import PyPDF2
from typing import List, Optional
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from smolagents import (
    tool, 
    ToolCallingAgent, 
    WebSearchTool, 
    OpenAIServerModel, 
    PromptTemplates, 
    PlanningPromptTemplate, 
    ManagedAgentPromptTemplate, 
    FinalAnswerPromptTemplate
)

# ==========================================
# 1. SCHEMAS
# ==========================================
class ClaimInfo(BaseModel):
    claim_number: str
    policy_number: str
    claimant_name: str
    date_of_loss: str
    loss_description: str
    estimated_repair_cost: float
    vehicle_details: Optional[str] = None

class PolicyQueries(BaseModel):
    queries: List[str] = Field(default_factory=list)

class PolicyRecommendation(BaseModel):
    policy_section: str
    recommendation_summary: str
    deductible: Optional[float] = Field(
        default=None,
        description="The deductible pulled from coverage_data.csv or inferred from validation output."
    )
    settlement_amount: Optional[float] = Field(
        default=None,
        description="Optional raw estimate from the model before final payout math."
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence from 0.0 to 1.0."
    )
    confidence_reason: str = Field(
        ...,
        description="A short readable 1-2 sentence summary explaining why the confidence score was chosen."
    )

class ClaimDecision(BaseModel):
    claim_number: str
    covered: bool
    deductible: float
    recommended_payout: float
    confidence_score: float
    confidence_reason: str
    notes: Optional[str] = None

# ==========================================
# 2. IMMEDIATE BACKEND INDEXING (Baked-in Knowledge)
# ==========================================
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="auto_insurance_policy")

def initialize_policy_db():
    policy_file = "policy.pdf"
    if os.path.exists(policy_file) and collection.count() == 0:
        print("Indexing Knowledge Base...")
        with open(policy_file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            policy_text = "".join([(page.extract_text() or "").replace("\u2028", " ").replace("\u2029", " ") for page in reader.pages])
        
        # There was an issue where the entire policy pdf was being passed in, potentially due to incorrect scraping of n/n/ so switched to characters
        chunk_size = 1000
        policy_chunks = [policy_text[i:i + chunk_size] for i in range(0, len(policy_text), chunk_size)]
        
        ids = [f"chunk_{i}" for i in range(len(policy_chunks))]
        embeddings = embedder.encode(policy_chunks).tolist()
        collection.add(documents=policy_chunks, embeddings=embeddings, ids=ids)
        print(f"Knowledge Base Ready: {len(policy_chunks)} chunks indexed.")

initialize_policy_db()

# Global LLM placeholder for inner tool usage (updated per session)
llm_model = None
last_decision_json = None

# ==========================================
# 3. CUSTOM TOOLS (Restored & Refined)
# ==========================================
@tool
def parse_claim(json_data: str) -> str:
    """Parses claim JSON data to validate structure.
    
    Args:
        json_data: The raw JSON string containing the claim data.
    """
    try:
        data = json.loads(json_data)
        claim_info = ClaimInfo.model_validate(data)
        return claim_info.model_dump_json()
    except Exception as e:
        return f"Error parsing claim: {str(e)}"

@tool
def is_valid_query(query: str) -> str:
    """
    Validates policy standing and returns the policy-specific deductible.
    Args:
        query: The parsed claim JSON string.
    """
    try:
        claim_info = ClaimInfo.model_validate_json(query)

        import csv
        if not os.path.exists("coverage_data.csv"):
            return json.dumps({
                "valid": False,
                "message": "System Error: Coverage database not found.",
                "deductible": None
            })

        with open("coverage_data.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            policy = next(
                (p for p in reader if p["policy_number"] == claim_info.policy_number),
                None
            )

        if not policy:
            return json.dumps({
                "valid": False,
                "message": "Policy not found.",
                "deductible": None
            })

        dues = str(policy.get("premium_dues_remaining", "")).strip().lower() in ("true", "1", "yes")
        if dues:
            return json.dumps({
                "valid": False,
                "message": "Outstanding dues found.",
                "deductible": None
            })

        raw_deductible = str(policy.get("deductible", "")).strip()
        try:
            deductible = float(raw_deductible) if raw_deductible else 500.0
        except ValueError:
            deductible = 500.0

        return json.dumps({
            "valid": True,
            "message": "Valid claim.",
            "deductible": deductible
        })

    except Exception as e:
        return json.dumps({
            "valid": False,
            "message": f"Error during CSV validation: {str(e)}",
            "deductible": None
        })

@tool
def generate_policy_queries(claim_info_json: str) -> str:
    """Generate queries to retrieve relevant policy sections based on claim info.
    
    Args:
        claim_info_json: A JSON string containing the parsed claim details.
    """
    global llm_model
    prompt = f"""
    Analyze the following auto insurance claim and generate exactly 2 short, keyword-based search queries to find the right policy sections.
    - Example good queries: "collision coverage", "deductible limits", "exclusions".
    - DO NOT write full sentences.
    - Claim Data: {claim_info_json}
    - Return a JSON object strictly matching this schema: {{"queries": ["keyword 1", "keyword 2"]}}
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm_model(messages)
        response_content = response.content if hasattr(response, 'content') else str(response)
        result = json.loads(response_content)
        return json.dumps(result)
    except Exception as e:
        return f"Error generating policy queries: {str(e)}"

@tool
def retrieve_policy_text(queries_json: str) -> str:
    """Retrieves policy text from ChromaDB using generated queries.
    
    Args:
        queries_json: A JSON string containing a list of search queries.
    """
    try:
        queries_data = json.loads(queries_json)
        
        # LLM Generated raw lists occasionally, so defensive code was deployed to handle any raw lists and turn them into readable strings
        if isinstance(queries_data, list):
            query_strings = queries_data
        elif isinstance(queries_data, dict):
            query_strings = queries_data.get("queries", [])
        else:
            return "Error: Input must be a list of strings or a dict containing a 'queries' list."
            
        policy_texts = []
        for q in query_strings:
            # Safely extract string if the LLM passes a list of dictionaries
            if isinstance(q, dict):
                q = q.get("query", str(q))
                
            query_embedding = embedder.encode([str(q)])[0].tolist()
            results = collection.query(query_embeddings=[query_embedding], n_results=1)
            if results['documents'] and len(results['documents'][0]) > 0:
                policy_texts.extend(results['documents'][0])
                
        # Combine text and enforce the 4000-character limit to prevent token 30k Limit crashes.
        combined_text = "\n\n".join(set(policy_texts))
        if len(combined_text) > 4000:
            return combined_text[:4000] + "\n... [Text Truncated to save memory]"
            
        return combined_text
        
    except json.JSONDecodeError:
        return "Error: Invalid JSON format provided to the tool."
    except Exception as e:
        return f"Error retrieving policy text: {str(e)}"

@tool
def generate_recommendation(claim_info_json: str, policy_text: str) -> str:
    """
    Generate a policy recommendation based on claim info and retrieved policy text.
    Args:
        claim_info_json: The validated claim info in JSON format.
        policy_text: The relevant text retrieved from the policy documents.
    """
    global llm_model

    prompt = f"""
Evaluate the following auto insurance claim against the policy text.
Requirements:
- Determine whether the claim appears covered.
- Include the applicable policy section.
- Include a concise recommendation summary.
- Include a confidence_score from 0.0 to 1.0.
- Include a confidence_reason that is a short readable 1-2 sentence summary.
- Base the confidence_reason on claim clarity, policy support, and data quality.
- Keep the summary concise and directly tied to real evidence in the claim and policy text.
- You may include an optional settlement_amount estimate, but final payout is calculated later by code.
Claim Info:
{claim_info_json}
Policy Text:
{policy_text}
Return JSON only, strictly matching this schema:
{{
  "policy_section": "str",
  "recommendation_summary": "str",
  "deductible": null,
  "settlement_amount": float or null,
  "confidence_score": float,
  "confidence_reason": "str"
}}
"""

    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm_model(messages)
        response_content = response.content if hasattr(response, "content") else str(response)

        result = json.loads(response_content)

        # We do NOT trust the LLM as source of truth for deductible in v7
        result["deductible"] = None

        PolicyRecommendation.model_validate(result)
        return json.dumps(result)

    except Exception as e:
        return f"Error generating recommendation: {str(e)}"

@tool
def finalize_decision(
    claim_info_json: str | dict,
    validation_json: str | dict,
    recommendation_json: str | dict
) -> str:
    """
    Finalize the claim decision using validation data as the source of truth for deductible math.
    Args:
        claim_info_json: The validated claim information.
        validation_json: The validation output containing policy validity and deductible.
        recommendation_json: The AI-generated recommendation output.
    """
    global last_decision_json

    try:
        if isinstance(claim_info_json, dict):
            claim_info = ClaimInfo.model_validate(claim_info_json)
        else:
            claim_info = ClaimInfo.model_validate_json(claim_info_json)

        if isinstance(validation_json, dict):
            validation_data = validation_json
        else:
            validation_data = json.loads(validation_json)

        if isinstance(recommendation_json, dict):
            rec_data = recommendation_json
        else:
            rec_data = json.loads(recommendation_json)

        valid = bool(validation_data.get("valid", False))
        deductible = float(validation_data.get("deductible") or 0.0)
        confidence = float(rec_data.get("confidence_score") or 0.0)
        confidence_reason = str(rec_data.get("confidence_reason") or "Unclear basis")
        repair_cost = float(claim_info.estimated_repair_cost or 0.0)
        summary = rec_data.get("recommendation_summary", "")

        covered = (
            valid and
            "not covered" not in summary.lower()
        )

        payout = max(0.0, repair_cost - deductible) if covered else 0.0

        decision = ClaimDecision(
            claim_number=claim_info.claim_number,
            covered=covered,
            deductible=deductible,
            recommended_payout=payout,
            confidence_score=confidence,
            confidence_reason=confidence_reason,
            notes=(
                f"{summary} Final payout calculated as estimated repair cost "
                f"(${repair_cost:,.2f}) minus deductible (${deductible:,.2f})."
            )
        )

        last_decision_json = decision.model_dump_json(indent=2)
        return last_decision_json
        
    except Exception as e:
        return f"Error finalizing decision: {str(e)}"

# ==========================================
# 4. PROMPT TEMPLATES 
# ==========================================
system_prompt = """
You are an expert insurance claim-processing agent specializing in auto insurance.
You follow a strict, multi-step reasoning process.
CLAIM PROCESSING ORDER (MANDATORY):
1. Parse the claim JSON using `parse_claim`.
2. Validate the claim using `is_valid_query`.
   - Read the returned JSON carefully.
   - If "valid" is false, STOP immediately and return an invalid-claim decision.
   - If "valid" is true, preserve the full validation output for later steps.
3. Generate policy-related search queries using `generate_policy_queries`.
4. Retrieve relevant policy text using `retrieve_policy_text`.
5. Use the web search tool to estimate typical repair costs for the described damage.
   If the claimed amount is clearly unreasonable, reflect that in your recommendation.
6. Use `generate_recommendation`.
   - Include a confidence_score from 0.0 to 1.0.
   - Include a short readable 1-2 sentence confidence_reason explaining the score based on claim clarity, policy support, and data quality.
   - This step is for coverage reasoning only.
7. Use `finalize_decision` with the parsed claim, the validation output, and the recommendation output.
   - This step performs the final payout calculation in code using estimated_repair_cost minus deductible.
   - Do not rely on settlement_amount alone for the final payout.
ALWAYS follow this exact sequence.
Do not reorder, skip, or combine steps.
"""

prompt_templates = PromptTemplates(
    system_prompt=system_prompt,
    planning=PlanningPromptTemplate(
        initial_facts="Claim details:\n{claim_info_json}\nPolicy details:\n{policy_text}",
        initial_plan="Follow the strict claim processing sequence: Parse -> Validate -> Query -> Retrieve -> Web Search Estimate -> Recommend -> Finalize.",
        update_facts_pre_messages="Reassess facts:",
        update_facts_post_messages="Facts updated.",
        update_plan_pre_messages="Revise plan based on new facts:",
        update_plan_post_messages="Plan updated."
    ),
    managed_agent=ManagedAgentPromptTemplate(
        task="Process claim: {task_description}",
        report="Generate final decision: {results}"
    ),
    final_answer=FinalAnswerPromptTemplate(
        
        pre_messages="""Return only the final claim decision as valid JSON.
        Do not add markdown.
        Do not add headings.
        Do not add explanation outside the JSON.
        Do not wrap the JSON in code fences.
        The output must be a single valid JSON object matching the final decision structure.""",
        post_messages="Return only valid JSON.",
        final_answer_template="""{final_answer}"""
    )
)

# ==========================================
# 5. STATELESS PROCESSING GATEKEEPER
# ==========================================
def build_dashboard_html(decision: dict) -> str:
    covered = bool(decision.get("covered", False))
    payout = float(decision.get("recommended_payout", 0.0) or 0.0)
    deductible = float(decision.get("deductible", 0.0) or 0.0)
    confidence = float(decision.get("confidence_score", 0.0) or 0.0)
    confidence_pct = max(0, min(100, int(confidence * 100)))
    claim_number = decision.get("claim_number", "Unknown")
    confidence_summary = decision.get("confidence_reason", "No confidence summary provided.")
    notes = decision.get("notes", "")

    if covered and confidence >= 0.75:
        status_icon = "✅"
        status_text = "Claim Approved"
        status_class = "approved"
    elif covered:
        status_icon = "⚠️"
        status_text = "Claim Needs Review"
        status_class = "review"
    else:
        status_icon = "❌"
        status_text = "Claim Denied"
        status_class = "denied"

    return f"""
    <div class="claim-dashboard">
        <div class="claim-id">Claim #{claim_number}</div>
        <div class="status-banner {status_class}">
            <div class="status-icon">{status_icon}</div>
            <div class="status-title">{status_text}</div>
        </div>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">${payout:,.2f}</div>
                <div class="metric-label">Recommended Payout</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${deductible:,.2f}</div>
                <div class="metric-label">Deductible</div>
            </div>
        </div>
        <div class="summary-card">
            <div class="section-title">Coverage Summary</div>
            <div class="summary-text">{notes}</div>
        </div>
        <div class="confidence-card">
            <div class="confidence-left">
                <div class="progress-ring {status_class}" style="--p:{confidence_pct};">
                    <div class="progress-inner">{confidence_pct}%</div>
                </div>
                <div class="confidence-score">{confidence:.2f}</div>
            </div>
            <div class="confidence-right">
                <div class="section-title">Confidence Summary</div>
                <div class="confidence-text">{confidence_summary}</div>
            </div>
        </div>
    </div>
    """

def ui_process_claim(api_key, base_url, claim_no, policy_no, name, date, cost, vehicle, desc):
    if not api_key or not api_key.startswith("sk-"):
        return (
            "<div style='color:#ef4444;font-weight:700;'>Invalid API key.</div>",
            "### ❌ Error\nPlease provide a valid OpenAI API Key in the Settings tab."
        )

    payload = {
        "claim_number": claim_no,
        "policy_number": policy_no,
        "claimant_name": name,
        "date_of_loss": date,
        "loss_description": desc,
        "estimated_repair_cost": cost,
        "vehicle_details": vehicle
    }

    return execute_agent_workflow(api_key, base_url, json.dumps(payload))

def execute_agent_workflow(api_key, base_url, claim_json):
    global llm_model, last_decision_json
    os.environ['OPENAI_API_KEY'] = api_key
    os.environ['OPENAI_BASE_URL'] = base_url or "https://api.openai.com/v1"
    
    last_decision_json = None
    
    llm_model = OpenAIServerModel(
        model_id="gpt-4.1",
        api_base=os.environ['OPENAI_BASE_URL'],
        api_key=os.environ['OPENAI_API_KEY']
    )

    agent = ToolCallingAgent(
        tools=[
            parse_claim,
            is_valid_query,
            generate_policy_queries,
            retrieve_policy_text,
            generate_recommendation,
            finalize_decision,
            WebSearchTool()
        ],
        model=llm_model,
        prompt_templates=prompt_templates,
        add_base_tools=False
    )

    try:
        result = agent.run(f"Process this claim JSON strictly according to the mandatory workflow: {claim_json}")

        decision = None
        if last_decision_json:
            try:
                decision = json.loads(last_decision_json)
            except Exception:
                decision = None

        if decision and isinstance(decision, dict) and "claim_number" in decision:
            dashboard_html = build_dashboard_html(decision)

            markdown_summary = ""
            return dashboard_html, markdown_summary

        return (
            "<div style='color:#f59e0b;font-weight:700;'>Result returned, but dashboard data could not be parsed.</div>",
            str(result)
        )

    except Exception as e:
        return (
            "<div style='color:#ef4444;font-weight:700;'>Agent execution failed.</div>",
            f"### ❌ Agent Execution Error\n{str(e)}"
        )

# ==========================================
# 6. GRADIO UI (Guided & Anti-Breakage)
# ==========================================
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    .claim-dashboard {max-width: 820px; margin: 0 auto; font-family: Arial, sans-serif;}
    .claim-id {font-size: 12px; opacity: 0.75; margin-bottom: 8px;}
    .status-banner {display: flex; align-items: center; gap: 12px; padding: 16px 18px; border-radius: 16px; margin-bottom: 14px; border: 1px solid rgba(255,255,255,0.08);}
    .status-banner.approved {background: rgba(34,197,94,0.12);}
    .status-banner.review {background: rgba(245,158,11,0.12);}
    .status-banner.denied {background: rgba(239,68,68,0.12);}
    .status-icon {font-size: 28px; line-height: 1;}
    .status-title {font-size: 24px; font-weight: 700;}
    .metric-grid {display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px;}
    .metric-card, .summary-card, .confidence-card {
        border-radius: 16px;
        padding: 18px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .metric-value {
        font-size: 30px;
        font-weight: 700;
        line-height: 1.1;
        text-align: center;
        width: 100%;
    }
    .metric-label {
        margin-top: 8px;
        font-size: 13px;
        opacity: 0.8;
        text-align: center;
        width: 100%;
    }
    .section-title {font-size: 15px; font-weight: 700; text-align: center; margin-bottom: 10px;}
    .summary-text, .confidence-text {font-size: 14px; line-height: 1.5; opacity: 0.95;}
    .confidence-card {
        display: grid;
        grid-template-columns: 180px 1fr;
        gap: 18px;
        align-items: center;
    }
    .confidence-left {display: flex; flex-direction: column; align-items: center; gap: 10px;}
    .confidence-score {font-size: 24px; font-weight: 700;}
    .progress-ring {
        --size: 110px;
        width: var(--size);
        height: var(--size);
        border-radius: 50%;
        display: grid;
        place-items: center;
        background:
            radial-gradient(closest-side, #0b1020 74%, transparent 75% 100%),
            conic-gradient(currentColor calc(var(--p) * 1%), rgba(255,255,255,0.08) 0);
    }
    .progress-ring.approved {color: #22c55e;}
    .progress-ring.review {color: #f59e0b;}
    .progress-ring.denied {color: #ef4444;}
    .progress-inner {font-size: 18px; font-weight: 700;}
    @media (max-width: 700px) {
        .metric-grid {grid-template-columns: 1fr;}
        .confidence-card {grid-template-columns: 1fr;}
    }
    """
) as demo:
    gr.Markdown("# 🚗 Agentic Auto-Insurance Claims Processor")
    gr.Markdown("*The Policy Knowledge Base is initialized and ready. Submit your claim below.*")
    
    with gr.Tab("Settings"):
        api_key_input = gr.Textbox(
            label="OpenAI API Key", 
            type="password", 
            placeholder="sk-...",
            info="Your key is only used for this session."
        )
        base_url_input = gr.Textbox(label="API Base URL (Optional)", value="https://api.openai.com/v1")
        
    with gr.Tab("Claim Adjudicator"):
        with gr.Row():
            # Guided User Inputs
            with gr.Column():
                gr.Markdown("### 📝 Claim Details")
                claim_no = gr.Textbox(label="Claim Number", placeholder="CLAIM-100", info="Format: CLAIM-XXX")
                policy_no = gr.Dropdown(
                    label="Policy Number", 
                    choices=["PN-1", "PN-2", "PN-3", "PN-4", "PN-5", "PN-6", "PN-7", "PN-8", "PN-9", "PN-10"], 
                    info="Select an active policy ID."
                )
                claimant_name = gr.Textbox(label="Claimant Name", placeholder="Jane Doe")
                loss_date = gr.Textbox(label="Date of Loss", placeholder="YYYY-MM-DD", info="Must follow YYYY-MM-DD format.")
                
                loss_desc = gr.Textbox(
                    label="Loss Description", 
                    placeholder="Describe the incident...",
                    lines=2
                )
                repair_cost = gr.Number(
                    label="Estimated Repair Cost ($)", 
                    value=500.0, 
                    minimum=0, 
                    info="Do not use negative values."
                )
                vehicle_info = gr.Textbox(label="Vehicle Details", placeholder="2022 Tesla Model 3")
                
                submit_btn = gr.Button("Evaluate Claim", variant="primary")
            
            # Decision Output
            
            with gr.Column():
                gr.Markdown("### ⚖️ Agent Decision")
                dashboard_display = gr.HTML(value="")
                output_display = gr.Markdown(value="*Detailed results will appear here after evaluation...*")

        # Preset Examples Component
        gr.Examples(
            examples=[
                ["CLAIM-001", "PN-1", "John Smith", "2023-10-15", 850.0, "2020 Honda Civic", "Front bumper damage from low-speed collision."],
                ["CLAIM-002", "PN-3", "Alice Wong", "2024-02-10", 12000.0, "2023 Ford F-150", "Extensive side impact damage from running a red light."],
            ],
            inputs=[claim_no, policy_no, claimant_name, loss_date, repair_cost, vehicle_info, loss_desc],
            label="Load Example Claims"
        )

    submit_btn.click(
        fn=ui_process_claim,
        inputs=[api_key_input, base_url_input, claim_no, policy_no, claimant_name, loss_date, repair_cost, vehicle_info, loss_desc],
        outputs=[dashboard_display, output_display]
    )

if __name__ == "__main__":
    demo.launch()