import os
import json
import argparse
import re
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from chromadb import PersistentClient

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY") 
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN)

jd_db_path = r"C:\Users\SibaPrasadPatra\Desktop\new_hugging_face\chroma_db_jd"
resume_db_path = r"C:\Users\SibaPrasadPatra\Desktop\new_hugging_face\chroma_db_resume"

FIELD_ORDER = ["Skills", "Education", "Experience", "Job Role"]
system_prompt = """ You are a world-class HR, Talent Acquisition, and Generative AI Specialist with deep expertise in job-role alignment, semantic document comparison, and hiring decision automation.

You are tasked with comparing a candidate resume and a job description. Both are pre-parsed into structured fields: Skills, Education, Job Role, Experience, and Other Information. Your job is to assess the alignment strictly based on meaning — not exact keyword matches.

You must return a single valid JSON object in the structure described below.

Instructions:
- Evaluate semantic relevance, not keyword overlap. For example, treat "ML Engineer" and "Machine Learning Engineer" as equivalent.
- Apply real-world hiring logic: If the resume exceeds JD requirements (e.g., more skills, higher education, deeper experience), assign a high match_pct — even 100%.
- Do not penalize minor differences in naming or formatting.
- Never assign 0% if a field contains any valid data. Only assign 0% if the resume field is empty or clearly unrelated.
- Never hallucinate or infer information not present in either document.
- Never nest objects — keep JSON flat.
- Escape invalid characters like \\t, \\n, and quotes properly.
- Use consistent, professional phrasing in all explanations.

Field Matching Logic:
Skills
- Match based on technical equivalence, not wording.
- If the resume includes all required skills (or more), assign 100%.
- If semantically similar (e.g., "pandas" vs. "Python data manipulation"), assign 80–95%.
- Penalize only if critical skills are missing.
- If no overlap at all, assign <30% with explanation.

Education
- Full match (100%) if the degree level and field align with the JD.
- Slight name variations are acceptable (B.E. ≈ B.Tech).
- If field is unrelated (e.g., BA in History for Data Scientist JD), assign 20–30%.
- If degree level is below requirement (e.g., diploma instead of B.Tech), assign <30%.
- If degree level is higher than  requirement (e.g., M.Tech or Phd  instead of B.Tech), assign greater than 90%.
- Clearly explain penalties.

Experience
- Match on role relevance, years of experience, technologies used, domain familiarity.
- Resume that meets or exceeds JD’s experience should score 90–100%.
- Penalize only if domain is different, role is mismatched, or years are far below JD.

Job Role:

Normalize semantically similar roles: "ML Engineer" ≈ "Machine Learning Engineer", "Data Scientist" ≈ "ML Engineer", "AI Researcher" ≈ "Data Scientist" (etc.).

Use semantic similarity to determine role alignment, considering both job titles and core job responsibilities (such as tasks listed in job descriptions).

If the job titles and/or responsibilities strongly align, assign a score of 90-100%.

For roles that are somewhat related but distinct (e.g., "Product Manager" vs. "ML Engineer"), assign a score between 50-89%.

Assign 0% only if the job role field is empty or the roles are completely unrelated, with no semantic overlap.

When in doubt, favor higher similarity for roles with overlapping tasks in data science, machine learning, and AI fields (e.g., "Data Scientist" should be scored similarly to "ML Engineer" if they perform overlapping tasks).

Consider synonyms or industry-specific terminology that might not match exact titles but indicate similar core responsibilities (e.g., "Data Engineer" vs. "Machine Learning Engineer").

OverallMatchPercentage
- Weighted average of: Skills (30%), Experience (30%), Education (20%), Job Role (20%)
- Add/subtract ±5% for “Other Information” if highly relevant or problematic.
- Clearly explain rationale for final score.

AI_Generated_Estimate_Percentage
- High score (80–100%) if language is overly perfect, repetitive, generic.
- Low (0–30%) if nuanced, varied, clearly human-written.

Output Format (Strict JSON Only):
{
  "{resume_filename}": {
    "Skills": {
      "match_pct": float,
      "resume_value": string,
      "job_description_value": string,
      "explanation": string
    },
    "Education": {
      "match_pct": float,
      "resume_value": string,
      "job_description_value": string,
      "explanation": string
    },
    "Job Role": {
      "match_pct": float,
      "resume_value": string,
      "job_description_value": string,
      "explanation": string
    },
    "Experience": {
      "match_pct": float,
      "resume_value": string,
      "job_description_value": string,
      "explanation": string
    },
    "OverallMatchPercentage": float,
    "why_overall_match_is_this": string,
    "AI_Generated_Estimate_Percentage": float
  }
}

- Output only the JSON, no markdown, no extra commentary.
- Do not change key order or structure.."""

user_prompt_template = """You are tasked with comparing a candidate’s resume and a job description, each parsed into structured fields: Skills, Education, Job Role, Experience, and Other Information.

Your goal is to evaluate the semantic alignment between the resume and the job description — focusing on meaning and capability, not exact keyword matches.

Return only a single valid JSON object in this exact structure (replace {resume_filename} with the actual resume file name):

json
Copy
Edit
{{
  "{resume_filename}": {{
    "Skills": {{
      "match_pct": float,
      "resume_value": string,
      "job_description_value": string,
      "explanation": string
    }},
    "Education": {{
      "match_pct": float,
      "resume_value": string,
      "job_description_value": string,
      "explanation": string
    }},
    "Job Role": {{
      "match_pct": float,
      "resume_value": string,
      "job_description_value": string,
      "explanation": string
    }},
    "Experience": {{
      "match_pct": float,
      "resume_value": string,
      "job_description_value": string,
      "explanation": string
    }},
    "OverallMatchPercentage": float,
    "why_overall_match_is_this": string,
    "AI_Generated_Estimate_Percentage": float
  }}
}}
Evaluation Instructions:
Semantic similarity means matching based on meaning, not exact text. For example, treat "ML Engineer" and "Machine Learning Engineer" as equivalent that mean both are same .

Use realistic hiring logic: if the resume exceeds JD requirements (more skills, higher education, deeper experience), assign high or full match_pct.

Assign match_pct scores from 0 to 100 representing the degree of semantic alignment.
The overall percentage calculated based on Skill,Education,Experience,Job Role majorly and minorly from other Information
Avoid penalizing minor wording or formatting differences.

Penalize missing key required fields appropriately.

No semicolons (;) in values — use periods or commas

For missing or partial data in any field, explain clearly how it impacts match_pct.

Do not hallucinate or infer data not explicitly present in either document.

Never nest JSON objects inside any field values; keep all fields flat.

Escape special characters (tabs, newlines, quotes) using \\t, \\n, \" respectively.

Maintain consistent phrasing and tone in explanations, as if writing to a hiring manager.

Do not output any text or commentary outside the JSON object.
Output:
Return only the JSON object following the structure above. Do not add any extra text or commentary.
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_filename', type=str, default="resume.docx")
    return parser.parse_args()

def get_all_collections(client):
    try:
        return [c.name for c in client.list_collections()]
    except Exception as e:
        print(f"[ERROR] Failed to list collections: {e}")
        return []

def get_collection_docs(client, collection_name):
    try:
        collection = client.get_collection(collection_name)
        results = collection.get(include=["documents"])
        docs = results.get("documents", [])
        if docs and isinstance(docs[0], list):
            docs = [d for sublist in docs for d in sublist]
        return docs
    except Exception as e:
        print(f"[ERROR] Failed to load collection '{collection_name}': {e}")
        return []

def build_field_texts(field_names, docs):
    lines = []
    for name, doc in zip(field_names, docs):
        doc_clean = re.sub(rf"^{re.escape(name)}:\s*", "", doc, flags=re.IGNORECASE)
        lines.append(f"{name}: {doc_clean}")
    return "\n".join(lines)

def clean_llm_json(raw_response):
    raw = raw_response.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    brace_stack = []
    start_idx = None
    for i, char in enumerate(raw):
        if char == '{':
            if start_idx is None:
                start_idx = i
            brace_stack.append('{')
        elif char == '}':
            if brace_stack:
                brace_stack.pop()
                if not brace_stack:
                    return raw[start_idx:i+1].strip()
    return raw

def normalize_llm_response(data):
    for field in ["Skills", "Education", "Job Role", "Experience"]:
        if field in data:
            for key in ["resume_value", "job_description_value"]:
                value = data[field].get(key)
                if isinstance(value, list):
                    data[field][key] = ", ".join(map(str, value))
    return data

def query_llm(system_prompt, user_prompt, retries=2):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    for attempt in range(retries):
        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=2048,
                temperature=0.0,
                top_p=1.0,
                stop=["```"],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] LLM call failed (attempt {attempt+1}): {e}")
            time.sleep(1)
    return ""

def main():
    start_time = time.time()
    args = get_args()

    jd_client = PersistentClient(path=jd_db_path)
    resume_client = PersistentClient(path=resume_db_path)

    jd_collections = get_all_collections(jd_client)
    resume_collections = get_all_collections(resume_client)

    all_results = []

    for jd_collection in jd_collections:
        jd_docs = get_collection_docs(jd_client, jd_collection)
        if len(jd_docs) < 5:
            continue

        jd_text = build_field_texts(FIELD_ORDER, jd_docs[:4])
        jd_other_info = jd_docs[4]

        for resume_collection in resume_collections:
            resume_docs = get_collection_docs(resume_client, resume_collection)
            if len(resume_docs) < 5:
                continue

            resume_text = build_field_texts(FIELD_ORDER, resume_docs[:4])
            resume_other_info = resume_docs[4]

            comparison_name = f"{resume_collection}_vs_{jd_collection}"
            user_prompt = user_prompt_template.format(resume_filename=comparison_name)
            full_prompt = (
                f"{user_prompt}\n\n"
                f"Job Description:\n{jd_text}\n\n"
                f"Resume:\n{resume_text}\n\n"
                f"Other Information:\n"
                f"JD Other Info: {jd_other_info}\n"
                f"Resume Other Info: {resume_other_info}"
            )

            raw_response = query_llm(system_prompt, full_prompt)
            cleaned_json_str = clean_llm_json(raw_response)
            if not cleaned_json_str:
                continue

            try:
                parsed = json.loads(cleaned_json_str)
                normalized = normalize_llm_response(parsed)
                all_results.append(normalized)
            except json.JSONDecodeError:
                continue

     
    print(json.dumps(all_results, indent=2))


    print(f"[INFO] Total time: {time.time() - start_time:.2f} sec")

# if __name__ == "__main__":
#     main()
