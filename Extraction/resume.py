import os
import json
import re
import fitz   
from docx import Document
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

class LLMResumeParser:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", hf_token=None):
        hf_token = hf_token or os.getenv("HUGGINGFACE_API_KEY")
        if not hf_token:
            raise ValueError("Hugging Face API token must be provided.")

        self.model = model_name
        self.client = InferenceClient(model=self.model, token=hf_token)
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        return '''
You are an expert resume parser.

Your task is to extract structured data from unstructured resume text into exactly five predefined flat sections. You must return only one well-formed JSON object using the exact keys and definitions below. Do not add, omit, rename, or modify any section.

1. "skill"
Include only hard technical skills such as programming languages, frameworks, libraries, software tools, platforms, cloud services, or technical methodologies.
Do NOT include certifications, soft skills, role descriptions, degrees, or company names.
Examples: Python, TensorFlow, SQL, AWS, Docker, CI/CD, Agile, Power BI

2. "education"
Include only formal academic qualifications such as:
- Schooling (10th, 12th)
- Undergraduate (B.Tech, BSc)
- Postgraduate (M.Tech, MBA, MSc)
- Doctorate (PhD)

Each item must contain only the degree/program and institution name. Do NOT include years, CGPA, platforms like Coursera.

Examples:
- B.Tech in Computer Science from IIT Bombay
- 10th from Delhi Public School
- MBA from IIM Ahmedabad

3. "experience"
Each experience must be written as a single plain text string. Combine job title, company, location, and responsibilities into one sentence.

Example:
"Data Scientist at ABM, Nov22–Present — Developed SQL AI Agent using LangChain and Streamlit for natural language to SQL query conversion."

Do not use structured fields or key-value format.

4. "job role"
Include one specific job title that best represents the candidate’s most recent or primary designation.
Example:
- Data Scientist

5. "other information"
Include everything else:
- Certifications
- Soft skills
- Languages spoken
- Hobbies and interests
- Awards, objectives, extracurriculars

Certifications should go here regardless of provider.

Output Format:
{
  "skill": [],
  "education": [],
  "experience": [],
  "job role": [],
  "other information": []
}

Strict Rules:
- Use plain text list for each key
- Do NOT use nested objects or keys
- Do NOT hallucinate
- Do NOT wrap output in any explanation
'''

    def clean_text(self, text: str) -> str:
        text = text.strip()
        text = text.replace("\n", "\\n").replace("\t", "\\t")
        text = re.sub(r"[^\x00-\x7F]+", "", text)  
        text = re.sub(r"\s+", " ", text)  
        return text.strip()

    def extract_fields(self, resume_text: str) -> dict:
        cleaned_text = self.clean_text(resume_text)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": cleaned_text}
        ]

        try:
            response = self.client.chat_completion(
                messages=messages,
                temperature=0,      
                top_p=1.0,             
                max_tokens=None
                 

            )

            raw_output = response.choices[0].message.content.strip()
            print(f"\n🔍 Raw LLM Output:\n{raw_output}\n")

            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}')
            if json_start == -1 or json_end == -1:
                print(" Could not find valid JSON in response.")
                return {}

            json_text = raw_output[json_start:json_end + 1]
            result = json.loads(json_text)
            expected_keys = ["skill", "education", "experience", "job role", "other information"]
            for key in expected_keys:
                if key not in result or not isinstance(result[key], list):
                    result[key] = []

            return result

        except Exception as e:
            print(f" LLM inference failed: {e}")
            return {}

    def save_to_json(self, data: dict, output_dir: str, original_file: str):
        if not data or all(not section for section in data.values()):
            print(" No data to save — skipping JSON write.")
            return

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(original_file))[0]
        output_path = os.path.join(output_dir, f"{base_name}.json")

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f" Saved: {output_path}")
        except Exception as e:
            print(f" Failed to save JSON for {original_file}: {e}")

    def extract_text_from_file(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                doc = fitz.open(file_path)
                texts = []
                for page in doc:
                    text = page.get_text()
                    if not text.strip():
                        blocks = page.get_text("blocks")
                        text = "\n".join(b[4].strip() for b in sorted(blocks, key=lambda b: (b[1], b[0])) if b[4].strip())
                    texts.append(text.strip())
                return " ".join(texts)
            elif ext == ".docx":
                doc = Document(file_path)
                return " ".join(para.text.strip() for para in doc.paragraphs if para.text.strip())
            else:
                print(f" Unsupported file format: {file_path}")
                return ""
        except Exception as e:
            print(f" Error extracting text: {e}")
            return ""


 
def clear_json_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.json'):
                try:
                    os.remove(os.path.join(folder_path, filename))
                    print(f"Deleted: {filename}")
                except Exception as e:
                    print(f"Cannot delete {filename}: {e}")
    else:
        os.makedirs(folder_path)


 
def process_resumes(input_path: str, output_dir: str, hf_token: str):
    parser = LLMResumeParser(hf_token=hf_token)
    clear_json_folder(output_dir)

    if os.path.isfile(input_path):
        files = [input_path] if input_path.lower().endswith((".pdf", ".docx")) else []
    elif os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                 if f.lower().endswith((".pdf", ".docx"))]
    else:
        print(" Invalid input path.")
        return

    for file_path in files:
        print(f"\nProcessing: {file_path}")
        text = parser.extract_text_from_file(file_path)
        if not text.strip():
            print("Skipping empty or unreadable file.")
            continue
        parsed = parser.extract_fields(text)
        parser.save_to_json(parsed, output_dir, file_path)


 
# if __name__ == "__main__":
#     hf_token = os.getenv("HUGGINGFACE_API_KEY")
 
#     input_path = r"C:\Users\SibaPrasadPatra\Desktop\new_hugging_face\resumes"
#     output_path = r"C:\Users\SibaPrasadPatra\Desktop\new_hugging_face\resume_json"
#     process_resumes(input_path, output_path,hf_token)
