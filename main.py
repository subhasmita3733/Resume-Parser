import os
import time
import traceback
from Extraction.resume import process_resumes as extract_all_resumes
from Extraction.jd import process_jds as extract_all_jds
from Embedding.resume import embed_all_jsons_from_folder as embed_resumes
from Embedding.jd import embed_all_jsons_from_folder as embed_jds
from LLM.compare import main as run_llm_comparison

RESUME_RAW_FOLDER = r"C:\Users\SibaPrasadPatra\Desktop\new_hugging_face\resumes"
JD_RAW_FOLDER = r"C:\Users\SibaPrasadPatra\Desktop\new_hugging_face\JD"

RESUME_JSON_FOLDER = r"C:\Users\SibaPrasadPatra\Desktop\new_hugging_face\resume_json"
JD_JSON_FOLDER = r"C:\Users\SibaPrasadPatra\Desktop\new_hugging_face\jd_json"

CHROMA_DB_RESUME = r"C:\Users\SibaPrasadPatra\Desktop\new_hugging_face\chroma_db_resume"
CHROMA_DB_JD = r"C:\Users\SibaPrasadPatra\Desktop\new_hugging_face\chroma_db_jd"

def timed_step(step_name, func, *args, **kwargs):
    print(f"\n[STEP] Starting: {step_name}")
    start = time.time()
    try:
        func(*args, **kwargs)
        print(f"[DONE] {step_name} completed in {time.time() - start:.2f} sec\n")
    except Exception as e:
        print(f"[ERROR] Failed during '{step_name}': {e}")
        traceback.print_exc()

def main():
    print("\n=== Resume vs JD Matching Pipeline Started ===")
    timed_step("Resume Extraction", extract_all_resumes, RESUME_RAW_FOLDER, RESUME_JSON_FOLDER)
    timed_step("JD Extraction", extract_all_jds, JD_RAW_FOLDER, JD_JSON_FOLDER)
    timed_step("Resume Embedding", embed_resumes, RESUME_JSON_FOLDER, CHROMA_DB_RESUME)
    timed_step("JD Embedding", embed_jds, JD_JSON_FOLDER, CHROMA_DB_JD)
    timed_step("LLM-Based Comparison", run_llm_comparison)
    print("\nPipeline Completed Successfully\n")
if __name__ == "__main__":
    main()
