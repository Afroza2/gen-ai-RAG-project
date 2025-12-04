import re
from langchain_core.prompts import PromptTemplate

def ask_rag(retriever, question: str, qa_prompt, llm, max_context_chars=1500):
    """
    Retrieves documents and generates an answer using the LLM.
    Truncates context to avoid overflow.
    """
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    context = context[:max_context_chars]

    prompt_text = qa_prompt.format(context=context, question=question)
    resp = llm.invoke(prompt_text)
    return resp, docs

def score_question(scores_list, index, vf, vr, rf, rr):
    """
    Updates the scores list at the given index (1-based) with the provided values.
    """
    # Adjust index to 0-based
    idx = index - 1
    if 0 <= idx < len(scores_list):
        scores_list[idx]["vanilla_faith"] = vf
        scores_list[idx]["vanilla_relev"] = vr
        scores_list[idx]["rag_faith"] = rf
        scores_list[idx]["rag_relev"] = rr
    else:
        print(f"Error: Index {index} out of range.")

def compute_metrics(scores_list):
    """
    Computes average faithfulness and relevancy for both Vanilla and RAG systems.
    """
    metrics = {
        "vanilla_faith_avg": 0.0,
        "vanilla_relev_avg": 0.0,
        "rag_faith_avg": 0.0,
        "rag_relev_avg": 0.0,
        "count": 0
    }
    
    valid_entries = 0
    for s in scores_list:
        # Check if all scores are present (not None)
        if all(k in s and s[k] is not None for k in ["vanilla_faith", "vanilla_relev", "rag_faith", "rag_relev"]):
            metrics["vanilla_faith_avg"] += s["vanilla_faith"]
            metrics["vanilla_relev_avg"] += s["vanilla_relev"]
            metrics["rag_faith_avg"] += s["rag_faith"]
            metrics["rag_relev_avg"] += s["rag_relev"]
            valid_entries += 1
            
    if valid_entries > 0:
        metrics["vanilla_faith_avg"] /= valid_entries
        metrics["vanilla_relev_avg"] /= valid_entries
        metrics["rag_faith_avg"] /= valid_entries
        metrics["rag_relev_avg"] /= valid_entries
    
    metrics["count"] = valid_entries
    return metrics

def parse_score(response_text):
    """
    Extracts a number from the LLM response and normalizes it to 0.0-1.0.
    Assumes the LLM returns a score out of 10.
    """
    try:
        # Find the first number in the text (integer or float)
        match = re.search(r'\d+(\.\d+)?', response_text)
        if match:
            val = float(match.group())
            # Clamp to 0-10
            val = max(0, min(10, val))
            return val / 10.0
        return 0.0
    except:
        return 0.0

def evaluate_answer_with_llm(llm, question, answer, context=None):
    """
    Uses the LLM to evaluate the answer for Faithfulness and Relevancy.
    Returns a tuple (faithfulness_score, relevancy_score).
    """
    
    # Relevancy Prompt
    relevancy_template = """You are an impartial judge. Evaluate the following answer based on the question provided.
Question: {question}
Answer: {answer}

Rate the Relevancy of the answer to the question on a scale from 0 to 10.
0 means the answer is completely irrelevant or does not address the question at all.
10 means the answer fully and perfectly addresses all parts of the question.

Return ONLY the number (0-10)."""
    
    # Faithfulness Prompt (only if context is provided)
    faithfulness_template = """You are an impartial judge. Evaluate the following answer based on the context provided.
Context: {context}
Answer: {answer}

Rate the Faithfulness of the answer to the context on a scale from 0 to 10.
0 means the answer contains hallucinations or information not supported by the context.
10 means the answer is fully supported by the context.

Return ONLY the number (0-10)."""
    
    try:
        # Calculate Relevancy
        # We use a simple PromptTemplate for formatting
        rel_prompt = relevancy_template.format(question=question, answer=answer)
        rel_resp = llm.invoke(rel_prompt)
        rel_score = parse_score(rel_resp)
        
        # Calculate Faithfulness
        if context:
            faith_prompt = faithfulness_template.format(context=context, answer=answer)
            faith_resp = llm.invoke(faith_prompt)
            faith_score = parse_score(faith_resp)
        else:
            # For Vanilla LLM without context, faithfulness to "source documents" is 0.0 or N/A.
            # We return 0.0 as a safe default for "ungrounded".
            faith_score = 0.0 

        return faith_score, rel_score
        
    except Exception as e:
        print(f"Error during LLM evaluation: {e}")
        return 0.0, 0.0
