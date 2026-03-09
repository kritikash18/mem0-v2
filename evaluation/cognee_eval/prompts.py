"""
Prompts for Cognee Evaluation.

Contains prompts for:
- Answer generation from Cognee search results
- LLM judge evaluation
"""


# =============================================================================
# ANSWER GENERATION PROMPT
# =============================================================================

ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with answering questions based on information retrieved from a knowledge graph memory system.

# CONTEXT:
You have access to search results from a knowledge graph that was built from audio content. The results may include entities, relationships, summaries, or raw text chunks. Your goal is to provide accurate, concise answers based solely on these results.

# INSTRUCTIONS:
1. Carefully analyze all provided search results to find information relevant to the question
2. Use logical inference and reasoning when the answer is implied but not explicitly stated
3. Look for semantic equivalence between the question and result content
4. For numerical or temporal questions, extract exact values
5. Prioritize explicit information over inference, but use inference when necessary
6. Do NOT introduce external knowledge - rely exclusively on the provided results
7. If the answer cannot be determined from the results, respond with "I don't know"

# SEARCH RESULTS:
{{results}}

# QUESTION:
{{question}}

# ANSWER:
"""


# =============================================================================
# LLM JUDGE PROMPT (same as audio_eval for fair comparison)
# =============================================================================

LLM_JUDGE_PROMPT = """
Evaluate whether the generated answer is correct compared to the gold answer.

QUESTION: {question}
GOLD ANSWER: {gold_answer}
GENERATED ANSWER: {generated_answer}

EVALUATION CRITERIA:
- Be generous: if the generated answer conveys the same meaning, mark CORRECT
- Accept paraphrases, synonyms, and different formats (e.g., "500" vs "five hundred")
- Accept partial matches if they capture the key information
- Mark WRONG only if the answer is factually incorrect or completely misses the point

Return your evaluation as JSON: {{"label": "CORRECT"}} or {{"label": "WRONG"}}
"""
