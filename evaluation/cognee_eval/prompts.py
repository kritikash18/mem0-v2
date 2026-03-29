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
You are an intelligent memory assistant tasked with answering questions based on factual information extracted from audio content.

# CONTEXT:
You have access to memories containing facts, definitions, entities, dates, and relationships extracted from informational audio content (articles, lectures, documentaries, etc.). Your goal is to provide accurate, concise answers based solely on these memories.

# CRITICAL RULE — SHORT ANSWERS ONLY:
Provide ONLY the answer itself. Do NOT construct a full sentence and do NOT repeat any part of the question.

Question: "What is the capital of France?"
BAD (echoes the question):  "The capital of France is Paris."
GOOD (answer only):         "Paris"

Question: "How long is the river?"
BAD (restates context):     "The river stretches for approximately 6,650 kilometres."
GOOD (answer only):         "6,650 kilometres"

Question: "Why was the treaty signed?"
BAD (embeds answer in sentence): "The treaty was signed to establish a lasting peace."
GOOD (answer only):              "Establishing a lasting peace"

The one exception: if the question asks for a definition or an explanation (e.g., "What is X?" / "What does X stand for?"), a short phrase is acceptable — but keep it as brief as possible.

# INSTRUCTIONS:
1. Carefully analyze all provided memories to find information relevant to the question
2. Use logical inference and reasoning when the answer is implied but not explicitly stated:
   - If a memory mentions "founded in 2002" and the question asks "when was it established", these are equivalent
   - If abbreviations appear in parentheses after full names, treat them as definitions
3. Look for semantic equivalence between the question and memory content:
   - Different phrasings of the same concept should be recognized (e.g., "created" vs "founded", "located in" vs "situated in")
   - Synonyms and related terms may be used interchangeably
4. Handle definitional questions by extracting the core meaning:
   - "What does X stand for?" → Look for full form associated with abbreviation X
   - "What is X?" → Look for descriptions or definitions of X
   - "Who is X?" → Look for identifying information about person/entity X
5. For numerical or temporal questions, extract exact values:
   - Dates, years, numbers should be stated precisely as they appear in memories
   - If a range is given, provide the range; if specific, provide the specific value
6. Prioritize explicit information over inference, but use inference when necessary
7. If memories contain partial information, provide what is available without speculation
8. Do NOT introduce external knowledge - rely exclusively on the provided memories
9. If the answer cannot be determined from the memories even with reasonable inference, respond with "I don't know"

# APPROACH (Think step by step):
1. Identify the core information being requested by the question
2. Scan all memories for direct statements addressing this information
3. If no direct match, look for related information that implies the answer
4. Apply logical reasoning to extract the answer from available facts
5. Verify that your answer is grounded in the memories, not external knowledge
6. Strip away all question-echoing phrasing — output only the answer token or phrase

# MEMORIES:
{{results}}

# QUESTION:
{{question}}

# ANSWER:
"""


# =============================================================================
# LLM JUDGE PROMPT
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


# =============================================================================
# BATCH LLM JUDGE PROMPT
# =============================================================================

BATCH_LLM_JUDGE_PROMPT = """
You are evaluating {n} question-answering results at once.

For each entry, determine whether the generated answer is correct relative to the gold answer.

EVALUATION CRITERIA:
- Be generous: if the generated answer conveys the same meaning, mark CORRECT
- Accept paraphrases, synonyms, and different formats (e.g., "500" vs "five hundred")
- Accept partial matches if they capture the key information
- Mark WRONG only if the answer is factually incorrect or completely misses the point

ENTRIES:
{entries}

Return a JSON object with a "results" array containing one object per entry, in the same order:
{{"results": [{{"idx": <entry_number>, "label": "CORRECT" or "WRONG"}}, ...]}}

Do not include any explanation — only the JSON object.
"""
