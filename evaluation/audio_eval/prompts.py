"""
Prompts for Audio Memory Evaluation.

Contains prompts for:
- Answer generation from retrieved memories
- LLM judge evaluation
- Custom reading comprehension memory extraction
"""

from datetime import datetime

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
{{memories}}

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


# =============================================================================
# READING COMPREHENSION MEMORY EXTRACTION PROMPT
# =============================================================================

READING_COMPREHENSION_MEMORY_PROMPT = f"""You are a Reading Comprehension Memory Organizer, specialized in extracting factual information from audio transcriptions of informational content (articles, passages, lectures, documentaries, etc.).

Your primary role is to extract key facts, definitions, entities, events, and relationships from the transcribed content and organize them into distinct, retrievable memory units. This allows for accurate question-answering based on the content.

# [IMPORTANT]: EXTRACT FACTUAL INFORMATION FROM THE CONTENT. DO NOT EXTRACT CONVERSATIONAL METADATA OR REFERENCES TO "THE USER" OR "THE SPEAKER".
# [IMPORTANT]: FOCUS ON THE SUBJECT MATTER BEING DISCUSSED, NOT WHO IS SPEAKING ABOUT IT.

Types of Information to Extract:

1. **Definitions and Explanations**: What things are, what they mean, what they stand for
   - Example: "UMC stands for United Methodist Church", "It is a mainline Protestant Methodist denomination"

2. **Named Entities**: Organizations, people, places, dates, events
   - Example: "Founded in 1968", "John and Charles Wesley in England", "The Great Awakening in the United States"

3. **Relationships and Connections**: How entities relate to each other
   - Example: "UMC traces its roots to the revival movement of John and Charles Wesley"

4. **Attributes and Characteristics**: Properties, features, qualities of entities
   - Example: "The Church's theological orientation is decidedly Wesleyan"
   - Example: "It embraces both liturgical and evangelical elements"

5. **Historical Facts and Events**: When things happened, what occurred
   - Example: "Formed by the Union of the Methodist Church U.S.A. and the Evangelical United Brethren Church"

6. **Processes and Mechanisms**: How things work, what they do
   - Example from a different context: "Photosynthesis converts light energy into chemical energy"

7. **Numerical and Quantitative Information**: Numbers, measurements, statistics
   - Example from a different context: "The population is approximately 500,000"

# Few-Shot Examples:

Input: The United Methodist Church, UMC, is a mainline Protestant Methodist denomination. In the 19th century its main predecessor was a leader in Evangelicalism. Founded in 1968 by the Union of the Methodist Church U.S.A. and the Evangelical United Brethren Church. The UMC traces its roots back to the revival movement of John and Charles Wesley in England as well as the Great Awakening in the United States.

Output: {{
  "facts": [
    "UMC stands for United Methodist Church",
    "The United Methodist Church is a mainline Protestant Methodist denomination",
    "In the 19th century, UMC's main predecessor was a leader in Evangelicalism",
    "UMC was founded in 1968",
    "UMC was formed by the Union of the Methodist Church U.S.A. and the Evangelical United Brethren Church",
    "UMC traces its roots to the revival movement of John and Charles Wesley in England",
    "UMC traces its roots to the Great Awakening in the United States"
  ]
}}

Input: Photosynthesis is a process used by plants to convert light energy into chemical energy. It occurs primarily in the chloroplasts. The process requires sunlight, water, and carbon dioxide. Oxygen is released as a byproduct.

Output: {{
  "facts": [
    "Photosynthesis is a process used by plants to convert light energy into chemical energy",
    "Photosynthesis occurs primarily in the chloroplasts",
    "Photosynthesis requires sunlight, water, and carbon dioxide",
    "Oxygen is released as a byproduct of photosynthesis"
  ]
}}

Input: Mount Everest is the Earth's highest mountain. It is located in the Mahalangur Himal sub-range of the Himalayas. The international border between China and Nepal runs across its summit point. Its elevation of 8,848.86 meters was established by a 2020 survey.

Output: {{
  "facts": [
    "Mount Everest is the Earth's highest mountain",
    "Mount Everest is located in the Mahalangur Himal sub-range of the Himalayas",
    "The international border between China and Nepal runs across Mount Everest's summit point",
    "Mount Everest's elevation is 8,848.86 meters",
    "Mount Everest's elevation was established by a 2020 survey"
  ]
}}

Input: Hello, how are you today?

Output: {{"facts": []}}

Input: This is a beautiful day for a walk in the park.

Output: {{"facts": []}}

Input: user: The United Methodist Church, UMC, is a mainline Protestant Methodist denomination. Founded in 1968.

Output: {{
  "facts": [
    "UMC stands for United Methodist Church",
    "The United Methodist Church is a mainline Protestant Methodist denomination",
    "UMC was founded in 1968"
  ]
}}

# Critical Guidelines:

- **Extract objective facts from the content**, not subjective opinions unless they're attributed facts
- **Break down complex sentences** into atomic, self-contained facts
- **Each fact should be independently understandable** without needing the others
- **Preserve proper nouns, dates, numbers** exactly as stated
- **Do NOT extract**:
  - Conversational fillers ("um", "you know", "basically")
  - References to the speaker or listener unless they're the subject
  - Meta-commentary about the content itself
- **Extract content regardless of source**: Whether from user message, audio transcription, or text
- **Ignore role prefixes**: Input may have "user:", "assistant:", etc. - these are formatting artifacts, not part of the content
- **Return empty list** if the input contains no factual information
- **Maintain the language** of the original content
- Today's date is {datetime.now().strftime("%Y-%m-%d")}
- **Return JSON format**: {{"facts": ["fact1", "fact2", ...]}}

# IMPORTANT: Handling Role-Prefixed Input

If the input has role prefixes like "user: " or "assistant: ", ignore them completely. They are system artifacts, not part of the actual content to extract from.

Example with role prefix:
- Input: "user: Mount Everest is 8,848 meters high"
- Extract: "Mount Everest is 8,848 meters high" (ignore "user: ")

# IMPORTANT: Consolidation for Long Content

If the content is lengthy and produces more than 20 facts:
1. **First extract all atomic facts** (don't self-censor)
2. **Then consolidate** by:
   - Merging highly related facts that cover the same information
   - Combining facts about the same entity when appropriate (e.g., "X was founded in 1968" + "X is located in USA" → "X was founded in 1968 and is located in USA")
   - Prioritizing unique information over redundant details
   - Keeping the most important/central facts
   - **Target maximum: 20 facts** (though fewer is fine if content is brief)
3. **Ensure no critical information is lost** - if merging, make facts slightly longer rather than dropping information
4. Each consolidated fact can be 2-3 sentences if needed to preserve information

Example of consolidation:
Before (5 facts):
- "The UMC was founded in 1968"
- "The UMC is located in the United States"
- "The UMC has 12 million members"
- "The UMC is a Protestant denomination"
- "The UMC follows Wesleyan theology"

After consolidation (2-3 facts):
- "The UMC is a Protestant denomination founded in 1968 in the United States"
- "The UMC has 12 million members and follows Wesleyan theology"

Following is the transcribed content or text. Extract all relevant factual information, consolidate if needed to stay under 20 facts, and return them in the JSON format shown above.
"""


# =============================================================================
# READING COMPREHENSION UPDATE MEMORY PROMPT
# =============================================================================

READING_COMPREHENSION_UPDATE_MEMORY_PROMPT = """You are a Reading Comprehension Memory Manager specialized in maintaining factual information extracted from informational content.

Your task is to manage a knowledge base of facts by deciding whether to ADD, UPDATE, DELETE, or keep (NONE) facts.

# CRITICAL: Reading Comprehension Principles

1. **Preserve Atomic Facts**: Keep definitional facts separate (e.g., "X stands for Y" should stay separate from "X is a type of Z")
2. **No Aggressive Merging**: Only merge facts if they are truly redundant, not just related
3. **Preserve Exact Information**: Definitions, numbers, dates, and relationships must stay verbatim
4. **Prioritize Searchability**: Keep facts that directly answer "what", "when", "where", "who" questions

# Operations

**ADD**: New information not present in existing memory
- Use ADD for ALL new facts from the current passage
- Example: Memory has "UMC is Protestant", new fact "UMC was founded in 1968" → ADD

**UPDATE**: Same topic but more complete/accurate information **in an existing memory**
- ONLY use UPDATE when modifying a memory that already exists in the Old Memory list
- Example: Old Memory has "Founded in 1960s", new fact "Founded in 1968" → UPDATE (more specific)
- Example: Old Memory has "Has members", new fact "Has 12 million members" → UPDATE (more complete)
- NEVER use UPDATE for a fact that's being added in the current batch — those are always ADD

**DELETE**: Contradictory or incorrect information **in an existing memory**
- ONLY use DELETE when removing a memory that already exists in the Old Memory list
- Example: Old Memory has "Founded in 1960", new fact "Founded in 1968" → DELETE old, ADD new

**NONE**: Fact already present in Old Memory (even if worded differently)
- Example: Old Memory has "UMC stands for United Methodist Church", new fact "UMC means United Methodist Church" → NONE

# Important Guidelines for Reading Comprehension

- **DO NOT merge** definitional facts with descriptive facts
  - Bad: "United Methodist Church (UMC) is a Protestant denomination founded in 1968"
  - Good: Keep separate: "UMC stands for United Methodist Church", "UMC is a Protestant denomination", "UMC was founded in 1968"

- **DO NOT rephrase** facts unless updating
  - Keep "UMC stands for X" instead of rephrasing to "The United Methodist Church (UMC)..."

- **DO merge** only truly redundant facts
  - "UMC founded in 1968" + "UMC was established in 1968" → merge (same info)
  - "UMC founded in 1968" + "UMC is Protestant" → DO NOT merge (different info)

# Output Format

Return JSON with "memory" key containing list of facts with their status:

```json
{
  "memory": [
    {
      "id": "0",
      "text": "fact text here",
      "event": "ADD|UPDATE|DELETE|NONE",
      "old_memory": "..." (only for UPDATE/DELETE)
    }
  ]
}
```

# Examples

**Example 1: Multiple New Facts from Current Passage — ALL are ADD**
Old Memory:
```json
[
  {"id": "0", "text": "Marie Curie was a physicist"}
]
```
Retrieved Facts: ["Marie Curie won the Nobel Prize in 1903", "She conducted research on radioactivity", "She was the first woman to win a Nobel Prize"]

New Memory:
```json
{
  "memory": [
    {"id": "0", "text": "Marie Curie was a physicist", "event": "NONE"},
    {"id": "1", "text": "Marie Curie won the Nobel Prize in 1903", "event": "ADD"},
    {"id": "2", "text": "She conducted research on radioactivity", "event": "ADD"},
    {"id": "3", "text": "She was the first woman to win a Nobel Prize", "event": "ADD"}
  ]
}
```
✓ All new facts from the current passage are ADD, even though they relate to the same entity

**Example 2: Preserve Atomic Facts, Update Existing**
Old Memory:
```json
[
  {"id": "0", "text": "UMC is a Protestant denomination"}
]
```
Retrieved Facts: ["UMC stands for United Methodist Church", "UMC is a mainline Protestant Methodist denomination"]

New Memory:
```json
{
  "memory": [
    {"id": "0", "text": "UMC is a mainline Protestant Methodist denomination", "event": "UPDATE", "old_memory": "UMC is a Protestant denomination"},
    {"id": "1", "text": "UMC stands for United Methodist Church", "event": "ADD"}
  ]
}
```
✓ Updated existing memory to more specific version, added new definition as separate fact

**Example 3: No Unnecessary Merging**
Old Memory:
```json
[
  {"id": "0", "text": "UMC stands for United Methodist Church"},
  {"id": "1", "text": "UMC was founded in 1968"}
]
```
Retrieved Facts: ["UMC is a Protestant denomination"]

New Memory:
```json
{
  "memory": [
    {"id": "0", "text": "UMC stands for United Methodist Church", "event": "NONE"},
    {"id": "1", "text": "UMC was founded in 1968", "event": "NONE"},
    {"id": "2", "text": "UMC is a Protestant denomination", "event": "ADD"}
  ]
}
```
✓ Added new fact without merging with existing ones

Now process the facts:
"""
