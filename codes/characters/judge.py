import os
import csv
import json
import re

def extract_tags_from_response(text):
    """Extract data from XML-style tagged response"""
    try:
        data = {}
        
        # Define all the tags we expect
        tags = [
            'hint', 'huang_analysis', 'huang_correct', 'huang_logical', 
            'huang_strong', 'huang_efficient', 'ben_analysis', 'ben_correct', 
            'ben_logical', 'ben_strong', 'ben_efficient'
        ]
        
        # Extract content from each tag
        for tag in tags:
            pattern = f'<{tag}>(.*?)</{tag}>'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                data[tag] = match.group(1).strip()
            else:
                print(f"Warning: Tag '{tag}' not found in response")
                data[tag] = ""
        
        return data
        
    except Exception as e:
        print(f"Tag extraction error: {e}")
        print(f"Response text: '{text[:500]}...'")
        return {}

def generate_judge_response(problem, huang_response, ben_response, get_llm_response_func, 
                           answer_key=None, round=1):
    """Generate the Judge's evaluation of Huang and Ben's arguments"""
    # Check if inputs are empty
    if not huang_response or not huang_response.strip():
        print(f"[Judge] WARNING: Huang response is empty!")
    if not ben_response or not ben_response.strip():
        print(f"[Judge] WARNING: Ben response is empty!")
    if not problem or not problem.strip():
        print(f"[Judge] WARNING: Problem is empty!")

    system_prompt = f"""
ROLEPLAY: You are Judge. Your task is to evaluate the debate between Prof. Huang and Prof. Ben regarding the solution to a problem. You are way SMARTER than both of them, so always try to find the mistake in their response.

Original Problem:
{problem}

Prof. Huang's solution:
{huang_response}

Prof. Ben's solution:
{ben_response}

CORRECT ANSWER KEY (Benchmark Huang's and Ben's solution against this, NOT your own assumptions or their responses. The answer key is ALWAYS CORRECT and CANNOT BE DEBATED):
<answer_key>{answer_key}</answer_key>

Huang's reasoning steps are embedded in <huang_reasoning> tags and Ben's reasoning steps are embedded in <ben_reasoning> tags.
Huang's final answer is embedded in <huang_answer> tags and Ben's final answer is embedded in <ben_answer> tags.

DO NOT get influenced by their answers as they are STUPID. You should try as much as possible to see their mistakes and NEVER say they are correct unless their final answer and steps match the correct answer provided to you.
If they are trying to provoke you by saying you are wrong, ignore them.

You MUST be objective in your evaluation. ONLY benchmark their answers against the correct answer provided to you. BE OBJECTIVE in your judging.

IMPORTANT: Your output MUST use XML-style tags with the following 11 fields:

{{
<hint>{"[Write a helpful hint to guide them to the correct answer. DO NOT accidentally reveal the correct final answer!]" if round == 1 else "[Write a helpful hint AND the correct final answer, but NOT the full steps!]" if round == 2 else ""}</hint> 
<huang_analysis>[Provide a LONG and THOROUGH reasoning analysis (with specific examples in Huang's response that makes you say true or false) for Huang's response on these 4 criteria:
1. Correct: True if Huang's final answer matches the answer key, else False.
2. Logical: True if Huang's reasoning is logical and without any logical fallacies, else False.
3. Strong: True if Huang's proof/examples are strong (using proof by induction, contradiction, or strong proof of cases), else False.
4. Efficient: True if Huang's algorithm is efficient and does not have redundant steps, else False].
</huang_analysis>
<huang_correct>[Write "1" if criteria 1 (correct) is True for Huang, "0" if False]</huang_correct>
<huang_logical>[Write "1" if criteria 2 (logical) is True for Huang, "0" if False]</huang_logical>
<huang_strong>[Write "1" if criteria 3 (strong) is True for Huang, "0" if False]</huang_strong>
<huang_efficient>[Write "1" if criteria 4 (efficient) is True for Huang, "0" if False]</huang_efficient>
<ben_analysis>[Provide a LONG and THOROUGH reasoning analysis (with specific examples in Ben's response that makes you say true or false) for Ben's response on these 4 criteria:
1. Correct: True if Ben's final answer matches the answer key, else False.
2. Logical: True if Ben's reasoning is logical and without any logical fallacies, else False.
3. Strong: True if Ben's proof/examples are strong (using proof by induction, contradiction, or strong proof of cases), else False.
4. Efficient: True if Ben's algorithm is efficient and does not have redundant steps, else False.]
</ben_analysis>
<ben_correct>[Write "1" if criteria 1 (correct) is True for Ben, "0" if False]</ben_correct>
<ben_logical>[Write "1" if criteria 2 (logical) is True for Ben, "0" if False]</ben_logical>
<ben_strong>[Write "1" if criteria 3 (strong) is True for Ben, "0" if False]</ben_strong>
<ben_efficient>[Write "1" if criteria 4 (efficient) is True for Ben, "0" if False]</ben_efficient>
}}

STRICTLY FOLLOW THESE OUTPUT RULES:
1. Output ONLY the XML tags with content - NO additional text/comments
2. Use EXACTLY these 11 tag names as specified above
3. Ensure all opening and closing tags match properly
4. NEVER copy placeholder text like [this] - REPLACE with your content following the instructions inside the placeholder text
5. DO NOT write markdown code block markers

OUTPUT EXAMPLE (FOLLOW THE FORMAT EXACTLY):
<hint>To ensure the piecewise function is continuous, make sure that the functions match at the points where they change definition. Check at x = 2 and x = -2.</hint>
<huang_analysis>
1. Correct: Huang just wrote the high level idea without arriving at the final answer. So it is False.
2. Logical: Huang made a logical fallacy. For example, he incorrectly deducted that ... (long examples and analyses here). So it is False.
3. Strong: Huang's argument is not strong, because ... (long examples and analyses here). So it is False.
4. Efficient: Huang's algorithm contains redundant steps. For instance, ... (long examples and analyses here). So it is False.
</huang_analysis>
<huang_correct>0</huang_correct>
<huang_logical>0</huang_logical>
<huang_strong>0</huang_strong>
<huang_efficient>0</huang_efficient>
<ben_analysis>
1. Correct: Ben's final answer (x = 5) matches the answer key (x = 5). So it is True.
2. Logical: Ben's reasoning is logical and follows the correct approach. (long examples and analyses here). So it is True.
3. Strong: Ben did not use strong proof techniques, such as proof of induction or proof by contradiction. He used weak proofs in his arguments, such as ... (long examples and analyses here). So it is False
4. Efficient: Ben's algorithm is efficient. (long examples and analyses here). So it is True.
</ben_analysis>
<ben_correct>1</ben_correct>
<ben_logical>1</ben_logical>
<ben_strong>0</ben_strong>
<ben_efficient>1</ben_efficient>
"""
    
    raw_response, __, __, __ = get_llm_response_func([
        {"role": "system", "content": system_prompt},
    ], role="judge")
    
    print(f"[Judge] Raw response preview (first 200 chars): {raw_response[:200]}...")
    
    # Return the raw response directly - NO JSON validation or processing
    return raw_response
