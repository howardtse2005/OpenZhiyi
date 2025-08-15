def generate_huang_response(problem, prev_huang_response="", prev_ben_response="", huang_correctness_score=0, huang_logic_score=0, 
                            huang_proof_score=0, huang_efficiency_score=0, 
                            huang_format_score=0, huang_length_score=0, 
                            huang_originality_score=0, ben_correctness_score=0, ben_logic_score=0, 
                            ben_proof_score=0, ben_efficiency_score=0, judge_hint="", round=0, reward=0, llm_callback=None):
    """Generate a response from Prof. Huang Zhiyi's persona"""
    if round == 1:
        history_info = ""
    else:
        history_info = f"""
CONTEXT HISTORY:
YOUR PREVIOUS ROUND'S RESPONSE: {prev_huang_response}

YOUR PREVIOUS ROUND'S FEEDBACK:
* Response quality feedback:
{"Your response was incorrect. " if huang_correctness_score == 0 else ""}{"Your response was logically flawed. " if huang_logic_score == 0 else ""}{"Your proof was weak or missing. " if huang_proof_score == 0 else ""}{"Your solution was inefficient. " if huang_efficiency_score == 0 else ""}

* Response format feedback:
{"Your response did not follow the XML format. " if huang_format_score == 0 else ""}{"Your response length was too long. " if huang_length_score == 0 else ""}{"Your response was too similar with judge's hints or previous responses. " if huang_originality_score == 0 else ""}

BEN'S PREVIOUS ROUND'S RESPONSE: {prev_ben_response}

BEN'S PREVIOUS ROUND'S FEEDBACK:
{"Ben's response was incorrect. " if ben_correctness_score == 0 else ""}{"Ben's response was logically flawed. " if ben_logic_score == 0 else ""}{"Ben's proof was weak or missing. " if ben_proof_score == 0 else ""}{"Ben's solution was inefficient. " if ben_efficiency_score == 0 else ""}

YOUR PREVIOUS ROUND'S REWARD (Scaled 0-1): {reward}

In your <huang_reasoning> tag, you MUST include:
* The strength of Ben's solution (if any)
* The weaknesses of Ben's solution (if any)
* How you would adjust your current round's response based on Ben's previous round's solution
"""

    system_prompt = f"""
ROLEPLAY: You are Prof. Huang. You are a genius problem solver.
You will compete with another professor, Prof. Ben.
There will be maximum 3 rounds in total, and you will alternate responses with Ben. Current round is round {round}.
There will be a judge who will score both of you and may provide hints.
Your reward depends on how well you perform compared to Ben.

RULES:
1. Your response MUST be a step-by-step detailed solution to the problem, which includes strong proofs (proof by induction, contradiction, or strong proof by cases).
2. DO NOT provoke the judge! DO NOT write anything else unrelated to the problem!
3. Your output MUST STRICTLY follow this XML: "<huang_reasoning> [YOUR_REASONING_HERE] </huang_reasoning> <huang_answer> [YOUR_FINAL_ANSWER_HERE] </huang_answer>". DO NOT write any other tags! DO NOT write anything outside the <huang_reasoning> and <huang_answer> tags!
4. Your output MUST NOT be too short or too long.
5. Your solution MUST NOT be similar to previous round's responses or Judge's hints.

{f"JUDGE'S HINT: <hint> {judge_hint} </hint>" if judge_hint else ""}

"""

    # Build the messages as a list of dicts, not a string
    messages = [
        {'role': 'system', 'content': system_prompt + history_info},
        {'role': 'user', 'content': problem}
    ]
    raw_response, response_tensor, query_tensor, query_dict = llm_callback(messages, temp=0.9, role="huang", problem=problem)

    print(f"[Huang] Response generated successfully")
    
    # Return the raw response directly - no JSON processing needed
    return raw_response, response_tensor, query_tensor, query_dict, str(messages)