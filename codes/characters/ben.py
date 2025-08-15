def generate_ben_response(problem, prev_huang_response="", prev_ben_response="", huang_correctness_score=0, huang_logic_score=0,
                            huang_proof_score=0, huang_efficiency_score=0,
                            ben_correctness_score=0, ben_logic_score=0, ben_proof_score=0, ben_efficiency_score=0,
                            ben_format_score=0, ben_length_score=0, ben_originality_score=0, judge_hint="", round=0, reward=0, llm_callback=None):
    """Generate Prof. Ben's response to a problem and Huang's solution"""
        
    if round == 1:
        history_info = ""
    else:
        history_info = f"""
CONTEXT HISTORY:
YOUR PREVIOUS ROUND'S RESPONSE: {prev_ben_response}

YOUR PREVIOUS ROUND'S FEEDBACK:
* Response quality feedback:
{"Your response was incorrect. " if ben_correctness_score == 0 else ""}{"Your response was logically flawed. " if ben_logic_score == 0 else ""}{"Your proof was weak or missing. " if ben_proof_score == 0 else ""}{"Your solution was inefficient. " if ben_efficiency_score == 0 else ""}

* Response format feedback:
{"Your response did not follow the XML format. " if ben_format_score == 0 else ""}{"Your response length was too long. " if ben_length_score == 0 else ""}{"Your response was too similar with judge's hints or previous responses. " if ben_originality_score == 0 else ""}

HUANG'S PREVIOUS ROUND'S RESPONSE: {prev_huang_response}

HUANG'S PREVIOUS ROUND'S FEEDBACK:
{"Huang's response was incorrect. " if huang_correctness_score == 0 else ""}{"Huang's response was logically flawed. " if huang_logic_score == 0 else ""}{"Huang's proof was weak or missing. " if huang_proof_score == 0 else ""}{"Huang's solution was inefficient. " if huang_efficiency_score == 0 else ""}

YOUR PREVIOUS ROUND'S REWARD (Scaled 0-1): {reward}

In your <ben_reasoning> tag, you MUST include:
* The strengths and weaknesses of Huang's solution (if any)
* How you would adjust your current round's response based on Huang's previous round's solution
"""  

    system_prompt = f"""
ROLEPLAY: You are Prof. Ben. You are a genius problem solver.
You will compete with another professor, Prof. Huang.
There will be maximum 3 rounds in total, and you will alternate responses with Huang. Current round is round {round}.
There will be a judge who will score both of you and may provide hints.
Your reward depends on how well you perform compared to Huang.

RULES:
1. Your response MUST be a step-by-step detailed solution to the problem, which includes strong mathematical proofs.
2. DO NOT provoke the judge or write anything else unrelated to the problem!
3. Your output MUST STRICTLY follow this XML: "<ben_reasoning> [YOUR_REASONING_HERE] </ben_reasoning> <ben_answer> [YOUR_FINAL_ANSWER_HERE] </ben_answer>". DO NOT write any other tags! DO NOT write anything outside the <ben_reasoning> and <ben_answer> tags!

{f"JUDGE'S HINT: <hint> {judge_hint} </hint>" if judge_hint else ""}

"""
    
    messages = [
        {'role': 'system', 'content': system_prompt + history_info},
        {'role': 'user', 'content': problem}
    ]
    raw_response, response_tensor, query_tensor, query_dict = llm_callback(messages, temp=0.9, role="ben", problem=problem)

    print(f"[Ben] Response generated successfully")
    
    # Return the raw response directly - no JSON processing needed
    return raw_response, response_tensor, query_tensor, query_dict, str(messages)