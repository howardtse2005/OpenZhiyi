import datetime
import json

def escape_answer_key(answer_key):
    """Escape curly braces and quotes for safe prompt interpolation."""
    if not answer_key:
        return ""
    return answer_key.replace('{', '{{').replace('}', '}}').replace('"', '\"')

def initialize_output_txt_file(output_file_path):
    """Initialize text file for output"""
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write("="*100 + "\n")
        file.write("DEBATE TRAINING RESULTS\n")
        file.write("="*100 + "\n\n")

def initialize_logs_file(logs_file_path):
    """Initialize logs file for error tracking"""
    with open(logs_file_path, 'w', encoding='utf-8') as file:
        file.write(f"DEBATE TRAINER ERROR LOGS - {datetime.datetime.now()}\n")
        file.write("="*50 + "\n\n")

def log_error(logs_file_path, error_name, error_details="", round_number=None):
    """Log errors to logs.txt file"""
    with open(logs_file_path, 'a', encoding='utf-8') as file:
        file.write("=" * 13 + "\n")
        file.write(f"{error_name}\n")
        file.write("=" * 13 + "\n")
        if error_details:
            file.write(f"{error_details}\n")
        file.write(f"Timestamp: {datetime.datetime.now()}\n")
        if round_number is not None:
            file.write(f"Round: {round_number}\n")
        file.write("\n")

def _escape_json_field(value):
    """Escape curly braces and quotes for safe JSONL writing."""
    if isinstance(value, str):
        return value.replace('{', '{{').replace('}', '}}').replace('"', '\\"')
    elif isinstance(value, dict) or isinstance(value, list):
        return json.dumps(value, ensure_ascii=False).replace('{', '{{').replace('}', '}}').replace('"', '\\"')
    return value

def append_huang_jsonl(huang_jsonl_path, huang_prompt, huang_resp, huang_reward):
    """Append to Huang .jsonl file, escaping curly braces and double quotes."""
    record = {
        "query": _escape_json_field(huang_prompt),
        "response": _escape_json_field(huang_resp),
        "reward": huang_reward
    }
    with open(huang_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def append_ben_jsonl(ben_jsonl_path, ben_prompt, ben_resp, ben_reward):
    """Append to Ben .jsonl file, escaping curly braces and double quotes."""
    record = {
        "query": _escape_json_field(ben_prompt),
        "response": _escape_json_field(ben_resp),
        "reward": ben_reward
    }
    with open(ben_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def append_result_to_txt(output_file_path, result):
    """Append a single result to the text file in a readable format"""
    with open(output_file_path, 'a', encoding='utf-8') as file:
        file.write(f"ROUND {result.get('round_number', 'N/A')}\n")
        file.write("-" * 50 + "\n")
        file.write(f"PROBLEM:\n{result.get('problem', 'N/A')}\n\n")
        file.write(f"SOLUTION:\n{result.get('solution', 'N/A')}\n\n")
        file.write(f"HINT:\n{result.get('hint', 'N/A')}\n\n")

        file.write("HUANG'S RESPONSE:\n")
        file.write(f"Raw Response: {result.get('huang_raw_response', 'N/A')}\n")
        file.write(f"Reasoning: {result.get('huang_reasoning', 'N/A')}\n")
        file.write(f"Answer: {result.get('huang_answer', 'N/A')}\n")
        file.write(f"Correctness Score: {result.get('huang_correctness_score', 0)}\n")
        file.write(f"Logic Score: {result.get('huang_logic_score', 0)}\n")
        file.write(f"Proof Score: {result.get('huang_proof_score', 0)}\n")
        file.write(f"Efficiency Score: {result.get('huang_efficiency_score', 0)}\n")
        file.write(f"Format Score: {result.get('huang_format_score', 0)}\n")
        file.write(f"Length Score: {result.get('huang_length_score', 0)}\n")
        file.write(f"Originality Score: {result.get('huang_originality_score', 0)}\n")
        file.write(f"Total Score: {result.get('huang_total_score', 0)}\n")
        file.write(f"Reward: {result.get('huang_reward', 0)}\n")
        file.write("\n")
        
        file.write("BEN'S RESPONSE:\n")
        file.write(f"Raw Response: {result.get('ben_raw_response', 'N/A')}\n")
        file.write(f"Reasoning: {result.get('ben_reasoning', 'N/A')}\n")
        file.write(f"Answer: {result.get('ben_answer', 'N/A')}\n")
        file.write(f"Correctness Score: {result.get('ben_correctness_score', 0)}\n")
        file.write(f"Logic Score: {result.get('ben_logic_score', 0)}\n")
        file.write(f"Proof Score: {result.get('ben_proof_score', 0)}\n")
        file.write(f"Efficiency Score: {result.get('ben_efficiency_score', 0)}\n")
        file.write(f"Format Score: {result.get('ben_format_score', 0)}\n")
        file.write(f"Length Score: {result.get('ben_length_score', 0)}\n")
        file.write(f"Originality Score: {result.get('ben_originality_score', 0)}\n")
        file.write(f"Total Score: {result.get('ben_total_score', 0)}\n")
        file.write(f"Reward: {result.get('ben_reward', 0)}\n")
        file.write("\n")
        
        file.write("JUDGE'S RESPONSE:\n")
        from codes.characters.judge import extract_tags_from_response
        judge_data = extract_tags_from_response(result.get('judge_response', '{}'))
        file.write(json.dumps(judge_data, indent=2))
        file.write("=" * 100 + "\n\n")
    print(f"Result saved to {output_file_path} (Round {result.get('round_number', 'N/A')})")
