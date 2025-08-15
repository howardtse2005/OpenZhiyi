import re
from difflib import SequenceMatcher
from codes.config import Config as cfg

"""
Check the format validity, response length, and originality for format scores, length scores, and originality scores.
Response Format: 10 points
Length: 10 points
Originality: 10 points
"""

### 1. CHECK RESPONSE FORMAT FOR FORMAT SCORES ###
def check_response_format(response, role):
    """
    Validate response format and extract content in one function.
    Returns (reasoning, answer, format_score, format_reason)
    """
    if role.lower() not in ['huang', 'ben']:
        return "", "", 0
            
    # Define the expected pattern for each role
    reasoning_tag = f"{role.lower()}_reasoning"
    answer_tag = f"{role.lower()}_answer"
        
    # Check for unauthorized tags (any XML-like tags that aren't the expected ones)
    # Improved pattern to avoid capturing mathematical expressions with < and >
    # Only match tags that start with a letter and contain valid XML tag characters
    unauthorized_tags = []
    tag_pattern = r'<([a-zA-Z][a-zA-Z0-9_]*(?:\s[^>]*)?)\s*>'
    all_tags = re.findall(tag_pattern, response)
        
    for tag in all_tags:
        # Clean the tag name (remove any attributes)
        tag_name = tag.split()[0] if ' ' in tag else tag
        if tag_name not in [reasoning_tag, answer_tag]:
            unauthorized_tags.append(tag_name)
        
    # Also check for closing tags
    closing_tag_pattern = r'</([a-zA-Z][a-zA-Z0-9_]*)\s*>'
    closing_tags = re.findall(closing_tag_pattern, response)
        
    for tag in closing_tags:
        if tag not in [reasoning_tag, answer_tag]:
            unauthorized_tags.append(f"/{tag}")
        
    if unauthorized_tags:
        punishment_reason = f"Contains unauthorized XML tags: {', '.join(set(unauthorized_tags))}"
        print(f"❌ {role.capitalize()} format validation failed - {punishment_reason}")
        return "", "", 0
        
    # Check for text outside the XML tags
    # Remove the valid XML content and check if there's any significant text left
    temp_response = response
        
    # Remove the reasoning section
    if f"<{reasoning_tag}>" in temp_response and f"</{reasoning_tag}>" in temp_response:
        start = temp_response.find(f"<{reasoning_tag}>")
        end = temp_response.find(f"</{reasoning_tag}>") + len(f"</{reasoning_tag}>")
        temp_response = temp_response[:start] + temp_response[end:]
        
    # Remove the answer section
    if f"<{answer_tag}>" in temp_response and f"</{answer_tag}>" in temp_response:
        start = temp_response.find(f"<{answer_tag}>")
        end = temp_response.find(f"</{answer_tag}>") + len(f"</{answer_tag}>")
        temp_response = temp_response[:start] + temp_response[end:]
        
    # Check if there's any meaningful text left (ignore whitespace)
    remaining_text = temp_response.strip()
    if remaining_text:
        punishment_reason = f"Contains text outside XML tags: '{remaining_text[:100]}...'" if len(remaining_text) > 100 else f"Contains text outside XML tags: '{remaining_text}'"
        print(f"❌ {role.capitalize()} format validation failed - {punishment_reason}")
        return "", "", 0
        
    # Pattern similar to DeepSeek R1: requires both reasoning and answer tags
    pattern = rf".*<{reasoning_tag}>.*?</{reasoning_tag}>.*<{answer_tag}>.*?</{answer_tag}>.*"
        
    # Check if the response matches the required format
    match = bool(re.match(pattern, response, re.DOTALL))
        
    if not match:
        punishment_reason = "Missing required XML tags"
        print(f"❌ {role.capitalize()} format validation failed - {punishment_reason}")
        return "", "", 0
        
    # Extract content if format is valid
    reasoning = ""
    answer = ""
        
    if role.lower() == 'huang':
        if "<huang_reasoning>" in response and "</huang_reasoning>" in response:
            start = response.find("<huang_reasoning>") + len("<huang_reasoning>")
            end = response.find("</huang_reasoning>")
            reasoning = response[start:end].strip()
            
        if "<huang_answer>" in response and "</huang_answer>" in response:
            start = response.find("<huang_answer>") + len("<huang_answer>")
            end = response.find("</huang_answer>")
            answer = response[start:end].strip()
        
    elif role.lower() == 'ben':
        if "<ben_reasoning>" in response and "</ben_reasoning>" in response:
            start = response.find("<ben_reasoning>") + len("<ben_reasoning>")
            end = response.find("</ben_reasoning>")
            reasoning = response[start:end].strip()
            
        if "<ben_answer>" in response and "</ben_answer>" in response:
            start = response.find("<ben_answer>") + len("<ben_answer>")
            end = response.find("</ben_answer>")
            answer = response[start:end].strip()
        
    print(f"⭐ {role.capitalize()} format validation passed")
    return reasoning, answer, cfg.format_score

### 2. CHECK RESPONSE LENGTH FOR LENGTH SCORES ###
def count_words(text):
    """
    Count the number of words in a text string.
    Returns integer count of words.
    """
    if not text or not text.strip():
        return 0
    
    # Remove extra whitespace and split by whitespace
    words = text.strip().split()
    return len(words)

def check_response_length(response, role, max_length=1000):
    """
    Check if the reasoning and answer lengths are within acceptable bounds.
    Returns (length_score, length_reason)
    """

    word_count = count_words(response)
    
    # Check reasoning length
    print(f"[LENGTH]{role.capitalize()} length = {word_count} words")

    if word_count > max_length:
        return 0
    
    return cfg.length_score

### 3. CHECK SIMILARITY FOR ORIGINALITY SCORES ###
def calculate_similarity(text1, text2):
    """
    Calculate similarity percentage between two texts using SequenceMatcher.
    Returns a float between 0.0 and 1.0 (0% to 100% similarity).
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to lowercase and strip whitespace for comparison
    text1_clean = text1.lower().strip()
    text2_clean = text2.lower().strip()
    
    # Use SequenceMatcher to calculate similarity
    similarity = SequenceMatcher(None, text1_clean, text2_clean).ratio()
    return similarity

def check_originality(participant_answer, judge_hint, participant_name, prev_huang_resp=None, prev_ben_resp=None):
    """
    Check if participant's answer is too similar to judge's hint, answer key, or opponent's answer.
    Returns (originality_score, originality_reason)
    """
    if not participant_answer:
        return 0

    hint_threshold = cfg.hint_threshold 
    prev_threshold = cfg.prev_threshold

    # Check similarity with judge hint
    if judge_hint:
        hint_similarity = calculate_similarity(participant_answer, judge_hint)
        if hint_similarity >= hint_threshold:
            print(f"🚨 CHEATING DETECTED: {participant_name}'s answer is {hint_similarity:.1%} similar to judge's hint!")
            return 0

    # Check similarity with previous responses
    if prev_huang_resp:
        huang_sim = calculate_similarity(participant_answer, prev_huang_resp)
        if huang_sim >= prev_threshold:
            print(f"🚨 CHEATING DETECTED: {participant_name}'s answer is {huang_sim:.1%} similar to previous Huang response!")
            return 0
    if prev_ben_resp:
        ben_sim = calculate_similarity(participant_answer, prev_ben_resp)
        if ben_sim >= prev_threshold:
            print(f"🚨 CHEATING DETECTED: {participant_name}'s answer is {ben_sim:.1%} similar to previous Ben response!")
            return 0

    return cfg.originality_score