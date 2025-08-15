import math

def reward_function(participant_score=0, opponent_score=0, gamma=5):
    """
    Calculate the reward for a participant based on their score compared to an opponent's score.
    
    Args:
        participant_score (int): The score of the participant.
        opponent_score (int): The score of the opponent.
    
    Returns:
        int: Reward value based on the comparison of scores.
    """
    
    # Normalize scores to 0-1 range
    participant_score = participant_score /100
    opponent_score = opponent_score /100

    diff = participant_score - opponent_score
    reward = 1 / (1 + math.exp(-gamma * diff))
    
    return reward