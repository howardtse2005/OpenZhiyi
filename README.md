# Project OpenZhiyi: Reinforcement Learning through small LLMs through debate mechanism with Supervised LLM Judge to improve math reasoning

This project is about leveraging PPO (Proximal Policy Optimization) traditionally used in RLHF (Reinforcement Learning through Human Feedback). Traditionally, PPO works by a pretrained base model giving responses to a set of queries (problems) and humans give scores to the quality of the responses. Those scores will be normalized and become the RL reward of the model. In Project OpenZhiyi, 2 small models (Prof. Huang and Prof. Ben) are given a set of math problems and an LLM judge (larger model, has access to the solutions / answer keys) will give scores (and hints if they cannot full scores) to the 2 models' responses. The scores are calculated based on these 7 rubrics:
1. Correctness: The response's final answer matches the answer key
2. Logical: The reasoning flow is logical and follows the solution / answer key without logical fallacies
3. Strong proof: The response contains strong mathematical proofs
4. Efficiency: No redundancy and method / algorithm is efficient
5. Format correctness: Follow the output format which contains reasoning and final answer
6. Response length: The response is not too long (the maximum limit is dynamic, adjusted to the solution / answer key. If the solution is long, the model is allowed to reason longer)
7. Originality: The response should not be too similar with Judge's hints or the previous round's response (if the models cannot get full scores in the first round.
The weightings for the 7 rubrics can be adjusted in the config.py

There will be at most 3 rounds per problem. The round will stop early if both models achieve perfect scores. The model's reward is computed by normalizing the difference between its total score and its opponent's total score on a sigmoid scale from 0-1.

## Getting Started

1.  **Clone the repository:**
    ```bash
    cd ~
    git clone https://github.com/howardtse2005/OpenZhiyi.git
    cd OpenZhiyi/
    ```

2.  **Install dependencies:**
    The code was tested on Python 3.10 in Conda Environment
    ```bash
    conda create -n openzhiyi python=3.10
    conda activate openzhiyi
    conda install pytorch::pytorch conda-forge::trl conda-forge::transformers conda-forge::peft
    ```
    Then install Judge LLM API (here it is OpenAI API) by following the official documentations.

3.  **Edit the configuration settings:**
    Navigate to codes/config.py and adjust the config settings.

4.  **Prepare dataset:**
    The example problems dataset can be found on test.csv. The csv file should contain 'problem' and 'solution' rows. The 'problem' row will be visible to both models and the judge, while 'solution' row is only visible to judge.
    
5.  **Train the model:**
    ```bash
    cd OpenZhiyi/codes
    python3 main.py
    ```
    The trained models will be saved to huang_model and ben_model

## Future Directions

*   Optimize the trainers to save memory
*   Investigate different training methods aside from PPO
*   Train and test on multiple datasets, including math, algorithms, and reasoning puzzles
