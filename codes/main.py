import csv
from transformers import AutoTokenizer
import torch
import signal
import sys
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer

# Import custom modules
from codes.config import Config as cfg
from codes.characters.huang import generate_huang_response
from codes.characters.ben import generate_ben_response
from codes.characters.judge import generate_judge_response, extract_tags_from_response
from codes.utils.checker import check_response_format, check_response_length, check_originality
from codes.utils.reward import reward_function
from codes.utils.logger import log_error, append_huang_jsonl, append_ben_jsonl, append_result_to_txt, initialize_logs_file, initialize_output_txt_file, escape_answer_key
from codes.utils.memory_utils import memory_cleanup, init_nvml, print_memory_stats

class Debate:
    def __init__(self):       
        self.base_model_name = cfg.base_model_name
        self.huang_model_path = cfg.huang_model_path
        self.ben_model_path = cfg.ben_model_path

        self.huang_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.base_model_name,
            quantization_config=cfg.huang_quantization_config,
            peft_config=cfg.lora_config,
            device_map="cuda:0",
        )

        self.ben_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.base_model_name,
            quantization_config=cfg.ben_quantization_config,
            peft_config=cfg.lora_config,
            device_map="cuda:0",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize for new debate
        self.huang_correctness_score, self.huang_logic_score, self.huang_proof_score, self.huang_efficiency_score, self.huang_format_score, self.huang_length_score, self.huang_originality_score, self.huang_reward = 0, 0, 0, 0, 0, 0, 0, 0
        self.ben_correctness_score, self.ben_logic_score, self.ben_proof_score, self.ben_efficiency_score, self.ben_format_score, self.ben_length_score, self.ben_originality_score, self.ben_reward = 0, 0, 0, 0, 0, 0, 0, 0
        self.prev_huang_resp, self.prev_ben_resp, self.last_judge_analysis = "", "", ""
        self.current_round = 1
        
        # Initialize logs file
        self.logs_file_path = cfg.logs_file_path
        initialize_logs_file(self.logs_file_path)

        # Initialize Huang and Ben .jsonl files to document the debate from Huang and Ben POV
        self.huang_jsonl_path = cfg.huang_jsonl_path
        self.ben_jsonl_path = cfg.ben_jsonl_path

        open(self.huang_jsonl_path, "w", encoding="utf-8").close()
        open(self.ben_jsonl_path, "w", encoding="utf-8").close()

        # Initialize huang and ben ppo models
        config = cfg.ppo_config

        self.huang_trainer = PPOTrainer(
            model=self.huang_model,
            config=config,
            tokenizer=self.tokenizer,
            dataset=None,
        )

        self.ben_trainer = PPOTrainer(
            model=self.ben_model,
            config=config,
            tokenizer=self.tokenizer,
            dataset=None,
        )

        init_nvml()

    def get_llm_response(self, messages, temp=0.9, role="huang", problem=""):
        """
        LLM router: role='huangben' uses HuggingFace Qwen locally, role='judge' uses DeepSeek-V3 mandatory.
        """
        if role == "judge":
            print(f"[DEBUG] Judge API Call - Using DeepSeek-V3...")
            try:
                response = cfg.get_api_response(messages, temp)
                print(f"[DEBUG] Judge API call successful!")
                return response, None, None, None
            except Exception as e:
                error_msg = f"Judge API Error: {str(e)}"
                print(error_msg)
                log_error(self.logs_file_path, "JUDGE_API_ERROR", f"Judge API failed: {str(e)}", self.current_round)
                return error_msg, None, None, None
        else:
            print(f"[DEBUG] Using HuggingFace for role: {role}")
            try:
                # Use HuggingFace for Huang and Ben
                return self.get_huggingface_response(messages, temp, role, problem=problem)
            except Exception as e:
                error_msg = f"API Error (HuggingFace): {str(e)}"
                print(error_msg)
                log_error(self.logs_file_path, "HUGGINGFACE_API_ERROR", f"HuggingFace failed for role {role}: {str(e)}", self.current_round)
                return error_msg, None, None, None

    def get_huggingface_response(self, messages, temp=0.9, role="huang", problem=""):
        """Get response from PPO-optimized model with proper tensor shapes"""
        try:
            # Convert messages (list of dicts) to a prompt string using chat template
            chat_messages = []
            for msg in messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    chat_messages.append({"role": msg.role, "content": msg.content})
                elif isinstance(msg, dict):
                    chat_messages.append(msg)
            prompt = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            # Tokenize prompt and match dataset format
            input_ids = self.tokenizer.encode(prompt)
            query_dict = {"query": prompt, "input_ids": input_ids}

            query_tensor = torch.tensor(query_dict["input_ids"], dtype=torch.long).to("cuda:0")

            # Match dataset format
            query_dict = {"query": query_tensor, "input_ids": input_ids}
            prompt_length = query_tensor.shape[0]

            # Generation settings
            generation_kwargs = {
                "min_length": -1,
                "top_k": 20,
                "top_p": 0.8,
                "min_p": 0.0,
                "do_sample": True,
                "temperature": temp,
                "pad_token_id": self.tokenizer.eos_token_id,
                "max_new_tokens": 20000,
                "use_cache": False,
            }
    
            # Generate response
            trainer = self.huang_trainer if role == "huang" else self.ben_trainer
            response_tensor = trainer.generate(query_tensor, **generation_kwargs)[0][prompt_length:]
            response = self.tokenizer.decode(response_tensor, skip_special_tokens=True)

            return response, response_tensor, query_tensor, query_dict
            
        except Exception as e:
            print(f"Error in {role} generation: {str(e)}")
            raise
        finally:
            print_memory_stats()
            memory_cleanup(self.huang_model, self.ben_model)

    def train(self, input_csv_path, output_file_path):
        try:
            initialize_output_txt_file(output_file_path)
            self.output_file_path = output_file_path
            self.output_file_initialized = True
        except Exception as e:
            log_error(self.logs_file_path, "OUTPUT_FILE_INIT_ERROR", f"Error initializing output file: {str(e)}", self.current_round)
            return

        def signal_handler(sig, frame):
            print(f"\nInterrupt received. Data has been saved to {output_file_path}")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        total_processed = 0
        
        try:
            with open(input_csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    problem = row.get('problem', '')
                    solution = row.get('solution', '')
                    
                    if problem and solution:
                        try:
                            debate_results = self.process_debate(problem, solution)
                            total_processed += len(debate_results)
                            print(f"Completed processing problem. Total results so far: {total_processed}")
                        except KeyboardInterrupt:
                            print(f"\nInterrupt received during processing. {total_processed} results saved to {output_file_path}")
                            raise
                        except Exception as e:
                            print(f"Error processing problem: {str(e)}")
                            log_error(self.logs_file_path, "PROBLEM_PROCESSING_ERROR", f"Error processing problem: {str(e)}\nProblem: {problem[:100]}...", self.current_round)
                            continue
                    else:
                        print(f"Skipping row with missing problem or solution")
                        log_error(self.logs_file_path, "MISSING_DATA_ERROR", "Row with missing problem or solution data", self.current_round)
        except FileNotFoundError:
            log_error(self.logs_file_path, "INPUT_FILE_NOT_FOUND", f"Input CSV file not found: {input_csv_path}", self.current_round)
            print(f"Error: Input CSV file not found: {input_csv_path}")
            return
        except Exception as e:
            log_error(self.logs_file_path, "CSV_READ_ERROR", f"Error reading CSV file: {str(e)}", self.current_round)
            print(f"Error reading CSV file: {str(e)}")
            return
        finally:
            self.huang_trainer.save_pretrained(self.huang_model_path)
            self.ben_trainer.save_pretrained(self.ben_model_path)

        print(f"Processing completed! Total rows processed: {total_processed}")
        print(f"All data saved to {output_file_path}")

    def process_debate(self, problem, solution):
        """Process a single debate and return structured results"""
        print(f"Processing problem: {problem[:100]}...")
        
        # Initialize for new debate
        self.huang_correctness_score, self.huang_logic_score, self.huang_proof_score, self.huang_efficiency_score, self.huang_format_score, self.huang_length_score, self.huang_originality_score, self.huang_reward = 0, 0, 0, 0, 0, 0, 0, 0
        self.ben_correctness_score, self.ben_logic_score, self.ben_proof_score, self.ben_efficiency_score, self.ben_format_score, self.ben_length_score, self.ben_originality_score, self.ben_reward = 0, 0, 0, 0, 0, 0, 0, 0
        self.prev_huang_resp, self.prev_ben_resp, self.last_judge_analysis = "", "", ""
        self.current_round = 1

        answer_key = escape_answer_key(solution)
        
        results = []
        
        for round_num in range(1, 4):  # 3 rounds maximum
            self.current_round = round_num
            print(f"Processing Round {round_num}")

            # Reset scores for each round
            huang_correctness_score, huang_logic_score, huang_proof_score, huang_efficiency_score, huang_format_score, huang_length_score, huang_originality_score = 0, 0, 0, 0, 0, 0, 0
            ben_correctness_score, ben_logic_score, ben_proof_score, ben_efficiency_score, ben_format_score, ben_length_score, ben_originality_score = 0, 0, 0, 0, 0, 0, 0

            # Get judge's hint and feedback
            judge_hint = ""
            try:
                judge_data = extract_tags_from_response(self.last_judge_analysis) if self.last_judge_analysis else {}
                if round_num > 1:
                    judge_hint = judge_data.get("hint", "")
            except Exception as e:
                log_error(self.logs_file_path, "JUDGE_HINT_EXTRACTION_ERROR", f"Error extracting judge hint: {str(e)}", self.current_round)
                judge_hint = ""


            # Get Huang's response
            try:
                huang_resp, huang_response_tensor, huang_query_tensor, huang_batch, huang_prompt = generate_huang_response(
                    problem,
                    prev_huang_response=self.prev_huang_resp,
                    prev_ben_response=self.prev_ben_resp,
                    huang_correctness_score=self.huang_correctness_score,
                    huang_logic_score=self.huang_logic_score,
                    huang_proof_score=self.huang_proof_score,
                    huang_efficiency_score=self.huang_efficiency_score,
                    huang_format_score=self.huang_format_score,
                    huang_length_score=self.huang_length_score,
                    huang_originality_score=self.huang_originality_score,
                    ben_correctness_score=self.ben_correctness_score,
                    ben_logic_score=self.ben_logic_score,
                    ben_proof_score=self.ben_proof_score,
                    ben_efficiency_score=self.ben_efficiency_score,
                    judge_hint=judge_hint,
                    round=round_num,
                    reward=self.huang_reward,
                    llm_callback=self.get_llm_response
                )
                if not huang_resp or not huang_resp.strip():
                    log_error(self.logs_file_path, "EMPTY_HUANG_RESPONSE", f"Huang returned empty response in round {round_num}", self.current_round)
                    huang_resp = "ERROR: Empty response from Huang"
            except Exception as e:
                log_error(self.logs_file_path, "HUANG_GENERATION_ERROR", f"Error generating Huang response: {str(e)}", self.current_round)
                huang_resp = f"ERROR: {str(e)}"

            # Get Ben's response
            try:
                ben_resp, ben_response_tensor, ben_query_tensor, ben_batch, ben_prompt = generate_ben_response(
                    problem, 
                    prev_huang_response=self.prev_huang_resp,
                    prev_ben_response=self.prev_ben_resp,
                    huang_correctness_score=self.huang_correctness_score,
                    huang_logic_score=self.huang_logic_score,
                    huang_proof_score=self.huang_proof_score,
                    huang_efficiency_score=self.huang_efficiency_score,
                    ben_correctness_score=self.ben_correctness_score,
                    ben_logic_score=self.ben_logic_score,
                    ben_proof_score=self.ben_proof_score,
                    ben_efficiency_score=self.ben_efficiency_score,
                    ben_format_score=self.ben_format_score,
                    ben_length_score=self.ben_length_score,
                    ben_originality_score=self.ben_originality_score,
                    judge_hint=judge_hint,
                    round=round_num,
                    reward=self.ben_reward,
                    llm_callback=self.get_llm_response
                )
                if not ben_resp or not ben_resp.strip():
                    log_error(self.logs_file_path, "EMPTY_BEN_RESPONSE", f"Ben returned empty response in round {round_num}", self.current_round)
                    ben_resp = "ERROR: Empty response from Ben"
            except Exception as e:
                log_error(self.logs_file_path, "BEN_GENERATION_ERROR", f"Error generating Ben response: {str(e)}", self.current_round)
                ben_resp = f"ERROR: {str(e)}"
            
            # Set response length limit
            max_response_length = len(solution.split()) * cfg.length_tolerance if solution else 1000

            # Check Huang's response (format, length, originality)
            huang_reasoning, huang_answer, huang_format_score = check_response_format(huang_resp, 'huang')
            huang_length_score = check_response_length(huang_resp, 'huang', max_response_length)
            huang_originality_score = check_originality(
                huang_resp, judge_hint,"Huang", self.prev_huang_resp, self.prev_ben_resp
            )

            # Check Ben's response (format, length, originality)
            ben_reasoning, ben_answer, ben_format_score = check_response_format(ben_resp, 'ben')
            ben_length_score = check_response_length(ben_resp, 'ben', max_response_length)
            ben_originality_score = check_originality(
                ben_resp, judge_hint, "Ben", self.prev_huang_resp, self.prev_ben_resp
            )

            # Truncate responses to max_response_length words to prevent excessive length
            huang_resp = ' '.join(huang_resp.split()[:max_response_length]) + ('[TRUNCATED BECAUSE TOO LONG]' if len(huang_resp.split()) > max_response_length else '')
            ben_resp = ' '.join(ben_resp.split()[:max_response_length]) + ('[TRUNCATED BECAUSE TOO LONG]' if len(ben_resp.split()) > max_response_length else '')

            # Get Judge's response
            try:
                judge_resp = generate_judge_response(problem, huang_resp, ben_resp,  
                                                   self.get_llm_response,
                                                   answer_key=answer_key, round=round_num)
                if not judge_resp or not judge_resp.strip():
                    log_error(self.logs_file_path, "EMPTY_JUDGE_RESPONSE", f"Judge returned empty response in round {round_num}", self.current_round)
                    judge_resp = "ERROR: Empty response from Judge"
            except Exception as e:
                log_error(self.logs_file_path, "JUDGE_GENERATION_ERROR", f"Error generating Judge response: {str(e)}", self.current_round)
                judge_resp = f"ERROR: {str(e)}"
            
            # Update scores
            try:
                judge_data = extract_tags_from_response(judge_resp)
                try:
                    huang_correctness_score = cfg.correctness_score * int(judge_data.get("huang_correct", "0"))
                    huang_logic_score = cfg.logic_score * int(judge_data.get("huang_logical", "0"))
                    huang_proof_score = cfg.proof_score * int(judge_data.get("huang_strong", "0"))
                    huang_efficiency_score = cfg.efficiency_score * int(judge_data.get("huang_efficient", "0"))
                    ben_correctness_score = cfg.correctness_score * int(judge_data.get("ben_correct", "0"))
                    ben_logic_score = cfg.logic_score * int(judge_data.get("ben_logical", "0"))
                    ben_proof_score = cfg.proof_score * int(judge_data.get("ben_strong", "0"))
                    ben_efficiency_score = cfg.efficiency_score * int(judge_data.get("ben_efficient", "0"))
                except ValueError as ve:
                    print("Error extracting scores")
                    log_error(self.logs_file_path, "SCORE_EXTRACTION_ERROR", f"ValueError extracting scores: {str(ve)}", self.current_round)
            except Exception as e:
                print(f"Error updating scores: {str(e)}")
                log_error(self.logs_file_path, "SCORE_UPDATE_ERROR", f"Error updating scores: {str(e)}", self.current_round)
            
            # Store this round's results
            huang_total_score = huang_correctness_score + huang_logic_score + huang_proof_score + huang_efficiency_score + huang_format_score + huang_length_score + huang_originality_score
            ben_total_score = ben_correctness_score + ben_logic_score + ben_proof_score + ben_efficiency_score + ben_format_score + ben_length_score + ben_originality_score

            # Calculate the reward for this round
            huang_reward = reward_function(huang_total_score, ben_total_score, 5)
            ben_reward = reward_function(ben_total_score, huang_total_score, 5)

            # Write to huang_dataset.jsonl and ben_dataset.jsonl
            append_huang_jsonl(self.huang_jsonl_path, huang_prompt, huang_resp, huang_reward)
            append_ben_jsonl(self.ben_jsonl_path, ben_prompt, ben_resp, ben_reward)

            result = {
                'problem': problem,
                'solution': solution,
                'hint': judge_hint,
                'huang_raw_response': huang_resp,
                'huang_reasoning': huang_reasoning,
                'huang_answer': huang_answer,
                'ben_raw_response': ben_resp,
                'ben_reasoning': ben_reasoning,
                'ben_answer': ben_answer,
                'judge_response': judge_resp,
                'huang_correctness_score': huang_correctness_score,
                'huang_logic_score': huang_logic_score,
                'huang_proof_score': huang_proof_score,
                'huang_efficiency_score': huang_efficiency_score,
                'huang_format_score': huang_format_score,
                'huang_length_score': huang_length_score,
                'huang_originality_score': huang_originality_score,
                'huang_total_score': huang_total_score,
                'huang_reward': huang_reward,
                'ben_correctness_score': ben_correctness_score,
                'ben_logic_score': ben_logic_score,
                'ben_proof_score': ben_proof_score,
                'ben_efficiency_score': ben_efficiency_score,
                'ben_format_score': ben_format_score,
                'ben_length_score': ben_length_score,
                'ben_originality_score': ben_originality_score,
                'ben_total_score': ben_total_score,
                'ben_reward': ben_reward,
                'huang_prompt': huang_prompt,
                'ben_prompt': ben_prompt,
                'round_number': round_num
            }
            results.append(result)

            # Immediately save this result to text file
            try:
                append_result_to_txt(self.output_file_path, result)
            except Exception as e:
                log_error(self.logs_file_path, "FILE_WRITE_ERROR", f"Error writing result to file: {str(e)}", self.current_round)

            # Update last judge analysis
            self.last_judge_analysis = judge_resp

            # Update scores and previous response for next round
            self.huang_correctness_score = huang_correctness_score
            self.huang_logic_score = huang_logic_score
            self.huang_proof_score = huang_proof_score
            self.huang_efficiency_score = huang_efficiency_score
            self.huang_format_score = huang_format_score
            self.huang_length_score = huang_length_score
            self.huang_originality_score = huang_originality_score
            self.huang_reward = huang_reward
            self.ben_correctness_score = ben_correctness_score
            self.ben_logic_score = ben_logic_score
            self.ben_proof_score = ben_proof_score
            self.ben_efficiency_score = ben_efficiency_score
            self.ben_format_score = ben_format_score
            self.ben_length_score = ben_length_score
            self.ben_originality_score = ben_originality_score
            self.ben_reward = ben_reward
            self.prev_huang_resp = huang_resp
            self.prev_ben_resp = ben_resp
                
            # Tensorialize the rewards
            huang_reward_tensor = torch.tensor([huang_reward], dtype=torch.float32)
            ben_reward_tensor = torch.tensor([ben_reward], dtype=torch.float32)

            # Train after each round
            try:
                self.huang_model.gradient_checkpointing_enable()
                self.ben_model.gradient_checkpointing_enable()
                self.huang_model.config.use_cache = False
                self.ben_model.config.use_cache = False
                huang_stats = self.huang_trainer.step(
                    [huang_query_tensor],
                    [huang_response_tensor], 
                    [huang_reward_tensor],
                )
                ben_stats = self.ben_trainer.step(
                    [ben_query_tensor],
                    [ben_response_tensor], 
                    [ben_reward_tensor],
                )
                self.huang_trainer.log_stats(huang_stats, huang_batch, [huang_reward_tensor])
                self.ben_trainer.log_stats(ben_stats, ben_batch, [ben_reward_tensor])
                print(f"Round {round_num} training completed")
            except Exception as e:
                print(f"Training error: {e}")
                log_error(self.logs_file_path, "TRAINING_ERROR", f"Error during training step: {str(e)}", self.current_round)
            
            print_memory_stats()
            memory_cleanup(self.huang_model, self.ben_model)
            print(f"Round {round_num} completed")

            if huang_total_score == 100 and ben_total_score == 100:
                print("Both Huang and Ben achieved perfect scores. Ending debate early.")
                break
        
        return results

def main():
    """Main function to run the training data generation"""
    debate = Debate()
    
    input_csv = cfg.input_csv_path
    output_file = cfg.output_txt_path

    debate.train(input_csv, output_file)

if __name__ == "__main__":
    main()