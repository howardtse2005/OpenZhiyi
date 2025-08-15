import torch
from openai import AzureOpenAI
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import PPOConfig
from transformers import AutoTokenizer

from codes.api_config import OPENAI_API_KEY, AZURE_ENDPOINT, API_VERSION
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

class Config:
    # Initialize judge model
    judge_model_name = "gpt-4o-mini" # Options: "gpt-4o-mini", "deepseek-v3"  
    
    # Initialize huggingface model that you want to train
    base_model_name = "Qwen/Qwen3-4B"
    huang_model_path = "../huang_model" # Post-trained huang model path
    ben_model_path = "../ben_model" # Post-trained ben model path

    # Initialize input CSV file path
    input_csv_path = "../test.csv"

    # Initialize output text file path for debate process debugging
    output_txt_path = "../result.txt"

    # Initialize error logs file
    logs_file_path = "../logs.txt"

    # Initialize Huang and Ben .jsonl files to document the debate from Huang and Ben POV
    huang_jsonl_path = "../huang_dataset.jsonl"
    ben_jsonl_path = "../ben_dataset.jsonl"

    # Initialize score weights (out of 100)
    correctness_score = 15
    logic_score = 20
    proof_score = 20
    efficiency_score = 15
    format_score = 10
    length_score = 10
    originality_score = 10

    # Initialize maximum response length tolerance
    length_tolerance = 10  # Allow up to 10x the solution length

    # Initialize similarity thresholds
    hint_threshold = 0.8  # 80% similarity threshold for judge's hint
    prev_threshold = 0.3  # 30% similarity threshold for previous responses

    # Initialize Deepseek/GPT-4o-mini API
    def get_api_response (messages, temp=0.7):
        if Config.judge_model_name == "gpt-4o-mini":
            client = AzureOpenAI(
                azure_endpoint=AZURE_ENDPOINT,
                api_key=OPENAI_API_KEY,
                api_version=API_VERSION
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
        else:
            client = ChatCompletionsClient(
                endpoint="https://api-iw.azure-api.net/sig-shared-deepseek",
                credential=AzureKeyCredential(OPENAI_API_KEY),
            )
            response = client.complete(
                model="Deepseek-V3",
                messages=messages,
                temperature=temp
            )
        return response.choices[0].message.content

    # Initialize PPO Config
    ppo_config = PPOConfig(
        model_name=base_model_name,
        learning_rate=1.41e-5,
        batch_size=1,
        mini_batch_size=1,
        ppo_epochs=1,
        max_grad_norm=1.0,
        optimize_cuda_cache=True,
        cliprange_value=0.2,
        use_score_norm=False,
        use_score_scaling=False,
        gradient_checkpointing=True
    )

    # Initialize PPO trainer generation kwargs
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    generation_kwargs = {
        "min_length": -1,
        "top_k": 20,
        "top_p": 0.8,
        "min_p": 0.0,
        "do_sample": True,
        "temperature": 0.7,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 20000,
        "use_cache": False,
    }
    # Initialize LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "v_proj",
        ],
    )

    # Initialize quantization config
    huang_quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    ben_quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )