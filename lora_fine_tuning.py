import os
import jsonlines
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import logging
from accelerate import init_empty_weights
import bitsandbytes as bnb

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(dataset_path):
    """JSONL 데이터셋을 로드하는 함수"""
    try:
        inputs = []
        with jsonlines.open(dataset_path) as reader:
            for obj in reader:
                combined_text = f"### 입력:\n{obj['text']}\n\n### 출력:\n{obj['label']}\n"
                inputs.append({"text": combined_text})
        logger.info(f"데이터 {len(inputs)}개 로드 완료")
        return inputs
    except Exception as e:
        logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
        raise


def tokenize_function(examples, tokenizer):
    """데이터를 토크나이징하는 함수"""
    try:
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=512,
            return_tensors=None,
        )
    except Exception as e:
        logger.error(f"토크나이징 중 오류 발생: {str(e)}")
        raise


def find_all_linear_names(model):
    """모델에서 모든 Linear 레이어의 이름을 찾는 함수"""
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)


def main():
    # 모델 및 토크나이저 로드
    model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    dataset_path = "data/split_01.jsonl"

    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # 4비트 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config
        )

        # 모델 준비
        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()

        # 데이터 로드 및 전처리
        raw_data = load_data(dataset_path)
        dataset = Dataset.from_list(raw_data)
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )

        # LoRA 설정
        target_modules = find_all_linear_names(model)
        logger.info(f"Target modules: {target_modules}")

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
            inference_mode=False,
        )

        # LoRA 모델 준비
        lora_model = get_peft_model(model, lora_config)
        logger.info("트레이닝 가능한 파라미터의 비율: %.2f%%" % (
                100 * sum(p.numel() for p in lora_model.parameters() if p.requires_grad) /
                sum(p.numel() for p in lora_model.parameters())
        ))

        # 학습 설정
        training_args = TrainingArguments(
            output_dir="./lora_output",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=100,
            save_strategy="steps",
            save_steps=500,
            evaluation_strategy="no",
            warmup_steps=100,
            weight_decay=0.01,
            optim="paged_adamw_32bit"
        )

        # 데이터 콜레이터 설정
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # 트레이너 설정
        trainer = Trainer(
            model=lora_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # 학습 실행
        trainer.train()

        # 모델 저장
        lora_model.save_pretrained("./lora_output/final_model")
        logger.info("학습 완료 및 모델 저장됨")

    except Exception as e:
        logger.error(f"학습 중 오류 발생: {str(e)}")
        raise


if __name__ == "__main__":
    main()