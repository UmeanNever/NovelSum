import os
import json
import argparse
from typing import List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM
from utils import *

def truncate_texts(texts: List[str], tokenizer, max_length: int = 256) -> List[str]:
    truncated_texts = []
    for text in texts:
        encoded_input = tokenizer(text, return_tensors="pt")
        token_count = len(encoded_input["input_ids"][0])
        if token_count > max_length:
            truncated_input = tokenizer.decode(
                encoded_input["input_ids"][0][:max_length], 
                skip_special_tokens=True
            )
        else:
            truncated_input = text

        truncated_texts.append(truncated_input)
    return truncated_texts


def process_files(files: List[str], model_path: str, output_dir: str, gpu_id: int = 0,
                 max_token_length: int = 256):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        task="embed",
        enforce_eager=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    for input_path in tqdm(files, desc="calculating embeddings"):
        try:
            output_path = os.path.join(output_dir, os.path.basename(input_path)) # ensure embeddings correspond to input
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            text_list = load_data(input_path)

            truncated_prompts = truncate_texts(text_list, tokenizer, max_token_length)
            outputs = llm.embed(truncated_prompts)
            embeddings = [output.outputs.embedding for output in outputs]
            with open(output_path, 'w') as f:
                json.dump(embeddings, f, indent=4)
        except Exception as e:
            print(f"处理文件 {input_path} 时出错: {e}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="compute embedding for text files")
    
    parser.add_argument("--input_dir", nargs='+', required=True,
                       help="The input directories containing text files")
    parser.add_argument("--model_path", required=True,
                       help="The path to the model")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="gpu id (default: 0)")
    parser.add_argument("--max_length", type=int, default=256,
                       help="The max length for embedding (default: 256)")
    parser.add_argument("--output_dir", type=str,
                       help="The output directory")
    
    args = parser.parse_args()
    
    all_files = get_files_paths(args.input_dir)
    print(f"There are {len(all_files)} files in total")
    
    if not all_files:
        print("There are no files to process")
        return
        
    process_files(
        all_files, 
        args.model_path, 
        args.gpu_id, 
        args.max_length,
        args.output_dir
    )
    print("All files have been processed successfully")


if __name__ == "__main__":
    main()