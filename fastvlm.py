import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
from PIL import Image
import argparse


def predict(args):
    # Load the Fast VLM model
    model_path = args.model_path
    
    # Load model, tokenizer, and image processor
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Load and process the image
    image = Image.open(args.image_file).convert('RGB')
    
    # Prepare the prompt
    prompt = args.prompt
    
    # Process inputs
    inputs = image_processor(image, return_tensors="pt")
    text_inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **text_inputs,
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
        )
    
    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, 
                       default="~/.cache/huggingface/hub/models--apple--FastVLM-0.5B/snapshots/16375720c2d673fa583e57e9876afde27549c7d0")
    parser.add_argument("--image-file", type=str, required=True, help="Path to image file")
    parser.add_argument("--prompt", type=str, default="Describe the image.", help="Prompt for VLM")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    
    args = parser.parse_args()
    predict(args)