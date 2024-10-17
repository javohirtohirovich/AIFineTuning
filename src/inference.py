from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TextStreamer

# Load your model and tokenizer
model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"  # Model nomi
model, tokenizer = FastLanguageModel.from_pretrained("outputs/checkpoint-60")  # Fine-tuned checkpoint


# Enable faster inference
FastLanguageModel.for_inference(model)

# Define the alpaca prompt
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

# Prepare inputs
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Continue the fibonnaci sequence.",  # instruction
            "1, 1, 2, 3, 5, 8",  # input
            "",  # output - leave this blank for generation!
        )
    ],
    return_tensors="pt"
).to("cuda")

# Initialize TextStreamer
text_streamer = TextStreamer(tokenizer)

# Generate output with streaming
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
