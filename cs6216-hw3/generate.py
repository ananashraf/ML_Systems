import time
import itertools
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import torch
from transformers import AutoTokenizer

from model import PythiaForCausalLM, KVCache

default_device = 'cpu'


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


def _get_model_size(model):
    model_size = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size


def top_k_(logits, top_k=50, filter_value=-float("Inf")):
    # (TODO) Task 7 - Implement top-k sampling
    # Filter logits to only keep the top-k values
    # For the top-k values, keep them as they are, set the rest to filter_value
    pass


def top_p_(logits, top_p=0.9, min_tokens_to_keep=1, filter_value=-float("Inf")):
    # (TODO) Task 7 - Implement top-p sampling
    # The function should filter out logits such that the cumulative probability exceeds `top_p`.
    # The function should keep at least `min_tokens_to_keep` tokens.
    # Hint: softmax the logits and calculate the cumulative probabilities.
    pass


def logits_to_probs(
    logits: torch.Tensor, 
    temperature: float = 1.0, 
    top_k: Optional[int] = None, 
    top_p: Optional[float] = None
) -> torch.Tensor:
    logits = logits / temperature

    if top_k is not None and top_p is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time")

    if top_k is not None:
        logits = top_k_(logits, top_k)

    if top_p is not None:
        logits = top_p_(logits, top_p)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def stop_on_eos(token: torch.Tensor) -> bool:
    # (TODO) Task 6 - Implement stopping on EOS token
    # Return True if the token is an EOS token, False otherwise
    # Hint: EOS token you can find in the tokenizer
    pass


def prefill(
    model: torch.nn.Module,
    x: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0,
    generator: torch.Generator = None,
) -> Tuple[torch.Tensor, Tuple[KVCache]]:
    # (TODO) Task 4 - Implement prefilling
    # Input:  x - the input tensor of shape [batch_size, seq_len]
    #         attention_mask - the attention mask tensor of shape [batch_size, seq_len] (all ones for this assignment)
    # Output: next_token - the next token tensor of shape [batch_size]
    #         kvcaches - the key-value caches to be used in the decoding step

    # Hint: use `model.forward` to get the logits
    # Hint: use `logits_to_probs` to get the probabilities from the logits
    # Hint: check CausalLMOutputWithPast for the return type of `model.forward`
    # # Hint: You should use `torch.multinomial` and pass the `generator`, so we can reproduce the results
    # outputs = model(input_ids=x, attention_mask=attention_mask)
    
    # # Step 2: Get the logits from the last token
    # logits = outputs.logits[:, -1, :]
    
    # # Step 3: Convert logits to probabilities
    # probs = logits_to_probs(logits, temperature, top_k, top_p)
    
    # # Step 4: Sample the next token
    # next_token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    
    # # Step 5: Return the next token and the updated KV caches
    # return next_token, outputs.kvcaches
    # Step 1: Forward pass through the model

    # outputs = model(input_ids=x, attention_mask=attention_mask)
    
    # # Step 2: Get the logits from the last token
    # logits = outputs.logits[:, -1, :]
    
    # # Step 3: Convert logits to probabilities
    # probs = logits_to_probs(logits, temperature, top_k, top_p)
    
    # # Step 4: Sample the next token
    # next_token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    
    # # Step 5: Ensure next_token is a scalar
    # next_token = next_token.squeeze()
    
    # # Step 6: Return the next token and the updated KV caches
    # return next_token, outputs.kvcaches

    # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # Make sure the attention mask has the correct shape

    # Step 1: Forward pass through the model
    outputs = model(input_ids=x, attention_mask=attention_mask)

    # Step 2: Get the logits from the last token
    logits = outputs.logits[:, -1, :]

    # Step 3: Convert logits to probabilities
    probs = logits_to_probs(logits, temperature, top_k, top_p)

    # Step 4: Sample the next token
    next_token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    # Step 5: Ensure next_token is a scalar
    next_token = next_token.squeeze()

    # Step 6: Return the next token and the updated KV caches
    return next_token, outputs.kvcaches


    # outputs = model(input_ids=x, attention_mask=attention_mask)
    # logits = outputs.logits[:, -1, :]

    # # Convert logits to probabilities
    # probs = logits_to_probs(logits, temperature, top_k, top_p)

    # # Sample the next token from the probability distribution
    # next_token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    # # Return the next token and the updated KV caches
    # return next_token, outputs.kvcaches


def decode(
    model: torch.nn.Module,
    kvcaches: Tuple[KVCache],
    cur_token: torch.Tensor,
    attention_mask: torch.Tensor,
    num_new_tokens: int,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0,
    generator: torch.Generator = None,
) -> List[torch.Tensor]:
    # (TODO) Task 5 - Implement decoding
    # Input:  kvcaches - the key-value caches to be used in the decoding step
    #         cur_token - the current token tensor of shape [batch_size]
    #         attention_mask - the attention mask tensor of shape [batch_size, seq_len] (all ones for this assignment)
    #         num_new_tokens - the number of new tokens to generate
    # Output: new_tokens - the list of new token tensors of shape [batch_size]

    # Hint: You should use `torch.multinomial` and pass the `generator`, so we can reproduce the results
    # new_tokens = []
    # batch_size = cur_token.size(0)

    # for _ in range(num_new_tokens):
    #     # Forward pass through the model to get logits
    #     input_ids = cur_token.view(batch_size, -1)  # Ensure input_ids has shape [batch_size, seq_len]
    #     outputs = model(input_ids=input_ids, attention_mask=attention_mask, kvcaches=kvcaches)
    #     logits = outputs.logits[:, -1, :]

    #     # Convert logits to probabilities
    #     probs = logits_to_probs(logits, temperature, top_k, top_p)

    #     # Use argmax to pick the highest probability token for more deterministic behavior
    #     next_token = torch.argmax(probs, dim=-1)

    #     # Append the new token to the list of new tokens
    #     new_tokens.append(next_token)

    #     # Update the current token
    #     cur_token = next_token

    #     # Update the key-value caches
    #     kvcaches = outputs.kvcaches

    #     # Adjust the attention mask to include the new token
    #     attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)

    #     # Break if the EOS token is generated
    #     if stop_on_eos(next_token):
    #         break

    # return new_tokens


    new_tokens = []
    batch_size = cur_token.size(0)

    for step in range(num_new_tokens):
        input_ids = cur_token.view(batch_size, -1)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, kvcaches=kvcaches)
        logits = outputs.logits[:, -1, :]

        # Debugging outputs
        print(f"Step {step}: Logits:", logits)
        print(f"Step {step}: KV Caches:", outputs.kvcaches)

        # Convert logits to probabilities
        probs = logits_to_probs(logits, temperature, top_k, top_p)
        print(f"Step {step}: Probabilities:", probs)

        # Use argmax to pick the highest probability token
        next_token = torch.argmax(probs, dim=-1)
        print(f"Step {step}: Next Token (Argmax):", next_token)

        new_tokens.append(next_token)
        cur_token = next_token
        kvcaches = outputs.kvcaches  # Update KV caches

        # Verify the shape and content of the attention mask
        print(f"Step {step}: Attention Mask Before Update: {attention_mask.shape}")
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
        print(f"Step {step}: Attention Mask After Update: {attention_mask.shape}")

        if stop_on_eos(next_token):
            break

    return new_tokens



    # (TODO) Task 6 - Implement stopping on EOS token
    # use `stop_on_eos` to check if the token is an EOS token, if so, break the loop

    


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    max_seq_length = T + max_new_tokens

    device, dtype = prompt.device, prompt.dtype

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(max_seq_length, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty

    next_token, kvcaches = prefill(model, prompt.view(
        1, -1), attention_mask.view(1, -1), top_k, top_p, temperature)
    next_token = next_token.clone()
    seq[T] = next_token

    generated_tokens = decode(model, kvcaches, next_token.view(
        1, -1), attention_mask.view(1, -1), max_new_tokens - 1, top_k, top_p, temperature)
    
    seq = seq[:T + 1 + len(generated_tokens)]
    seq[T + 1:] = torch.cat(generated_tokens)

    return seq


def main():

    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--prompt', type=str,
                        default="Hello, my name is", help='Input prompt.')
    parser.add_argument('--num_samples', type=int,
                        default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int,
                        default=300, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-k for sampling.')
    parser.add_argument('--top_p', type=float, default=None,
                        help='Top-p for sampling.')
    parser.add_argument('--temperature', type=float,
                        default=0.9, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path(
        "checkpoints/EleutherAI/pythia-410m/pytorch_model.bin"), help='Model checkpoint path.')
    parser.add_argument('--device', type=str,
                        default=default_device, help='Device to use')

    args = parser.parse_args()

    assert args.checkpoint_path.is_file(), args.checkpoint_path

    tokenizer_path = args.checkpoint_path.parent / "tokenizer.json"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    precision = torch.bfloat16
    print(f"\033[91m> Using device={args.device}, precision={precision}\033[0m")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path.parent, clean_up_tokenization_spaces=True)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    prompt_length = encoded['input_ids'][0].size(0)

    model = PythiaForCausalLM.from_pretrained(
        args.checkpoint_path.parent, torch_dtype=precision)
    model_size = _get_model_size(model)

    aggregate_metrics = {
        'tokens_per_sec': [],
    }

    for i in range(0, args.num_samples):
        device_sync(device=args.device)
        t0 = time.perf_counter()
        y = generate(model, encoded['input_ids'][0], encoded['attention_mask'][0], args.max_new_tokens,
                     top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)
        device_sync(device=args.device)  # MKG
        t = time.perf_counter() - t0
        print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(tokens_sec)
        print(
            f"\x1b[6;30;42m> Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec\x1b[0m")
        print(f"\x1b[6;30;42m> Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s\x1b[0m")

    print("\x1b[6;30;42m==========\x1b[0m")
    print(
        f"\x1b[6;30;42m> Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}\x1b[0m")


if __name__ == '__main__':
    main()
