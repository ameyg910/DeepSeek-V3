import argparse

import torch

from src.config import ModelConfig
from models.deepseek_v3 import DeepSeekV3
from data.tokenizer import MathTokenizer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(model_path, tokenizer_path, device):
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model_config = ModelConfig(**checkpoint["model_config"])

    model = DeepSeekV3(
        config=model_config,
        pad_token_id=checkpoint["pad_token_id"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    tokenizer = MathTokenizer.load(tokenizer_path)

    return model, tokenizer, model_config


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=100,
    max_seq_len=512,
    temperature=0.8,
    top_k=50,
):
    # Convert prompt text into token IDs.
    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    bos_token_id = tokenizer.token_id(tokenizer.BOS)
    eos_token_id = tokenizer.token_id(tokenizer.EOS)

    input_ids = [bos_token_id] + prompt_ids

    generated_ids = input_ids.copy()

    for _ in range(max_new_tokens):
        # Keep context within the model's maximum sequence length.
        context_ids = generated_ids[-max_seq_len:]

        input_tensor = torch.tensor(
            [context_ids],
            dtype=torch.long,
            device=device,
        )

        # Model output: (batch_size, sequence_length, vocab_size)
        logits = model(input_tensor)

        # Select logits for the final token position.
        next_token_logits = logits[0, -1, :]

        # Greedy decoding.
        if temperature <= 0:
            next_token = torch.argmax(next_token_logits)

        else:
            next_token_logits = next_token_logits / temperature

            # Top-k sampling.
            if top_k > 0:
                k = min(top_k, next_token_logits.size(-1))

                values, _ = torch.topk(next_token_logits, k)
                threshold = values[-1]

                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold,
                    float("-inf"),
                )

            probabilities = torch.softmax(
                next_token_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            ).squeeze(0)

        next_token_id = next_token.item()
        generated_ids.append(next_token_id)

        if next_token_id == eos_token_id:
            break

    # Decode only the generated continuation.
    continuation_ids = generated_ids[len(input_ids):]

    return tokenizer.decode(continuation_ids)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="model_bundle/model.pt",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="model_bundle/tokenizer.json",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="The gradient of a function",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model, tokenizer, config = load_model(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        device=device,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Prompt: {args.prompt}")
    print("\nGenerated text:\n")

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=config.max_seq_len,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(output)


if __name__ == "__main__":
    main()