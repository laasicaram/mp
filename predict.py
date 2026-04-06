import argparse

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def answer_question(model, tokenizer, context: str, question: str, device: torch.device, max_length: int = 384) -> str:
    inputs = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = int(outputs.start_logits.argmax(dim=1).item())
    end_idx = int(outputs.end_logits.argmax(dim=1).item())
    if end_idx < start_idx:
        end_idx = start_idx

    token_ids = inputs["input_ids"][0, start_idx:end_idx + 1].detach().cpu().tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def prompt_multiline(label: str) -> str:
    print(label)
    print("Finish input with a blank line.")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Ask questions using the saved QA model")
    parser.add_argument("--model-dir", type=str, default="qa_model")
    parser.add_argument("--context", type=str)
    parser.add_argument("--question", type=str)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir).to(device)
    model.eval()

    if args.interactive:
        context = args.context or prompt_multiline("Enter context text:")
        if not context:
            raise SystemExit("Context is required for interactive mode.")

        print("Interactive QA ready. Type 'exit' to quit, 'context' to replace the context.")
        while True:
            question = input("Question> ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if question.lower() == "context":
                context = prompt_multiline("Enter new context text:")
                if not context:
                    print("Context unchanged.")
                continue
            if not question:
                continue

            answer = answer_question(model, tokenizer, context, question, device, args.max_length)
            print(f"Answer: {answer}")
        return

    if not args.context or not args.question:
        raise SystemExit("Non-interactive mode requires --context and --question.")

    answer = answer_question(model, tokenizer, args.context, args.question, device, args.max_length)
    print(f"Question: {args.question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
