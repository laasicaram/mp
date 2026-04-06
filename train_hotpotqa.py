import argparse
import math
import random
import re
import string
from pathlib import Path
from collections import Counter
from contextlib import nullcontext

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DistilBertModel,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

IGNORE_INDEX = -1


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def flatten_context(context: dict) -> str:
    sentences = []
    for sent_list in context["sentences"]:
        sentences.extend(sent_list)
    return " ".join(sentences)


def split_decay_params(model: nn.Module):
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    no_decay_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return decay_params, no_decay_params


def find_answer_span(context: str, answer: str):
    c = context.lower()
    a = answer.lower().strip()
    if not a:
        return -1, -1
    start = c.find(a)
    if start == -1:
        return -1, -1
    end = start + len(a)
    return start, end


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if len(p) == 0 and len(g) == 0:
        return 1.0
    if len(p) == 0 or len(g) == 0:
        return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        valid = targets != self.ignore_index
        if not valid.any():
            return ce.mean() * 0.0
        pt = torch.exp(-ce[valid])
        focal = ((1 - pt) ** self.gamma) * ce[valid]
        return focal.mean()


class BaselineQA(nn.Module):
    def __init__(self, num_answers: int):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_answers)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0])


class TinyReasoningCore(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, r, z_prev):
        x = torch.cat([q, r, z_prev], dim=-1)
        z = torch.relu(self.fc1(x))
        return self.fc2(z)


class TinyRecursiveQA(nn.Module):
    def __init__(self, num_answers: int, num_steps: int = 2):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_dim = self.encoder.config.hidden_size
        self.reasoning_core = TinyReasoningCore(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_answers)
        self.num_steps = num_steps

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_last_n_layers(self, n: int):
        if n <= 0:
            return
        layers = self.encoder.transformer.layer
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        q = out.last_hidden_state[:, 0]
        z = torch.zeros_like(q)
        for _ in range(self.num_steps):
            z = self.reasoning_core(q, q, z)
        return self.head(z)


def build_answer_vocab(ds, max_answers: int) -> dict:
    counter = Counter(ds["answer"])
    return {ans: idx for idx, (ans, _) in enumerate(counter.most_common(max_answers))}


def prepare_classification(args):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    train_full = ds["train"]
    val = ds["validation"]

    vocab_split = train_full
    if args.vocab_from_train_samples > 0:
        vocab_split = train_full.select(range(min(args.vocab_from_train_samples, len(train_full))))

    train = train_full
    if args.max_train_samples > 0:
        train = train.select(range(min(args.max_train_samples, len(train))))
    if args.max_val_samples > 0:
        val = val.select(range(min(args.max_val_samples, len(val))))

    answer2id = build_answer_vocab(vocab_split, args.num_answers)
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def prep(ex):
        enc = tok(
            ex["question"],
            flatten_context(ex["context"]),
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        enc["label"] = answer2id.get(ex["answer"], IGNORE_INDEX)
        return enc

    train = train.map(prep, remove_columns=train.column_names)
    val = val.map(prep, remove_columns=val.column_names)

    train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ignored_train = sum(int(y == IGNORE_INDEX) for y in train["label"])
    ignored_val = sum(int(y == IGNORE_INDEX) for y in val["label"])
    print(
        f"Ignored labels: train={ignored_train}/{len(train)} ({ignored_train/len(train):.2%}) | "
        f"val={ignored_val}/{len(val)} ({ignored_val/len(val):.2%})"
    )

    return train_loader, val_loader, answer2id


def build_classification_model(args, num_answers: int):
    if args.model == "baseline":
        model = BaselineQA(num_answers)
        lr = args.lr or 2e-5
    else:
        model = TinyRecursiveQA(num_answers, num_steps=args.num_steps)
        model.freeze_encoder()
        model.unfreeze_last_n_layers(args.unfreeze_last_n)
        lr = args.lr or 2e-4

    if args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, ignore_index=IGNORE_INDEX)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, label_smoothing=args.label_smoothing)

    decay_params, no_decay_params = split_decay_params(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    return model, criterion, optimizer, lr


def train_epoch_classification(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    grad_clip,
    grad_accum_steps,
    scaler,
    use_amp,
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    for step, batch in enumerate(tqdm(loader, desc="train_cls", leave=False), start=1):
        x = batch["input_ids"].to(device)
        m = batch["attention_mask"].to(device)
        y = batch["label"].to(device)

        with autocast_ctx:
            logits = model(x, m)
            loss = criterion(logits, y) / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % grad_accum_steps == 0 or step == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps

    return total_loss / max(1, len(loader))


@torch.no_grad()
def eval_classification(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="eval_cls", leave=False):
        x = batch["input_ids"].to(device)
        m = batch["attention_mask"].to(device)
        y = batch["label"].to(device)
        pred = model(x, m).argmax(dim=1)
        valid = y != IGNORE_INDEX
        if valid.any():
            correct += (pred[valid] == y[valid]).sum().item()
            total += valid.sum().item()
    return (correct / total) if total else 0.0


def prepare_extractive(args):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    train = ds["train"]
    val = ds["validation"]

    if args.max_train_samples > 0:
        train = train.select(range(min(args.max_train_samples, len(train))))
    if args.max_val_samples > 0:
        val = val.select(range(min(args.max_val_samples, len(val))))

    tokenizer = AutoTokenizer.from_pretrained(args.extractive_model_name, use_fast=True)

    def prep_train(batch):
        questions = [q.strip() for q in batch["question"]]
        contexts = [flatten_context(c) for c in batch["context"]]
        answers = batch["answer"]

        enc = tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=args.max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(enc["offset_mapping"]):
            c_start, c_end = find_answer_span(contexts[i], answers[i])
            if c_start == -1:
                start_positions.append(0)
                end_positions.append(0)
                continue

            sequence_ids = enc.sequence_ids(i)
            ctx_start = 0
            while ctx_start < len(sequence_ids) and sequence_ids[ctx_start] != 1:
                ctx_start += 1
            ctx_end = len(sequence_ids) - 1
            while ctx_end >= 0 and sequence_ids[ctx_end] != 1:
                ctx_end -= 1

            if ctx_start >= len(sequence_ids) or ctx_end < 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            if offsets[ctx_start][0] > c_start or offsets[ctx_end][1] < c_end:
                start_positions.append(0)
                end_positions.append(0)
                continue

            token_start = ctx_start
            while token_start <= ctx_end and offsets[token_start][0] <= c_start:
                token_start += 1
            token_start -= 1

            token_end = ctx_end
            while token_end >= ctx_start and offsets[token_end][1] >= c_end:
                token_end -= 1
            token_end += 1

            start_positions.append(token_start)
            end_positions.append(token_end)

        enc["start_positions"] = start_positions
        enc["end_positions"] = end_positions
        enc.pop("offset_mapping")
        return enc

    def prep_val(batch):
        questions = [q.strip() for q in batch["question"]]
        contexts = [flatten_context(c) for c in batch["context"]]
        answers = batch["answer"]

        enc = tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=args.max_length,
            padding="max_length",
        )
        enc["answer_text"] = answers
        return enc

    train = train.map(prep_train, batched=True, remove_columns=train.column_names)
    val = val.map(prep_val, batched=True, remove_columns=val.column_names)

    train.set_format(type="torch", columns=["input_ids", "attention_mask", "start_positions", "end_positions"])
    val.set_format(type="python", columns=["input_ids", "attention_mask", "answer_text"])

    def val_collate(features):
        return {
            "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
            "answer_text": [f["answer_text"] for f in features],
        }

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_collate,
    )
    return train_loader, val_loader, tokenizer


def train_epoch_extractive(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    grad_clip,
    grad_accum_steps,
    scaler,
    use_amp,
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    for step, batch in enumerate(tqdm(loader, desc="train_ext", leave=False), start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        with autocast_ctx:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            loss = out.loss / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % grad_accum_steps == 0 or step == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps

    return total_loss / max(1, len(loader))


@torch.no_grad()
def eval_extractive(model, loader, tokenizer, device):
    model.eval()
    exact_matches = 0
    f1_sum = 0.0
    total = 0

    for batch in tqdm(loader, desc="eval_ext", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answers = batch["answer_text"]

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        start_idx = out.start_logits.argmax(dim=1)
        end_idx = out.end_logits.argmax(dim=1)

        batch_n = min(input_ids.size(0), len(answers))
        for i in range(batch_n):
            s = int(start_idx[i].item())
            e = int(end_idx[i].item())
            if e < s:
                e = s

            token_ids = input_ids[i, s : e + 1].detach().cpu().tolist()
            pred = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            gold = str(answers[i]).strip()

            pred_n = normalize_text(pred)
            gold_n = normalize_text(gold)
            em = int(pred_n == gold_n)
            f1 = token_f1(pred, gold)

            exact_matches += em
            f1_sum += f1
            total += 1

    if total == 0:
        return 0.0, 0.0
    return exact_matches / total, f1_sum / total


def save_qa_model(model, tokenizer, output_dir: str):
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved model: {save_path}")


def run_extractive(args, device):
    print("Running span-based extractive QA (start/end prediction)")
    print(f"Model checkpoint: {args.extractive_model_name}")

    train_loader, val_loader, tokenizer = prepare_extractive(args)
    model = AutoModelForQuestionAnswering.from_pretrained(args.extractive_model_name).to(device)

    decay_params, no_decay_params = split_decay_params(model)
    lr = args.lr or 3e-5
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(args.warmup_ratio * total_updates)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_em = 0.0
    patience_count = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch_extractive(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            scaler=scaler,
            use_amp=use_amp,
        )
        em, f1 = eval_extractive(model, val_loader, tokenizer, device)
        print(f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} | EM={em:.4f} | F1={f1:.4f}")

        if em > best_em:
            best_em = em
            patience_count = 0
            save_qa_model(model, tokenizer, args.save_dir)
        else:
            patience_count += 1
            if patience_count >= args.early_stopping_patience:
                print("Early stopping triggered.")
                break

    print(f"Best EM: {best_em:.4f}")
    print(f"Final QA model directory: {args.save_dir}")


def run_classification(args, device):
    train_loader, val_loader, answer2id = prepare_classification(args)
    model, criterion, optimizer, lr = build_classification_model(args, num_answers=len(answer2id))
    model = model.to(device)
    print(f"Model={args.model} | num_answers={len(answer2id)} | lr={lr}")

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(args.warmup_ratio * total_updates)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc = 0.0
    patience_count = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch_classification(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_acc = eval_classification(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{args.epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.early_stopping_patience:
                print("Early stopping triggered.")
                break

    print(f"Best val_acc: {best_acc:.4f}")


def parse_args():
    ap = argparse.ArgumentParser(description="HotpotQA training (extractive QA + classification baselines)")
    ap.add_argument("--model", choices=["extractive_qa", "baseline", "tiny_recursive"], default="extractive_qa")

    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum-steps", type=int, default=2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--max-train-samples", type=int, default=20000)
    ap.add_argument("--max-val-samples", type=int, default=4000)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--num-answers", type=int, default=5000)
    ap.add_argument("--vocab-from-train-samples", type=int, default=-1)
    ap.add_argument("--num-steps", type=int, default=2)
    ap.add_argument("--unfreeze-last-n", type=int, default=2)
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--focal-gamma", type=float, default=2.0)

    ap.add_argument("--early-stopping-patience", type=int, default=2)
    ap.add_argument("--extractive-model-name", type=str, default="distilbert-base-uncased-distilled-squad")
    ap.add_argument("--save-dir", type=str, default="qa_model")
    return ap.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    if args.model == "extractive_qa":
        run_extractive(args, device)
    else:
        run_classification(args, device)


if __name__ == "__main__":
    main()
