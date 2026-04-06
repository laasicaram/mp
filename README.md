# Lightweight HotpotQA QA System

This project fine-tunes a lightweight question answering model on the HotpotQA dataset and provides an interactive terminal interface for asking questions from a given context.

## Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- DistilBERT-based extractive QA

## Project Files

- `train_hotpotqa.py`: trains the model
- `predict.py`: runs question answering from the terminal
- `requirements.txt`: Python dependencies

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the Model

```bash
./.venv/bin/python3 train_hotpotqa.py \
  --model extractive_qa \
  --epochs 4 \
  --batch-size 8 \
  --max-train-samples 20000 \
  --max-val-samples 4000 \
  --save-dir qa_model
```

This saves the best QA model to `qa_model/`.

## Interactive Question Answering

```bash
./.venv/bin/python3 predict.py --model-dir qa_model --interactive
```

Usage flow:
- paste the context text
- press Enter on a blank line to finish context input
- ask questions at `Question>`
- type `context` to replace the context
- type `exit` to quit

## One-Off Prediction

```bash
./.venv/bin/python3 predict.py \
  --model-dir qa_model \
  --context "The Eiffel Tower is located in Paris, France." \
  --question "Where is the Eiffel Tower located?"
```

## Notes

- The repository excludes the trained `qa_model/` directory from GitHub.
- Train locally first before running interactive prediction.
- The model answers questions by extracting an answer span from the provided context.
