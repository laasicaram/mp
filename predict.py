from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass
from itertools import combinations

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to", "of",
    "in", "on", "at", "for", "from", "with", "and", "or", "by", "that", "this", "it",
    "its", "as", "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "part", "company", "group", "family", "city", "state", "country", "name"
}


@dataclass
class Article:
    title: str
    sentences: list[str]
    body: str
    title_terms: set[str]
    body_terms: list[str]


@dataclass
class CandidateContext:
    text: str
    article_titles: list[str]
    retrieval_score: int


@dataclass
class AnswerResult:
    answer: str
    parsed_context: str
    score: float
    article_titles: list[str]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_for_overlap(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def content_terms(text: str) -> list[str]:
    return [t for t in normalize_for_overlap(text) if t not in STOPWORDS]


def unique_terms(text: str) -> set[str]:
    return set(content_terms(text))


def score_overlap(question_terms: set[str], text: str) -> int:
    tokens = content_terms(text)
    if not question_terms:
        return 0
    overlap = sum(1 for token in tokens if token in question_terms)
    unique_overlap = len(set(tokens) & question_terms)
    return (4 * unique_overlap) + overlap


def parse_hotpot_articles(raw_context: str) -> list[Article] | None:
    raw_context = raw_context.strip()
    if not raw_context:
        return None

    try:
        payload = json.loads(raw_context)
    except json.JSONDecodeError:
        return None

    titles = payload.get("title")
    sentences = payload.get("sentences")
    if not isinstance(titles, list) or not isinstance(sentences, list):
        return None

    articles = []
    for title, sentence_group in zip(titles, sentences):
        clean_title = str(title).strip()
        clean_sentences = [str(s).strip() for s in sentence_group if str(s).strip()]
        if not clean_title or not clean_sentences:
            continue
        body = " ".join(clean_sentences)
        articles.append(
            Article(
                title=clean_title,
                sentences=clean_sentences,
                body=body,
                title_terms=unique_terms(clean_title),
                body_terms=content_terms(clean_title + " " + body),
            )
        )
    return articles or None


def article_link_bonus(source: Article, target: Article, question_terms: set[str]) -> int:
    bonus = 0
    source_text = source.body.lower()
    target_title = target.title.lower()
    if target_title in source_text:
        bonus += 6
    if source.title.lower() in target.body.lower():
        bonus += 6
    shared_titles = source.title_terms & target.title_terms
    bonus += 2 * len(shared_titles - STOPWORDS)
    bonus += len(question_terms & target.title_terms)
    return bonus


def select_relevant_sentences(article: Article, question_terms: set[str], max_sentences: int = 4) -> str:
    scored = []
    for idx, sentence in enumerate(article.sentences):
        score = score_overlap(question_terms, sentence)
        if idx == 0:
            score += 1
        scored.append((score, idx, sentence))

    if question_terms:
        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen = sorted(scored[: min(max_sentences, len(scored))], key=lambda item: item[1])
    else:
        chosen = [(0, idx, sentence) for idx, sentence in enumerate(article.sentences[:max_sentences])]
    return " ".join(sentence for _, _, sentence in chosen)


def build_candidate_contexts(raw_context: str, question: str) -> list[CandidateContext]:
    articles = parse_hotpot_articles(raw_context)
    if not articles:
        return [CandidateContext(text=raw_context.strip(), article_titles=[], retrieval_score=0)]

    question_terms = set(content_terms(question))
    scored_articles = []
    for idx, article in enumerate(articles):
        title_bonus = 5 * len(question_terms & article.title_terms)
        body_score = score_overlap(question_terms, article.title + " " + article.body)
        scored_articles.append((body_score + title_bonus, idx, article))

    scored_articles.sort(key=lambda item: (-item[0], item[2].title))
    seed_score, _, seed_article = scored_articles[0]

    ranked = [(seed_score, seed_article)]
    for base_score, _, article in scored_articles[1:]:
        linked_score = base_score + article_link_bonus(seed_article, article, question_terms)
        ranked.append((linked_score, article))
    ranked.sort(key=lambda item: (-item[0], item[1].title))

    selected = [article for score, article in ranked if score > 0][:3]
    if not selected:
        selected = [seed_article]

    candidates = []
    for article in selected:
        body = select_relevant_sentences(article, question_terms)
        text = f"{article.title}: {body}"
        retrieval_score = score_overlap(question_terms, text) + 5 * len(question_terms & article.title_terms)
        candidates.append(CandidateContext(text=text, article_titles=[article.title], retrieval_score=retrieval_score))

    for combo in combinations(selected[:3], 2):
        parts = []
        retrieval_score = 0
        titles = []
        for article in combo:
            body = select_relevant_sentences(article, question_terms, max_sentences=3)
            parts.append(f"{article.title}: {body}")
            retrieval_score += score_overlap(question_terms, article.title + " " + body)
            titles.append(article.title)
        candidates.append(CandidateContext(text="\n\n".join(parts), article_titles=titles, retrieval_score=retrieval_score))

    unique = []
    seen = set()
    for candidate in candidates:
        if candidate.text in seen:
            continue
        seen.add(candidate.text)
        unique.append(candidate)
    return unique


def extract_best_span(start_logits, end_logits, sequence_ids, max_answer_len: int = 12):
    best_score = float("-inf")
    best_span = (0, 0)
    context_indexes = [i for i, sid in enumerate(sequence_ids) if sid == 1]
    if not context_indexes:
        return best_span, best_score

    for start in context_indexes:
        max_end = min(start + max_answer_len - 1, context_indexes[-1])
        for end in range(start, max_end + 1):
            if sequence_ids[end] != 1:
                continue
            score = start_logits[start] + end_logits[end]
            if score > best_score:
                best_score = score
                best_span = (start, end)
    return best_span, best_score


def run_reader(model, tokenizer, question: str, context: str, device: torch.device, max_length: int):
    encoded = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    sequence_ids = encoded.sequence_ids(0)
    offset_mapping = encoded.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0].detach().cpu().tolist()
    end_logits = outputs.end_logits[0].detach().cpu().tolist()
    (start_idx, end_idx), score = extract_best_span(start_logits, end_logits, sequence_ids)

    if sequence_ids[start_idx] != 1 or sequence_ids[end_idx] != 1:
        return "", score

    char_start = offset_mapping[start_idx][0]
    char_end = offset_mapping[end_idx][1]
    if char_end <= char_start:
        return "", score

    answer = context[char_start:char_end].strip()
    return answer, score


def answer_question(model, tokenizer, context: str, question: str, device: torch.device, max_length: int = 384) -> AnswerResult:
    candidate_contexts = build_candidate_contexts(context, question)
    best = AnswerResult(answer="", parsed_context="", score=float("-inf"), article_titles=[])

    for candidate in candidate_contexts:
        answer, reader_score = run_reader(model, tokenizer, question, candidate.text, device, max_length)
        combined_score = reader_score + candidate.retrieval_score
        if answer and combined_score > best.score:
            best = AnswerResult(
                answer=answer,
                parsed_context=candidate.text,
                score=combined_score,
                article_titles=candidate.article_titles,
            )

    if best.score == float("-inf"):
        fallback = candidate_contexts[0]
        return AnswerResult(answer="", parsed_context=fallback.text, score=float("-inf"), article_titles=fallback.article_titles)
    return best


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
    parser.add_argument("--show-parsed-context", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir).to(device)
    model.eval()

    if args.interactive:
        context = args.context or prompt_multiline("Enter context text or HotpotQA JSON:")
        if not context:
            raise SystemExit("Context is required for interactive mode.")

        print("Interactive QA ready. Type 'exit' to quit, 'context' to replace the context.")
        while True:
            question = input("Question> ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if question.lower() == "context":
                context = prompt_multiline("Enter new context text or HotpotQA JSON:")
                if not context:
                    print("Context unchanged.")
                continue
            if not question:
                continue

            result = answer_question(model, tokenizer, context, question, device, args.max_length)
            if args.show_parsed_context:
                print("Parsed context:")
                print(result.parsed_context)
                if result.article_titles:
                    print(f"Selected articles: {', '.join(result.article_titles)}")
            print(f"Answer: {result.answer or '[no confident span found]'}")
        return

    if not args.context or not args.question:
        raise SystemExit("Non-interactive mode requires --context and --question.")

    result = answer_question(model, tokenizer, args.context, args.question, device, args.max_length)
    if args.show_parsed_context:
        print("Parsed context:")
        print(result.parsed_context)
        if result.article_titles:
            print(f"Selected articles: {', '.join(result.article_titles)}")
    print(f"Question: {args.question}")
    print(f"Answer: {result.answer or '[no confident span found]'}")


if __name__ == "__main__":
    main()
