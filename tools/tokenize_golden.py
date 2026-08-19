#!/usr/bin/env python3
"""
Dump (text -> token ids) golden pairs for the browser tokenizer to be tested against.

Byte-level BPE is easy to get subtly wrong -- a leading space, an NFC normalisation, a
digit run, or an emoji built from a ZWJ sequence will each silently produce different
ids without producing an obviously broken result. The cases below are chosen to make
each of those failures visible.

The reference is the `tokenizers` implementation that ships with the model, used here
purely as an oracle. It is a dev-time dependency in .venv and is never imported by src/.

Usage:
    .venv/bin/python tools/tokenize_golden.py models/Qwen2.5-0.5B-Instruct \\
        --out tests/fixtures/tokenizer-golden.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

CASES: list[tuple[str, str]] = [
    ("empty", ""),
    ("single-space", " "),
    ("ascii-word", "hello"),
    ("leading-space", " hello"),
    ("trailing-space", "hello "),
    ("sentence", "The quick brown fox jumps over the lazy dog."),
    ("leading-space-sentence", " The quick brown fox."),
    ("double-space", "hello  world"),
    ("many-spaces", "a          b"),
    ("tabs", "def f():\n\tif x:\n\t\treturn 1\n"),
    ("newlines", "line one\nline two\n\nline four"),
    ("crlf", "windows\r\nline\r\n"),
    ("trailing-newlines", "text\n\n\n"),
    ("only-whitespace", "   \n\t  "),
    # The pretokenizer splits digits one at a time on this model; a wrong `\\p{N}`
    # translation shows up here immediately.
    ("digits", "1234567890"),
    ("number-in-text", "I have 42 apples and 3.14159 pies"),
    ("contraction-lower", "don't stop, it's fine, we've won, I'll go, he'd know"),
    # The split regex uses an inline case-insensitive group, which JS cannot express
    # directly and has to be rewritten by hand.
    ("contraction-upper", "DON'T STOP, IT'S FINE, WE'VE WON, I'LL GO, HE'D KNOW"),
    ("contraction-mixed", "Don'T sToP iT'S fInE"),
    ("punctuation-run", "!!!???...---***"),
    ("symbols", "a+b=c & d|e ^ f%g #h @i"),
    ("quotes", "\u201ccurly\u201d and 'straight' and \u00abguillemets\u00bb"),
    ("cjk-chinese", "\u4f60\u597d\u4e16\u754c"),
    ("cjk-japanese", "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059"),
    ("cjk-korean", "\uc548\ub155\ud558\uc138\uc694 \uc138\uacc4"),
    ("cjk-mixed", "Hello \u4e16\u754c, this is \u6df7\u5408 text"),
    ("emoji-simple", "\U0001f44b\U0001f30d"),
    ("emoji-zwj", "\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466"),
    ("emoji-in-text", "great work \U0001f389 well done \U0001f44d\U0001f3fd"),
    ("emoji-flags", "\U0001f1ec\U0001f1e7 \U0001f1ef\U0001f1f5 \U0001f1fa\U0001f1f8"),
    ("accents", "caf\u00e9 na\u00efve r\u00e9sum\u00e9 Stra\u00dfe"),
    # Same visible text as `accents` but decomposed: NFC normalisation must fold them
    # to the identical id sequence.
    ("accents-decomposed", "cafe\u0301 nai\u0308ve re\u0301sume\u0301 Stra\u00dfe"),
    ("cyrillic", "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440"),
    ("arabic", "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"),
    ("hebrew", "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd"),
    ("code", "const x = arr.map((v) => v * 2).filter(Boolean);"),
    ("json", '{"key": "value", "n": [1, 2, 3], "ok": true}'),
    ("url", "https://example.com/path?a=1&b=2#frag"),
    ("markdown", "# Heading\n\n- item one\n- item two\n\n```js\ncode();\n```\n"),
    ("repeated", "ababababababab"),
    ("long-word", "pneumonoultramicroscopicsilicovolcanoconiosis"),
    ("mixed-case", "CamelCase snake_case SCREAMING_SNAKE kebab-case"),
    ("control-chars", "a\u0000b\u0001c"),
    ("nbsp", "a\u00a0b"),
    ("zero-width", "a\u200bb\u200cc"),
    ("surrogate-pair-math", "\U0001d539\U0001d552\U0001d556"),
]

# Special tokens must survive as single ids rather than being byte-encoded.
SPECIAL_CASES: list[tuple[str, str]] = [
    ("special-endoftext", "<|endoftext|>"),
    ("special-im-start", "<|im_start|>"),
    ("special-chat-turn", "<|im_start|>user\nHello<|im_end|>\n"),
    ("special-embedded", "before <|im_end|> after"),
    ("special-adjacent", "<|im_start|><|im_end|>"),
    ("special-lookalike", "<|not_a_real_token|>"),
]

CHAT_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "4."},
    {"role": "user", "content": "Why?"},
]

# Conversation shapes the template renderer has to reproduce exactly. The no-system case
# matters most: this template injects a default system prompt when none is supplied, which
# is easy to miss and changes every token that follows.
CHAT_CASES: list[tuple[str, list[dict], bool]] = [
    ("single-user", [{"role": "user", "content": "Hello"}], True),
    ("single-user-no-gen", [{"role": "user", "content": "Hello"}], False),
    (
        "with-system",
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Define entropy."},
        ],
        True,
    ),
    (
        "multi-turn",
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4."},
            {"role": "user", "content": "Why?"},
        ],
        True,
    ),
    (
        "multi-turn-no-system",
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "Tell me a joke."},
        ],
        True,
    ),
    (
        "unicode-content",
        [{"role": "user", "content": "Translate 你好 and café \U0001f30d"}],
        True,
    ),
    (
        "multiline-content",
        [{"role": "user", "content": "line one\nline two\n\nline four"}],
        True,
    ),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(str(args.model_dir))

    pairs = []
    for name, text in CASES + SPECIAL_CASES:
        ids = tok.encode(text, add_special_tokens=False)
        pairs.append(
            {
                "name": name,
                "text": text,
                "ids": ids,
                # Decoding without skipping special tokens is the true round trip.
                "decoded": tok.decode(ids, skip_special_tokens=False),
            }
        )

    rendered = tok.apply_chat_template(CHAT_MESSAGES, tokenize=False, add_generation_prompt=True)
    chat = {
        "messages": CHAT_MESSAGES,
        "rendered": rendered,
        "ids": tok.encode(rendered, add_special_tokens=False),
    }

    chat_cases = []
    for name, messages, add_generation in CHAT_CASES:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation
        )
        chat_cases.append(
            {
                "name": name,
                "messages": messages,
                "addGenerationPrompt": add_generation,
                "rendered": text,
                "ids": tok.encode(text, add_special_tokens=False),
            }
        )

    fixture = {
        "modelDir": str(args.model_dir),
        "pairs": pairs,
        "chat": chat,
        "chatCases": chat_cases,
        "chatTemplate": tok.chat_template,
        "specialTokens": {
            "bos": tok.bos_token_id,
            "eos": tok.eos_token_id,
            "pad": tok.pad_token_id,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")

    total_ids = sum(len(p["ids"]) for p in pairs)
    print(f"wrote {len(pairs)} golden pairs ({total_ids} ids total) to {args.out}")
    print(f"  plus {len(chat_cases)} chat-template renderings")
    mismatched = [p["name"] for p in pairs if p["decoded"] != p["text"]]
    if mismatched:
        # Not an error: some inputs genuinely do not survive a round trip (NFC folding,
        # lone control characters). Recorded so the browser test expects the same thing.
        print(f"  note: {len(mismatched)} case(s) do not decode back to the input: {mismatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
