/**
 * Gate M2(a): the browser tokenizer must reproduce the reference tokenizer exactly.
 *
 * Golden pairs come from tools/tokenize_golden.py, which runs the `tokenizers`
 * implementation shipped with the model. "Close enough" is not a category here -- a
 * single differing id changes what the model sees.
 */

import { beforeAll, describe, expect, it } from 'vitest';

import {
  BpeTokenizer,
  TokenizerError,
  translatePretokenizerRegex,
} from '../src/tokenizer/bpe.js';
import {
  ChatTemplate,
  ChatTemplateError,
  loadChatTemplate,
} from '../src/tokenizer/chat_template.js';

import golden from './fixtures/tokenizer-golden.json';

const TOKENIZER_URL = new URL(
  '/models/qwen2.5-0.5b-instruct/tokenizer.json',
  location.href,
).href;

let tokenizer: BpeTokenizer | null = null;

beforeAll(async () => {
  const head = await fetch(TOKENIZER_URL, { method: 'HEAD' });
  if (!head.ok) {
    console.warn(`[skip] ${TOKENIZER_URL} not found; run tools/convert.py to enable`);
    return;
  }
  tokenizer = await BpeTokenizer.fromUrl(TOKENIZER_URL);
});

describe('pretokenizer regex translation', () => {
  it('expands an inline case-insensitive group into character classes', () => {
    const re = translatePretokenizerRegex("(?i:'s|'ll)|\\p{L}+");
    expect(re.source).toBe("(?:'[sS]|'[lL][lL])|\\p{L}+");
    expect(re.flags).toContain('u');
    expect("IT'S well".match(re)).toEqual(['IT', "'S", 'well']);
  });

  it('leaves a pattern without inline flags untouched', () => {
    expect(translatePretokenizerRegex('\\p{N}| ?\\p{L}+').source).toBe('\\p{N}| ?\\p{L}+');
  });

  it('refuses to guess at a group it cannot case-fold', () => {
    // Silently mistranslating the pretokenizer would corrupt every token downstream.
    expect(() => translatePretokenizerRegex('(?i:\\p{L}+)')).toThrow(TokenizerError);
  });

  it('rejects an unbalanced group', () => {
    expect(() => translatePretokenizerRegex("(?i:'s")).toThrow(/unbalanced/);
  });
});

describe('tokenizer vs Python golden pairs', () => {
  it('has the golden fixture', () => {
    expect(golden.pairs.length).toBeGreaterThan(40);
  });

  it('encodes every golden pair to identical ids', () => {
    if (!tokenizer) return;
    const failures: string[] = [];
    for (const pair of golden.pairs) {
      const ids = tokenizer.encode(pair.text);
      if (ids.length !== pair.ids.length || ids.some((v, i) => v !== pair.ids[i])) {
        failures.push(
          `  ${pair.name}: ${JSON.stringify(pair.text)}\n` +
            `    expected ${JSON.stringify(pair.ids)}\n` +
            `    got      ${JSON.stringify(ids)}`,
        );
      }
    }
    if (failures.length) {
      throw new Error(
        `${failures.length}/${golden.pairs.length} golden pairs mismatched:\n` +
          failures.join('\n'),
      );
    }
    console.log(
      `tokenizer: ${golden.pairs.length}/${golden.pairs.length} golden pairs match exactly ` +
        `(${golden.pairs.reduce((n, p) => n + p.ids.length, 0)} ids)`,
    );
  });

  it('decodes every golden pair back to the reference decoding', () => {
    if (!tokenizer) return;
    for (const pair of golden.pairs) {
      // Compared against the reference *decoding*, not the input: NFC folding means a
      // decomposed input legitimately does not survive a round trip.
      expect(tokenizer.decode(pair.ids), pair.name).toBe(pair.decoded);
    }
  });

  it('round-trips its own encoding for every pair', () => {
    if (!tokenizer) return;
    for (const pair of golden.pairs) {
      expect(tokenizer.decode(tokenizer.encode(pair.text)), pair.name).toBe(pair.decoded);
    }
  });

  it('keeps multi-byte characters intact when they span two tokens', () => {
    if (!tokenizer) return;
    // Decoding token by token would emit replacement characters here; the decoder has
    // to accumulate bytes across tokens before running TextDecoder.
    const text = '\u4f60\u597d\u{1f30d}\u00e9';
    const ids = tokenizer.encode(text);
    expect(ids.length).toBeGreaterThan(1);
    expect(tokenizer.decode(ids)).toBe(text);
  });

  it('encodes the chat template exactly', () => {
    if (!tokenizer) return;
    expect(tokenizer.encode(golden.chat.rendered)).toEqual(golden.chat.ids);
  });

  it('treats special tokens as single ids and lookalikes as ordinary text', () => {
    if (!tokenizer) return;
    expect(tokenizer.encode('<|im_start|>')).toEqual([151644]);
    expect(tokenizer.encode('<|not_a_real_token|>').length).toBeGreaterThan(1);
    expect(tokenizer.isSpecial(151644)).toBe(true);
  });

  it('can be told not to parse special tokens', () => {
    if (!tokenizer) return;
    const asText = tokenizer.encode('<|im_start|>', { parseSpecial: false });
    expect(asText.length).toBeGreaterThan(1);
    expect(tokenizer.decode(asText)).toBe('<|im_start|>');
  });

  it('skips special tokens on request when decoding', () => {
    if (!tokenizer) return;
    const ids = tokenizer.encode('<|im_start|>user\nHello<|im_end|>');
    expect(tokenizer.decode(ids, { skipSpecialTokens: true })).toBe('user\nHello');
  });
});

describe('chat template', () => {
  let template: ChatTemplate | null = null;

  beforeAll(async () => {
    const url = new URL('/models/qwen2.5-0.5b-instruct/tokenizer_config.json', location.href).href;
    if (!(await fetch(url, { method: 'HEAD' })).ok) return;
    template = (await loadChatTemplate(url)).template;
  });

  it('reproduces every golden rendering exactly', () => {
    if (!template) return;
    for (const testCase of golden.chatCases) {
      const rendered = template.render(testCase.messages, {
        addGenerationPrompt: testCase.addGenerationPrompt,
      });
      if (rendered !== testCase.rendered) {
        throw new Error(
          `${testCase.name} mismatch:\n  expected ${JSON.stringify(testCase.rendered)}\n` +
            `  got      ${JSON.stringify(rendered)}`,
        );
      }
    }
    console.log(
      `chat template: ${golden.chatCases.length}/${golden.chatCases.length} renderings exact`,
    );
  });

  it('injects the model default system prompt when none is supplied', () => {
    if (!template) return;
    // Easy to miss and it changes every token that follows, so it is pinned explicitly.
    const rendered = template.render([{ role: 'user', content: 'Hello' }], {
      addGenerationPrompt: true,
    });
    expect(rendered).toContain('You are Qwen, created by Alibaba Cloud.');
    expect(rendered.endsWith('<|im_start|>assistant\n')).toBe(true);
  });

  it('omits the generation prompt when not asked for', () => {
    if (!template) return;
    const rendered = template.render([{ role: 'user', content: 'Hello' }], {
      addGenerationPrompt: false,
    });
    expect(rendered.endsWith('<|im_start|>assistant\n')).toBe(false);
  });

  it('tokenizes its own rendering to the golden ids', () => {
    if (!template || !tokenizer) return;
    for (const testCase of golden.chatCases) {
      const rendered = template.render(testCase.messages, {
        addGenerationPrompt: testCase.addGenerationPrompt,
      });
      expect(tokenizer.encode(rendered), testCase.name).toEqual(testCase.ids);
    }
  });

  it('evaluates the expression subset it claims to support', () => {
    const cases: Array<[string, string]> = [
      ["{{ 'a' + 'b' }}", 'ab'],
      ['{% if 1 == 1 %}yes{% else %}no{% endif %}', 'yes'],
      ['{% if 1 != 1 %}yes{% elif 2 > 1 %}elif{% else %}no{% endif %}', 'elif'],
      ['{% for x in items %}{{ x }}{% if not loop.last %},{% endif %}{% endfor %}', '1,2,3'],
      ['{% for x in items %}{{ loop.index0 }}{% endfor %}', '012'],
      ['{% set y = 5 %}{{ y }}', '5'],
      ['{{ missing is defined }}', 'False'],
      ['{{ items is defined }}', 'True'],
      ['{{ obj.name }}', 'qwen'],
      ["{{ obj['name'] }}", 'qwen'],
      ['{{ items | length }}', '3'],
      ["{{ ' pad ' | trim }}", 'pad'],
      ['{{ obj | tojson }}', '{"name":"qwen"}'],
      ["{% if 'a' in letters %}in{% endif %}", 'in'],
    ];
    for (const [source, expected] of cases) {
      const result = new ChatTemplate(source).render([], {
        extra: { items: [1, 2, 3], obj: { name: 'qwen' }, letters: ['a', 'b'] },
      });
      expect(result, source).toBe(expected);
    }
  });

  it('refuses a construct it does not implement rather than rendering it wrong', () => {
    // A chat template that renders almost-right produces a prompt the model was never
    // trained on, and that is very hard to notice downstream.
    expect(() => new ChatTemplate('{% macro f() %}{% endmacro %}')).toThrow(ChatTemplateError);
    expect(() => new ChatTemplate('{{ x | nosuchfilter }}').render([])).toThrow(ChatTemplateError);
    expect(() => new ChatTemplate('{{ x is weird }}').render([])).toThrow(ChatTemplateError);
  });
});
