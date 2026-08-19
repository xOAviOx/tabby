/**
 * A minimal Jinja2 subset, enough to render the chat templates Llama-family models ship.
 *
 * Hardcoding ChatML would work for Qwen and break on the next model, which defeats M6's
 * "swap the model, change no engine code" requirement. So the template is interpreted
 * from `tokenizer_config.json` instead.
 *
 * Supported: `{{ }}` output, `{% if/elif/else/endif %}`, `{% for/endfor %}` with
 * `loop.first/last/index/index0/length`, `{% set %}`, `{# #}` comments, whitespace
 * control (`{%-`, `-%}`), and Jinja's `trim_blocks`/`lstrip_blocks` behaviour, which is
 * what `transformers` enables. Expressions cover string/number/bool literals, `+ - * /`,
 * comparisons, `and/or/not`, `in`, `is defined/none/string`, attribute and index access,
 * and the `tojson/trim/lower/upper/length/list/string/first/last` filters.
 *
 * Not supported: macros, includes, inheritance, custom tests. Anything unrecognised
 * throws with the offending source, because a chat template that renders *almost* right
 * produces a prompt the model was never trained on and is very hard to notice.
 */

export class ChatTemplateError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ChatTemplateError';
  }
}

export type TemplateValue = unknown;
export type TemplateContext = Record<string, TemplateValue>;

// -----------------------------------------------------------------------------------
// lexer
// -----------------------------------------------------------------------------------

interface RawToken {
  kind: 'text' | 'output' | 'statement' | 'comment';
  value: string;
  stripLeft: boolean;
  stripRight: boolean;
}

function lex(source: string): RawToken[] {
  const tokens: RawToken[] = [];
  let index = 0;

  while (index < source.length) {
    const next = source.indexOf('{', index);
    if (next === -1) {
      tokens.push({ kind: 'text', value: source.slice(index), stripLeft: false, stripRight: false });
      break;
    }
    const marker = source[next + 1];
    if (marker !== '{' && marker !== '%' && marker !== '#') {
      // A bare brace, e.g. inside a JSON literal in the template text.
      const chunkEnd = source.indexOf('{', next + 1);
      const stop = chunkEnd === -1 ? source.length : chunkEnd;
      tokens.push({
        kind: 'text',
        value: source.slice(index, stop),
        stripLeft: false,
        stripRight: false,
      });
      index = stop;
      continue;
    }

    if (next > index) {
      tokens.push({
        kind: 'text',
        value: source.slice(index, next),
        stripLeft: false,
        stripRight: false,
      });
    }

    const close = marker === '{' ? '}}' : marker === '%' ? '%}' : '#}';
    const end = source.indexOf(close, next + 2);
    if (end === -1) throw new ChatTemplateError(`unterminated tag at offset ${next}`);

    let body = source.slice(next + 2, end);
    const stripLeft = body.startsWith('-');
    if (stripLeft) body = body.slice(1);
    const stripRight = body.endsWith('-');
    if (stripRight) body = body.slice(0, -1);

    tokens.push({
      kind: marker === '{' ? 'output' : marker === '%' ? 'statement' : 'comment',
      value: body.trim(),
      stripLeft,
      stripRight,
    });
    index = end + close.length;
  }

  return applyWhitespaceRules(tokens);
}

/**
 * `transformers` compiles templates with trim_blocks and lstrip_blocks enabled, so a
 * block tag on its own line contributes no whitespace of its own. Explicit `-` markers
 * strip more aggressively, across newlines.
 */
function applyWhitespaceRules(tokens: RawToken[]): RawToken[] {
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (token.kind === 'text') continue;

    const previous = tokens[i - 1];
    if (previous?.kind === 'text') {
      if (token.stripLeft) {
        previous.value = previous.value.replace(/\s+$/, '');
      } else if (token.kind !== 'output') {
        // lstrip_blocks: drop indentation between the line start and a block tag.
        previous.value = previous.value.replace(/[ \t]+$/, '');
      }
    }

    const following = tokens[i + 1];
    if (following?.kind === 'text') {
      if (token.stripRight) {
        following.value = following.value.replace(/^\s+/, '');
      } else if (token.kind !== 'output') {
        // trim_blocks: swallow the single newline right after a block tag.
        following.value = following.value.replace(/^\r?\n/, '');
      }
    }
  }
  return tokens.filter((token) => token.kind !== 'comment' && token.value !== '');
}

// -----------------------------------------------------------------------------------
// expressions
// -----------------------------------------------------------------------------------

type Expr =
  | { type: 'literal'; value: TemplateValue }
  | { type: 'name'; name: string }
  | { type: 'attr'; target: Expr; name: string }
  | { type: 'index'; target: Expr; index: Expr }
  | { type: 'unary'; op: 'not' | '-'; operand: Expr }
  | { type: 'binary'; op: string; left: Expr; right: Expr }
  | { type: 'filter'; name: string; target: Expr }
  | { type: 'test'; target: Expr; name: string; negated: boolean };

const TOKEN_PATTERN =
  /\s*(?:(\d+\.\d+|\d+)|'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)"|(\w+)|(==|!=|<=|>=|[+\-*/<>()[\].,|]))/y;

interface ExprToken {
  kind: 'number' | 'string' | 'word' | 'punct';
  value: string;
}

function tokenizeExpression(source: string): ExprToken[] {
  const tokens: ExprToken[] = [];
  TOKEN_PATTERN.lastIndex = 0;
  while (TOKEN_PATTERN.lastIndex < source.length) {
    const start = TOKEN_PATTERN.lastIndex;
    const match = TOKEN_PATTERN.exec(source);
    if (!match) {
      if (source.slice(start).trim() === '') break;
      throw new ChatTemplateError(`cannot parse expression near "${source.slice(start, start + 24)}"`);
    }
    if (match[1] !== undefined) tokens.push({ kind: 'number', value: match[1] });
    else if (match[2] !== undefined) tokens.push({ kind: 'string', value: unescape(match[2]) });
    else if (match[3] !== undefined) tokens.push({ kind: 'string', value: unescape(match[3]) });
    else if (match[4] !== undefined) tokens.push({ kind: 'word', value: match[4] });
    else tokens.push({ kind: 'punct', value: match[5]! });
  }
  return tokens;
}

function unescape(text: string): string {
  return text.replace(/\\(.)/g, (_, char: string) => {
    if (char === 'n') return '\n';
    if (char === 't') return '\t';
    if (char === 'r') return '\r';
    return char;
  });
}

class ExpressionParser {
  private position = 0;

  constructor(
    private readonly tokens: ExprToken[],
    private readonly source: string,
  ) {}

  static parse(source: string): Expr {
    const parser = new ExpressionParser(tokenizeExpression(source), source);
    const expr = parser.parseOr();
    if (parser.position < parser.tokens.length) {
      throw new ChatTemplateError(`unexpected trailing input in "${source}"`);
    }
    return expr;
  }

  private peek(): ExprToken | undefined {
    return this.tokens[this.position];
  }

  private eat(value: string): boolean {
    const token = this.peek();
    if (token && token.value === value) {
      this.position += 1;
      return true;
    }
    return false;
  }

  private expect(value: string): void {
    if (!this.eat(value)) {
      throw new ChatTemplateError(`expected "${value}" in "${this.source}"`);
    }
  }

  private parseOr(): Expr {
    let left = this.parseAnd();
    while (this.eat('or')) {
      left = { type: 'binary', op: 'or', left, right: this.parseAnd() };
    }
    return left;
  }

  private parseAnd(): Expr {
    let left = this.parseNot();
    while (this.eat('and')) {
      left = { type: 'binary', op: 'and', left, right: this.parseNot() };
    }
    return left;
  }

  private parseNot(): Expr {
    if (this.eat('not')) return { type: 'unary', op: 'not', operand: this.parseNot() };
    return this.parseComparison();
  }

  private parseComparison(): Expr {
    let left = this.parseAdditive();
    for (;;) {
      const token = this.peek();
      if (!token) break;
      if (['==', '!=', '<', '>', '<=', '>='].includes(token.value)) {
        this.position += 1;
        left = { type: 'binary', op: token.value, left, right: this.parseAdditive() };
        continue;
      }
      if (token.value === 'in') {
        this.position += 1;
        left = { type: 'binary', op: 'in', left, right: this.parseAdditive() };
        continue;
      }
      if (token.value === 'is') {
        this.position += 1;
        const negated = this.eat('not');
        const name = this.peek();
        if (!name || name.kind !== 'word') {
          throw new ChatTemplateError(`expected a test name after "is" in "${this.source}"`);
        }
        this.position += 1;
        left = { type: 'test', target: left, name: name.value, negated };
        continue;
      }
      break;
    }
    return left;
  }

  private parseAdditive(): Expr {
    let left = this.parseMultiplicative();
    for (;;) {
      if (this.eat('+')) {
        left = { type: 'binary', op: '+', left, right: this.parseMultiplicative() };
      } else if (this.eat('-')) {
        left = { type: 'binary', op: '-', left, right: this.parseMultiplicative() };
      } else break;
    }
    return left;
  }

  private parseMultiplicative(): Expr {
    let left = this.parseUnary();
    for (;;) {
      if (this.eat('*')) left = { type: 'binary', op: '*', left, right: this.parseUnary() };
      else if (this.eat('/')) left = { type: 'binary', op: '/', left, right: this.parseUnary() };
      else break;
    }
    return left;
  }

  private parseUnary(): Expr {
    if (this.eat('-')) return { type: 'unary', op: '-', operand: this.parseUnary() };
    return this.parsePostfix();
  }

  private parsePostfix(): Expr {
    let target = this.parsePrimary();
    for (;;) {
      if (this.eat('.')) {
        const name = this.peek();
        if (!name || name.kind !== 'word') {
          throw new ChatTemplateError(`expected an attribute name in "${this.source}"`);
        }
        this.position += 1;
        target = { type: 'attr', target, name: name.value };
        continue;
      }
      if (this.eat('[')) {
        const index = this.parseOr();
        this.expect(']');
        target = { type: 'index', target, index };
        continue;
      }
      if (this.eat('|')) {
        const name = this.peek();
        if (!name || name.kind !== 'word') {
          throw new ChatTemplateError(`expected a filter name in "${this.source}"`);
        }
        this.position += 1;
        target = { type: 'filter', name: name.value, target };
        continue;
      }
      break;
    }
    return target;
  }

  private parsePrimary(): Expr {
    const token = this.peek();
    if (!token) throw new ChatTemplateError(`unexpected end of expression in "${this.source}"`);

    if (token.kind === 'number') {
      this.position += 1;
      return { type: 'literal', value: Number(token.value) };
    }
    if (token.kind === 'string') {
      this.position += 1;
      return { type: 'literal', value: token.value };
    }
    if (token.value === '(') {
      this.position += 1;
      const inner = this.parseOr();
      this.expect(')');
      return inner;
    }
    if (token.kind === 'word') {
      this.position += 1;
      if (token.value === 'true' || token.value === 'True') return { type: 'literal', value: true };
      if (token.value === 'false' || token.value === 'False') {
        return { type: 'literal', value: false };
      }
      if (token.value === 'none' || token.value === 'None') {
        return { type: 'literal', value: null };
      }
      return { type: 'name', name: token.value };
    }
    throw new ChatTemplateError(`unexpected "${token.value}" in "${this.source}"`);
  }
}

// -----------------------------------------------------------------------------------
// statements
// -----------------------------------------------------------------------------------

type Node =
  | { type: 'text'; value: string }
  | { type: 'output'; expr: Expr }
  | { type: 'if'; branches: Array<{ condition: Expr; body: Node[] }>; otherwise: Node[] }
  | { type: 'for'; name: string; iterable: Expr; body: Node[] }
  | { type: 'set'; name: string; expr: Expr };

function parse(tokens: RawToken[]): Node[] {
  let position = 0;

  const parseBody = (terminators: string[]): { body: Node[]; terminator: string } => {
    const body: Node[] = [];
    while (position < tokens.length) {
      const token = tokens[position];
      if (token.kind === 'text') {
        body.push({ type: 'text', value: token.value });
        position += 1;
        continue;
      }
      if (token.kind === 'output') {
        body.push({ type: 'output', expr: ExpressionParser.parse(token.value) });
        position += 1;
        continue;
      }

      const keyword = token.value.split(/\s+/, 1)[0];
      if (terminators.includes(keyword)) return { body, terminator: keyword };
      position += 1;

      if (keyword === 'if') {
        const branches = [
          { condition: ExpressionParser.parse(token.value.slice(2)), body: [] as Node[] },
        ];
        let otherwise: Node[] = [];
        let current = parseBody(['elif', 'else', 'endif']);
        branches[0].body = current.body;
        while (current.terminator === 'elif') {
          const elifToken = tokens[position];
          position += 1;
          const branch = {
            condition: ExpressionParser.parse(elifToken.value.slice(4)),
            body: [] as Node[],
          };
          current = parseBody(['elif', 'else', 'endif']);
          branch.body = current.body;
          branches.push(branch);
        }
        if (current.terminator === 'else') {
          position += 1;
          current = parseBody(['endif']);
          otherwise = current.body;
        }
        position += 1; // endif
        body.push({ type: 'if', branches, otherwise });
        continue;
      }

      if (keyword === 'for') {
        const match = /^for\s+(\w+)\s+in\s+([\s\S]+)$/.exec(token.value);
        if (!match) throw new ChatTemplateError(`unsupported for-loop: "${token.value}"`);
        const inner = parseBody(['endfor']);
        position += 1; // endfor
        body.push({
          type: 'for',
          name: match[1],
          iterable: ExpressionParser.parse(match[2]),
          body: inner.body,
        });
        continue;
      }

      if (keyword === 'set') {
        const match = /^set\s+(\w+)\s*=\s*([\s\S]+)$/.exec(token.value);
        if (!match) throw new ChatTemplateError(`unsupported set: "${token.value}"`);
        body.push({ type: 'set', name: match[1], expr: ExpressionParser.parse(match[2]) });
        continue;
      }

      throw new ChatTemplateError(`unsupported template statement: "${token.value}"`);
    }
    return { body, terminator: '' };
  };

  const { body } = parseBody([]);
  return body;
}

// -----------------------------------------------------------------------------------
// evaluation
// -----------------------------------------------------------------------------------

/** Jinja truthiness: empty string, empty list, 0, null and undefined are all false. */
function truthy(value: TemplateValue): boolean {
  if (value === null || value === undefined || value === false) return false;
  if (typeof value === 'string') return value.length > 0;
  if (typeof value === 'number') return value !== 0;
  if (Array.isArray(value)) return value.length > 0;
  return true;
}

function stringify(value: TemplateValue): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  return String(value);
}

const FILTERS: Record<string, (value: TemplateValue) => TemplateValue> = {
  tojson: (value) => JSON.stringify(value),
  trim: (value) => stringify(value).trim(),
  striptags: (value) => stringify(value),
  lower: (value) => stringify(value).toLowerCase(),
  upper: (value) => stringify(value).toUpperCase(),
  string: (value) => stringify(value),
  list: (value) => (Array.isArray(value) ? value : [...stringify(value)]),
  length: (value) =>
    Array.isArray(value) ? value.length : typeof value === 'string' ? value.length : 0,
  first: (value) => (Array.isArray(value) ? value[0] : undefined),
  last: (value) => (Array.isArray(value) ? value[value.length - 1] : undefined),
};

function evaluate(expr: Expr, context: TemplateContext): TemplateValue {
  switch (expr.type) {
    case 'literal':
      return expr.value;

    case 'name':
      return context[expr.name];

    case 'attr': {
      const target = evaluate(expr.target, context);
      if (target === null || target === undefined) return undefined;
      return (target as Record<string, TemplateValue>)[expr.name];
    }

    case 'index': {
      const target = evaluate(expr.target, context);
      if (target === null || target === undefined) return undefined;
      const key = evaluate(expr.index, context);
      return (target as Record<string | number, TemplateValue>)[key as string | number];
    }

    case 'unary': {
      const operand = evaluate(expr.operand, context);
      return expr.op === 'not' ? !truthy(operand) : -(operand as number);
    }

    case 'filter': {
      const filter = FILTERS[expr.name];
      if (!filter) throw new ChatTemplateError(`unsupported filter "${expr.name}"`);
      return filter(evaluate(expr.target, context));
    }

    case 'test': {
      const target = evaluate(expr.target, context);
      let result: boolean;
      switch (expr.name) {
        case 'defined':
          result = target !== undefined;
          break;
        case 'undefined':
          result = target === undefined;
          break;
        case 'none':
          result = target === null || target === undefined;
          break;
        case 'string':
          result = typeof target === 'string';
          break;
        case 'iterable':
          result = Array.isArray(target);
          break;
        default:
          throw new ChatTemplateError(`unsupported test "is ${expr.name}"`);
      }
      return expr.negated ? !result : result;
    }

    case 'binary': {
      if (expr.op === 'and') {
        return truthy(evaluate(expr.left, context)) ? evaluate(expr.right, context) : false;
      }
      if (expr.op === 'or') {
        const left = evaluate(expr.left, context);
        return truthy(left) ? left : evaluate(expr.right, context);
      }

      const left = evaluate(expr.left, context);
      const right = evaluate(expr.right, context);
      switch (expr.op) {
        case '+':
          return typeof left === 'string' || typeof right === 'string'
            ? stringify(left) + stringify(right)
            : (left as number) + (right as number);
        case '-':
          return (left as number) - (right as number);
        case '*':
          return (left as number) * (right as number);
        case '/':
          return (left as number) / (right as number);
        case '==':
          return left === right;
        case '!=':
          return left !== right;
        case '<':
          return (left as number) < (right as number);
        case '>':
          return (left as number) > (right as number);
        case '<=':
          return (left as number) <= (right as number);
        case '>=':
          return (left as number) >= (right as number);
        case 'in':
          if (Array.isArray(right)) return right.includes(left);
          if (typeof right === 'string') return right.includes(stringify(left));
          return false;
        default:
          throw new ChatTemplateError(`unsupported operator "${expr.op}"`);
      }
    }
  }
}

function render(nodes: Node[], context: TemplateContext): string {
  let out = '';
  for (const node of nodes) {
    switch (node.type) {
      case 'text':
        out += node.value;
        break;
      case 'output':
        out += stringify(evaluate(node.expr, context));
        break;
      case 'set':
        context[node.name] = evaluate(node.expr, context);
        break;
      case 'if': {
        let taken = false;
        for (const branch of node.branches) {
          if (truthy(evaluate(branch.condition, context))) {
            out += render(branch.body, context);
            taken = true;
            break;
          }
        }
        if (!taken) out += render(node.otherwise, context);
        break;
      }
      case 'for': {
        const iterable = evaluate(node.iterable, context);
        const items = Array.isArray(iterable) ? iterable : [];
        const savedItem = context[node.name];
        const savedLoop = context.loop;
        items.forEach((item, index) => {
          context[node.name] = item;
          context.loop = {
            index: index + 1,
            index0: index,
            first: index === 0,
            last: index === items.length - 1,
            length: items.length,
          };
          out += render(node.body, context);
        });
        context[node.name] = savedItem;
        context.loop = savedLoop;
        break;
      }
    }
  }
  return out;
}

export interface ChatMessage {
  role: string;
  content: string;
}

export interface RenderOptions {
  addGenerationPrompt?: boolean;
  bosToken?: string | null;
  eosToken?: string | null;
  /** Anything else the template might reference. */
  extra?: TemplateContext;
}

export class ChatTemplate {
  private readonly nodes: Node[];

  constructor(readonly source: string) {
    this.nodes = parse(lex(source));
  }

  render(messages: readonly ChatMessage[], options: RenderOptions = {}): string {
    const context: TemplateContext = {
      messages: messages.map((message) => ({ ...message })),
      add_generation_prompt: options.addGenerationPrompt ?? false,
      bos_token: options.bosToken ?? '',
      eos_token: options.eosToken ?? '',
      ...options.extra,
    };
    return render(this.nodes, context);
  }
}

interface TokenizerConfig {
  chat_template?: string;
  bos_token?: string | { content: string } | null;
  eos_token?: string | { content: string } | null;
}

function tokenText(token: TokenizerConfig['bos_token']): string | null {
  if (!token) return null;
  return typeof token === 'string' ? token : token.content;
}

export interface LoadedChatTemplate {
  template: ChatTemplate;
  bosToken: string | null;
  eosToken: string | null;
}

export async function loadChatTemplate(
  url: string,
  signal?: AbortSignal,
): Promise<LoadedChatTemplate> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new ChatTemplateError(`${url}: HTTP ${response.status} ${response.statusText}`);
  }
  const config = (await response.json()) as TokenizerConfig;
  if (!config.chat_template) {
    throw new ChatTemplateError(`${url} has no chat_template`);
  }
  return {
    template: new ChatTemplate(config.chat_template),
    bosToken: tokenText(config.bos_token),
    eosToken: tokenText(config.eos_token),
  };
}
