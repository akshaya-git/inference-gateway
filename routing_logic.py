"""Deterministic task routing. No inference, network calls, or tool-text scoring.

Matching model (v2):
- code_terms gate: sets the *default* level (3 for code-ish, 1 otherwise). It no
  longer vetoes base rules or modifiers (R3).
- base rules / modifiers: matched via stemmed token-set containment, so word
  order, singular/plural, and common inflections do not kill a match (R2).
  Each rule may list multiple phrase variants.
- route: level_routes map (optional) or the dense_above threshold (R4).
- controls: [model:...] / [complexity:1-6] prefixes; invalid controls are
  stripped and reported, never forwarded and never clamped (R6).
"""
import hashlib
import json
import re
import threading
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

PREFIX = re.compile(r'^\s*\[(model|complexity):([A-Za-z0-9_-]+)\]\s*', re.I)

# Function words and weak request verbs. They carry no routing signal and are
# excluded from token matching so rules like "change a color" reduce to {color}.
STOPWORDS = frozenset("""
a an and are as at be but by can could did do does for from had has have he her his
i if in into is it its me my no nor not of off on once one or our she so some such
than that the their them then there these they this to too was we were what when
where which while who why will with within without would you your yours please
just only also very really actually basically simply quite rather pretty even still
already always never often sometimes usually maybe perhaps sure ok okay yeah yes
hmm uh bit little few many much more most less least all each any both either
neither other another same different new old big small first last next previous
current here now today tomorrow yesterday make want need help get use using try
show tell give let put take keep hold turn move change add remove write create
build fix
""".split())


def stem(word: str) -> str:
    """Conservative suffix stemmer: keeps base and inflected forms comparable."""
    w = word.lower()
    if len(w) > 5 and w.endswith('ations'):
        return w[:-6]
    if len(w) > 5 and w.endswith('ation'):
        return w[:-5]
    if len(w) > 4 and w.endswith('ies'):
        return w[:-3] + 'y'
    if len(w) > 4 and w.endswith('ing'):
        return w[:-3]
    if len(w) > 3 and w.endswith('ed'):
        return w[:-2]
    if len(w) > 3 and w.endswith('es'):
        return w[:-2]
    if len(w) > 3 and w.endswith('s') and not w.endswith('ss'):
        return w[:-1]
    if len(w) > 4 and w.endswith('ly'):
        return w[:-2]
    if len(w) > 4 and w.endswith('er'):
        return w[:-2]
    if len(w) > 3 and w.endswith('e') and not w.endswith(('ee', 'le')):
        return w[:-1]
    return w


def tokens(text: str) -> set:
    return {stem(t) for t in re.findall(r'[a-z0-9]+', text.lower()) if t not in STOPWORDS}


def matches(text: str, terms) -> bool:
    """Token-set containment (stemmed) with an exact word-boundary fallback."""
    task_tokens = tokens(text)
    for term in terms:
        term_tokens = tokens(term)
        if term_tokens and term_tokens <= task_tokens:
            return True
        if re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', text, re.I):
            return True
    return False


def validate_rules(rules):
    if not isinstance(rules.get('version'), int) or rules['version'] < 1:
        raise ValueError('version must be a positive integer')
    if type(rules.get('dense_above')) is not int or not 1 <= rules['dense_above'] <= 5:
        raise ValueError('dense_above must be an integer 1-5')
    level_routes = rules.get('level_routes', {})
    if not isinstance(level_routes, dict):
        raise ValueError('level_routes must be an object')
    for key, value in level_routes.items():
        if str(key) not in '123456' or value not in ('moe', 'dense'):
            raise ValueError('level_routes maps levels 1-6 to moe/dense')
    if set(rules.get('levels', {})) != set('123456'):
        raise ValueError('Define all six levels')
    groups = rules['base_rules'] + rules['modifiers']
    if len({x['id'] for x in groups}) != len(groups):
        raise ValueError('Rule IDs must be unique')
    for terms in [rules['code_terms']] + [x['terms'] for x in groups]:
        if not terms or not all(isinstance(t, str) and t.strip() for t in terms):
            raise ValueError('Terms must be nonempty strings')
    if any(type(x['level']) is not int or not 1 <= x['level'] <= 6 for x in rules['base_rules']):
        raise ValueError('Base levels must be 1-6')
    return rules


class RoutingEngine:
    def __init__(self, path, capacity=512):
        self.path = Path(path)
        self.capacity = capacity
        self.tasks = OrderedDict()
        self.lock = threading.RLock()
        self.rules = validate_rules(json.loads(self.path.read_text()))
        self.last_error = None

    def reload(self):
        # Invalid edits preserve the last valid lookup.
        try:
            self.rules = validate_rules(json.loads(self.path.read_text()))
            self.last_error = None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.last_error = str(exc)

    def update_rules(self, rules):
        """Gateway integration hook: validate and atomically install a reviewed lookup."""
        with self.lock:
            rules = validate_rules(deepcopy(rules))
            if rules['version'] <= self.rules['version']:
                raise ValueError('Updates must increment version')
            temporary = self.path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(rules, indent=2) + '\n')
            temporary.replace(self.path)
            self.rules = rules

    def route(self, body, explicit=None):
        with self.lock:
            return self._route(body, explicit)

    def _route(self, body, explicit):
        self.reload()
        users = [text_content(m) for m in body.get('messages', []) if m.get('role') == 'user']
        key = hashlib.sha256(json.dumps(users).encode()).hexdigest()
        if not explicit and key in self.tasks:
            self.tasks.move_to_end(key)
            return deepcopy(self.tasks[key])
        flags, text, invalid = controls(users[-1] if users else '')
        previous_key = hashlib.sha256(json.dumps(users[:-1]).encode()).hexdigest()
        if (not explicit and not flags and not invalid
                and text.strip().lower().rstrip('.!') in ('continue', 'go on', 'keep going')
                and previous_key in self.tasks):
            decision = deepcopy(self.tasks[previous_key])
        else:
            # Ignore quoted lines and fenced examples in task scoring.
            task = re.sub(r'```[\s\S]*?```', '', text)
            task = '\n'.join(line for line in task.splitlines() if not line.lstrip().startswith('>'))
            code = matches(task, self.rules['code_terms'])
            matched = [r for r in self.rules['base_rules'] if matches(task, r['terms'])]
            modifier_hits = [r for r in self.rules['modifiers'] if matches(task, r['terms'])]
            levels = [r['level'] for r in matched]
            level = max(levels) if levels else (3 if code else 1)
            level = min(6, level + min(2, len(modifier_hits)))
            if 'complexity' in flags:
                level = int(flags['complexity'])
            level_routes = {str(k): v for k, v in self.rules.get('level_routes', {}).items()}
            if explicit or flags.get('model'):
                route = explicit or flags['model']
            else:
                route = level_routes.get(str(level), 'dense' if level > self.rules['dense_above'] else 'moe')
            source_parts = []
            if invalid:
                source_parts.append('invalid control stripped: ' + ', '.join(invalid))
            source_parts += [r['id'] for r in matched] + [r['id'] for r in modifier_hits]
            if explicit or flags.get('model'):
                source = 'explicit model'
            elif 'complexity' in flags:
                source = 'explicit complexity'
            else:
                source = ', '.join(source_parts) or 'default'
            reason = f'Level {level}; {source}; lookup v{self.rules["version"]}'
            decision = {
                'route': route, 'complexity': level, 'code_related': code,
                'matched_rules': [r['id'] for r in matched],
                'modifiers': [r['id'] for r in modifier_hits],
                'invalid_controls': list(invalid),
                'rules_version': self.rules['version'], 'source': source,
                'reason': reason,
            }
        self.tasks[key] = decision
        while len(self.tasks) > self.capacity:
            self.tasks.popitem(last=False)
        return deepcopy(decision)


def text_content(message):
    value = message.get('content', '')
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(p.get('text', '') for p in value if isinstance(p, dict) and p.get('type') == 'text')
    return ''


def controls(text):
    """Extract leading [model:...] / [complexity:N] controls.

    Valid controls are consumed; invalid ones (unknown model, complexity outside
    1-6) are stripped and returned so callers can warn instead of forwarding
    raw control text to the model.
    """
    values = {}
    invalid = []
    while match := PREFIX.match(text):
        kind, value = match.group(1).lower(), match.group(2).lower()
        if kind == 'model' and value not in ('dense', 'moe'):
            invalid.append(match.group(0).strip())
        elif kind == 'complexity' and (not value.isdigit() or not 1 <= int(value) <= 6):
            invalid.append(match.group(0).strip())
        else:
            values[kind] = value
        text = text[match.end():]
    return values, text, invalid


def strip_routing_controls(body):
    result = deepcopy(body)
    for message in result.get('messages', []):
        if message.get('role') == 'user':
            if isinstance(message.get('content'), str):
                message['content'] = controls(message['content'])[1]
            elif isinstance(message.get('content'), list):
                for part in message['content']:
                    if part.get('type') == 'text':
                        part['text'] = controls(part.get('text', ''))[1]
    return result
