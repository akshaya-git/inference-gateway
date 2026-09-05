"""Deterministic task routing. No inference, network calls, or tool-text scoring."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import threading

PREFIX = re.compile(r'^\s*\[(model|complexity):(dense|moe|[1-6])\]\s*', re.I)


def text_content(message):
    value = message.get('content', '')
    if isinstance(value, str):
        return value
    return '\n'.join(p.get('text', '') for p in value if isinstance(p, dict) and p.get('type') == 'text') if isinstance(value, list) else ''


def controls(text):
    values = {}
    while match := PREFIX.match(text):
        kind, value = match.group(1).lower(), match.group(2).lower()
        if (kind == 'model' and value not in ('dense', 'moe')) or (kind == 'complexity' and not value.isdigit()):
            break
        values[kind] = value
        text = text[match.end():]
    return values, text


def matches(text, terms):
    return any(re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', text, re.I) for term in terms)


def validate_rules(rules):
    if not isinstance(rules.get('version'), int) or rules['version'] < 1:
        raise ValueError('version must be a positive integer')
    if rules.get('dense_above') != 4:
        raise ValueError('dense_above must be 4')
    if set(rules.get('levels', {})) != set('123456'):
        raise ValueError('Define all six levels')
    groups = rules['base_rules'] + rules['modifiers']
    if len({x['id'] for x in groups}) != len(groups):
        raise ValueError('Rule IDs must be unique')
    for terms in [rules['code_terms']] + [x['terms'] for x in groups]:
        if not terms or not all(isinstance(t, str) and t.strip() for t in terms):
            raise ValueError('Terms must be nonempty strings')
    if any(type(x['level']) is not int or not 1 <= x['level'] <= 6 for x in rules['base_rules']):
        raise ValueError('Base levels must be 1–6')
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
        flags, text = controls(users[-1] if users else '')
        previous_key = hashlib.sha256(json.dumps(users[:-1]).encode()).hexdigest()
        if not explicit and not flags and text.strip().lower().rstrip('.!') in ('continue', 'go on', 'keep going') and previous_key in self.tasks:
            decision = deepcopy(self.tasks[previous_key])
        else:
            # Ignore quoted lines and fenced examples in task scoring.
            task = re.sub(r'```[\s\S]*?```', '', text)
            task = '\n'.join(line for line in task.splitlines() if not line.lstrip().startswith('>'))
            code = matches(task, self.rules['code_terms'])
            matched = [r for r in self.rules['base_rules'] if matches(task, r['terms'])] if code else []
            level = max((r['level'] for r in matched), default=3 if code else 1)
            modifiers = [r['id'] for r in self.rules['modifiers'] if matches(task, r['terms'])] if code else []
            level = min(6, level + min(2, len(modifiers)))
            if 'complexity' in flags:
                level = int(flags['complexity'])
            route = explicit or flags.get('model') or ('dense' if code and level > 4 else 'moe')
            reason = 'explicit model' if explicit or flags.get('model') else 'explicit complexity' if 'complexity' in flags else ', '.join([r['id'] for r in matched] + modifiers) or 'default'
            decision = {'route': route, 'complexity': level, 'code_related': code,
                        'rules_version': self.rules['version'], 'source': reason,
                        'reason': f'Level {level}; {reason}; lookup v{self.rules["version"]}'}
        self.tasks[key] = decision
        while len(self.tasks) > self.capacity:
            self.tasks.popitem(last=False)
        return deepcopy(decision)


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
                        break
    return result
