import json
from pathlib import Path

import pytest

from routing_logic import RoutingEngine, controls, strip_routing_controls


@pytest.fixture
def engine(tmp_path):
    p = tmp_path / 'rules.json'
    p.write_text((Path(__file__).parents[1] / 'routing_rules.json').read_text())
    return RoutingEngine(p)


def body(text):
    return {'messages': [{'role': 'user', 'content': text}]}


# 12-prompt regression suite from the routing gap analysis (CP-2).
@pytest.mark.parametrize('text,route,level', [
    # Auth terms are out of scope for now: flask/app make it code, no rule fires.
    ('Add authentication with JWT to my flask app', 'moe', 3),
    # Concurrency rule -> L6 dense.
    ('Fix the race condition in my worker pool', 'dense', 6),
    # System rule -> L6 dense.
    ('Redesign the system to handle 10x traffic', 'dense', 6),
    # Data migration rule -> L6 dense (sqlite is a code term, postgres is not required).
    ('Migrate my database from sqlite to postgres', 'dense', 6),
    # Plain feature: code default L3, no rule.
    ('Build a todo list app', 'moe', 3),
    # Security rule -> L6 dense.
    ('Review my pull request for security issues', 'dense', 6),
    # "function" is not a code term: not code, L1.
    ('What is the function of the liver?', 'moe', 1),
    # Invalid control stripped + reported, then scored normally (code -> L3).
    ('[complexity:7] write code', 'moe', 3),
    # Non-code, L1.
    ('Explain philosophy', 'moe', 1),
    # Benchmark vocabulary removed: browser is code (L3) + polished modifier (L4).
    ('Build polished browser Tetris', 'moe', 4),
    # Mechanical rule -> L1.
    ('Rename a variable in my python file', 'moe', 1),
    # Small rule -> L2.
    ('Add input validation to the login form', 'moe', 2),
    # Valid explicit controls still work.
    ('[complexity:4] Write code', 'moe', 4),
    ('[complexity:5] Write code', 'dense', 5),
])
def test_levels(engine, text, route, level):
    d = engine.route(body(text))
    assert (d['route'], d['complexity']) == (route, level)


def test_invalid_control_stripped_and_reported(engine):
    d = engine.route(body('[complexity:7] write code'))
    assert d['invalid_controls'] == ['[complexity:7]']
    assert 'invalid control stripped' in d['source']
    # The raw tag must never be forwarded to the model.
    stripped = strip_routing_controls(body('[complexity:7] write code'))
    assert stripped['messages'][0]['content'] == 'write code'


def test_invalid_model_control_stripped(engine):
    d = engine.route(body('[model:quantum] hello'))
    assert d['invalid_controls'] == ['[model:quantum]']
    assert d['route'] == 'moe'
    stripped = strip_routing_controls(body('[model:quantum] hello'))
    assert stripped['messages'][0]['content'] == 'hello'


def test_strip_multi_part_content(engine):
    b = {'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': '[model:dense] first'},
        {'type': 'image_url', 'image_url': {'url': 'http://x/y.png'}},
        {'type': 'text', 'text': '[complexity:9] second'},
    ]}]}
    out = strip_routing_controls(b)
    parts = out['messages'][0]['content']
    assert parts[0]['text'] == 'first'
    assert parts[2]['text'] == 'second'
    assert parts[1] == {'type': 'image_url', 'image_url': {'url': 'http://x/y.png'}}


def test_code_gate_does_not_veto_rules(engine):
    # No code terms at all, but the concurrency rule matches -> dense.
    d = engine.route(body('There is a race condition in the story'))
    assert d['code_related'] is False
    assert d['route'] == 'dense'
    assert d['matched_rules'] == ['concurrency']


def test_matched_rules_and_modifiers_reported(engine):
    d = engine.route(body('Fix the race condition in my worker pool'))
    assert d['matched_rules'] == ['concurrency']
    assert d['modifiers'] == []
    assert d['rules_version'] == 2
    assert 'concurrency' in d['reason']


def test_modifier_applied(engine):
    d = engine.route(body('Build a polished kanban board with drag and drop'))
    assert d['matched_rules'] == ['stateful']
    assert d['modifiers'] == ['visual_polish']


def test_dense_above_configurable(tmp_path):
    rules = json.loads((Path(__file__).parents[1] / 'routing_rules.json').read_text())
    rules['version'] = 9
    rules['dense_above'] = 2
    p = tmp_path / 'rules.json'
    p.write_text(json.dumps(rules))
    e = RoutingEngine(p)
    # L3 now exceeds dense_above=2 -> dense.
    assert e.route(body('Build a todo list app'))['route'] == 'dense'
    assert e.route(body('Explain philosophy'))['route'] == 'moe'


def test_level_routes_map(tmp_path):
    rules = json.loads((Path(__file__).parents[1] / 'routing_rules.json').read_text())
    rules['version'] = 9
    rules['level_routes'] = {'3': 'dense'}
    p = tmp_path / 'rules.json'
    p.write_text(json.dumps(rules))
    e = RoutingEngine(p)
    assert e.route(body('Build a todo list app'))['route'] == 'dense'
    # Unmapped levels fall back to the dense_above threshold.
    assert e.route(body('Explain philosophy'))['route'] == 'moe'
    assert e.route(body('Fix the race condition in my worker pool'))['route'] == 'dense'


def test_rules_validation(tmp_path):
    base = json.loads((Path(__file__).parents[1] / 'routing_rules.json').read_text())

    def attempt(mutate):
        rules = json.loads(json.dumps(base))
        rules['version'] = 9
        mutate(rules)
        p = tmp_path / 'rules.json'
        p.write_text(json.dumps(rules))
        with pytest.raises(ValueError):
            RoutingEngine(p)

    def bad_dense_above(r):
        r['dense_above'] = 7
    def bad_level_routes(r):
        r['level_routes'] = {'3': 'quantum'}
    def bad_level_routes_key(r):
        r['level_routes'] = {'9': 'moe'}
    def bad_level(r):
        r['base_rules'][0]['level'] = 0

    attempt(bad_dense_above)
    attempt(bad_level_routes)
    attempt(bad_level_routes_key)
    attempt(bad_level)


def test_tool_followup_sticky_across_reload(engine):
    b = body('Fix the race condition in my worker pool')
    d = engine.route(b)
    b['messages'].append({'role': 'tool', 'content': '[model:moe] change a color'})
    rules = json.loads(engine.path.read_text())
    rules['base_rules'] = []
    rules['version'] = 3
    engine.update_rules(rules)
    assert engine.route(b) == d


def test_override_strip(engine):
    b = body('[model:dense] Hello')
    assert engine.route(b)['route'] == 'dense'
    assert strip_routing_controls(b)['messages'][0]['content'] == 'Hello'
    assert engine.route(b, explicit='moe')['route'] == 'moe'


def test_invalid_rules_preserve_last_good(engine):
    engine.path.write_text('{}')
    assert engine.route(body('Fix the race condition in my worker pool'))['route'] == 'dense'
    assert engine.last_error


def test_continue_retains_route(engine):
    b = body('Fix the race condition in my worker pool')
    engine.route(b)
    b['messages'].append({'role': 'user', 'content': 'continue'})
    assert engine.route(b)['route'] == 'dense'


def test_controls_parser():
    values, text, invalid = controls('[model:dense] [complexity:3] hello')
    assert values == {'model': 'dense', 'complexity': '3'}
    assert text == 'hello'
    assert invalid == []
    values, text, invalid = controls('[complexity:0] [complexity:9] hi')
    assert values == {}
    assert text == 'hi'
    assert invalid == ['[complexity:0]', '[complexity:9]']
