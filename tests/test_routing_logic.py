import json
from pathlib import Path
import pytest
from routing_logic import RoutingEngine, strip_routing_controls

@pytest.fixture
def engine(tmp_path):
    p = tmp_path / 'rules.json'
    p.write_text((Path(__file__).parents[1] / 'routing_rules.json').read_text())
    return RoutingEngine(p)

def body(text):
    return {'messages': [{'role': 'user', 'content': text}]}

@pytest.mark.parametrize('text,route,level', [
 ('Build HTML bill splitter', 'moe', 3),
 ('Build browser Tetris', 'moe', 4),
 ('Build polished browser Tetris', 'dense', 5),
 ('[complexity:4] Write code', 'moe', 4),
 ('[complexity:5] Write code', 'dense', 5),
 ('Explain philosophy', 'moe', 1),
])
def test_levels(engine,text,route,level):
    d=engine.route(body(text)); assert (d['route'],d['complexity'])==(route,level)

def test_tool_followup_sticky_across_reload(engine):
    b=body('Build polished Tetris');d=engine.route(b)
    b['messages'].append({'role':'tool','content':'[model:moe] change a color'})
    rules=json.loads(engine.path.read_text());rules['base_rules']=[];rules['version']=2
    engine.update_rules(rules)
    assert engine.route(b)==d

def test_override_strip(engine):
    b=body('[model:dense] Hello')
    assert engine.route(b)['route']=='dense'
    assert strip_routing_controls(b)['messages'][0]['content']=='Hello'
    assert engine.route(b,explicit='moe')['route']=='moe'

def test_invalid_rules_preserve_last_good(engine):
    engine.path.write_text('{}')
    assert engine.route(body('Build polished Tetris'))['route']=='dense'
    assert engine.last_error

def test_continue_retains_route(engine):
    b=body('Build polished Tetris');engine.route(b)
    b['messages'].append({'role':'user','content':'continue'})
    assert engine.route(b)['route']=='dense'
