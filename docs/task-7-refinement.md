# Task 7: Instruction Refinement Loop

## Overview

Create a feedback loop that analyzes routing decisions and benchmark results to automatically refine the judge model's routing instructions. This enables the system to learn which model excels at which task type over time.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Instruction Refinement Loop                           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Data Collection                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │   │
│  │  │  Routing     │  │  Benchmark   │  │  User Feedback       │  │   │
│  │  │  Decisions   │  │  Results     │  │  (thumbs up/down)    │  │   │
│  │  │  (Task 3)    │  │  (Task 4)    │  │                      │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │   │
│  │         │                    │                    │              │   │
│  │         └────────────────────┴────────────────────┘              │   │
│  │                              │                                   │   │
│  │                              ▼                                   │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │              Analysis Engine                             │   │   │
│  │  │                                                           │   │   │
│  │  │  • Identify patterns (which tasks route to which model)  │   │   │
│  │  │  • Detect misroutes (low confidence + bad outcome)       │   │   │
│  │  │  • Calculate success rate per task type per model        │   │   │
│  │  │  • Generate insights for instruction refinement          │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                              │                                   │   │
│  │                              ▼                                   │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │              Instruction Generator                       │   │   │
│  │  │                                                           │   │   │
│  │  │  • Analyze patterns from data                            │   │   │
│  │  │  • Generate updated routing instructions                 │   │   │
│  │  │  • A/B test new instructions vs old                      │   │   │
│  │  │  • Roll out if improvement detected                      │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                              │                                   │   │
│  │                              ▼                                   │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │              Judge Model (Updated)                       │   │   │
│  │  │                                                           │   │   │
│  │  │  • New system prompt with refined instructions           │   │   │
│  │  │  • Better routing accuracy                               │   │   │
│  │  │  • Continuous improvement loop                           │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Agentic Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Agentic Development Pipeline                        │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│  │  Developer  │───▶│  Tester     │───▶│  Engineer   │                │
│  │  Agent      │    │  Agent      │    │  Agent      │                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
│        │                    │                    │                      │
│        ▼                    ▼                    ▼                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│  │  Build      │    │  Run        │    │  Verify     │                │
│  │  analysis   │    │  refinement │    │  improved   │                │
│  │  +          │    │  loop with  │    │  routing    │                │
│  │  generator  │    │  sample     │    │  accuracy   │                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
│        │                    │                    │                      │
│        └────────────────────┴────────────────────┘                      │
│                              │                                          │
│                              ▼                                          │
│                     ┌─────────────────┐                                │
│                     │  QA Agent       │                                │
│                     │  (verify loop   │                                │
│                     │   convergence)  │                                │
│                     └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Responsibility | Tools |
|-------|---------------|-------|
| **Developer** | Build analysis engine, instruction generator | `write`, `edit` |
| **Tester** | Run refinement loop with sample data | `bash` (pytest, curl) |
| **Engineer** | Verify improved routing accuracy | `bash` (curl) |
| **QA** | Check loop convergence, edge cases | `read`, `bash` |

## Step-by-Step Execution

### Step 1: Build Analysis Engine

**refinement/analysis.py** (new file):

```python
"""
Analysis engine for routing decision patterns.
"""
import json
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class RoutingPattern:
    """A pattern in routing decisions."""
    task_type: str
    route: str
    count: int
    avg_confidence: float
    success_rate: float
    avg_latency_ms: float
    examples: list[str] = field(default_factory=list)


@dataclass
class Misroute:
    """A detected misroute."""
    request_id: str
    task_type: str
    routed_to: str
    should_have_routed_to: str
    confidence: float
    reason: str


class RoutingAnalyzer:
    """Analyze routing decisions to identify patterns and misroutes."""
    
    def __init__(self):
        self.patterns: dict[str, dict[str, RoutingPattern]] = defaultdict(dict)
        self.misroutes: list[Misroute] = []
    
    def add_decision(self, decision: dict) -> None:
        """Add a routing decision to the analysis."""
        task_type = decision.get("task_type", "unknown")
        route = decision.get("route", "unknown")
        confidence = decision.get("confidence", 0.0)
        latency = decision.get("latency_ms", 0.0)
        success = decision.get("success", True)
        prompt = decision.get("prompt", "")[:100]
        
        if route not in self.patterns[task_type]:
            self.patterns[task_type][route] = RoutingPattern(
                task_type=task_type,
                route=route,
                count=0,
                avg_confidence=0.0,
                success_rate=0.0,
                avg_latency_ms=0.0,
            )
        
        pattern = self.patterns[task_type][route]
        pattern.count += 1
        pattern.avg_confidence = (pattern.avg_confidence * (pattern.count - 1) + confidence) / pattern.count
        pattern.success_rate = (pattern.success_rate * (pattern.count - 1) + (1.0 if success else 0.0)) / pattern.count
        pattern.avg_latency_ms = (pattern.avg_latency_ms * (pattern.count - 1) + latency) / pattern.count
        if len(pattern.examples) < 5:
            pattern.examples.append(prompt)
    
    def detect_misroutes(self, threshold: float = 0.3) -> list[Misroute]:
        """Detect potential misroutes based on low confidence and failure."""
        misroutes = []
        
        for task_type, routes in self.patterns.items():
            for route, pattern in routes.items():
                # Low confidence + high failure rate = potential misroute
                if pattern.avg_confidence < threshold and pattern.success_rate < 0.5:
                    # Determine what route it should have gone to
                    best_route = self._find_best_route(task_type, exclude=route)
                    if best_route:
                        misroutes.append(Misroute(
                            request_id="",
                            task_type=task_type,
                            routed_to=route,
                            should_have_routed_to=best_route,
                            confidence=pattern.avg_confidence,
                            reason=f"Low confidence ({pattern.avg_confidence:.2f}) and high failure rate ({pattern.success_rate:.2f})",
                        ))
        
        self.misroutes = misroutes
        return misroutes
    
    def _find_best_route(self, task_type: str, exclude: str) -> Optional[str]:
        """Find the best alternative route for a task type."""
        routes = self.patterns.get(task_type, {})
        best_route = None
        best_score = -1
        
        for route, pattern in routes.items():
            if route == exclude:
                continue
            score = pattern.success_rate * 0.7 + pattern.avg_confidence * 0.3
            if score > best_score:
                best_score = score
                best_route = route
        
        return best_route
    
    def get_insights(self) -> list[str]:
        """Generate insights from the analysis."""
        insights = []
        
        # Identify task types with clear best routes
        for task_type, routes in self.patterns.items():
            if len(routes) > 1:
                best_route = max(routes.values(), key=lambda p: p.success_rate)
                worst_route = min(routes.values(), key=lambda p: p.success_rate)
                
                if best_route.success_rate - worst_route.success_rate > 0.3:
                    insights.append(
                        f"Task type '{task_type}': {best_route.route} performs significantly better "
                        f"({best_route.success_rate:.0%} vs {worst_route.success_rate:.0%})"
                    )
        
        # Report misroutes
        if self.misroutes:
            insights.append(f"Detected {len(self.misroutes)} potential misroutes")
        
        return insights
    
    def export(self) -> dict:
        """Export analysis results."""
        return {
            "patterns": {
                task_type: {
                    route: {
                        "count": pattern.count,
                        "avg_confidence": pattern.avg_confidence,
                        "success_rate": pattern.success_rate,
                        "avg_latency_ms": pattern.avg_latency_ms,
                        "examples": pattern.examples,
                    }
                    for route, pattern in routes.items()
                }
                for task_type, routes in self.patterns.items()
            },
            "misroutes": [
                {
                    "task_type": m.task_type,
                    "routed_to": m.routed_to,
                    "should_have_routed_to": m.should_have_routed_to,
                    "confidence": m.confidence,
                    "reason": m.reason,
                }
                for m in self.misroutes
            ],
            "insights": self.get_insights(),
        }
```

### Step 2: Build Instruction Generator

**refinement/generator.py** (new file):

```python
"""
Instruction generator for refining judge model routing instructions.
"""
import json
from typing import Optional
from refinement.analysis import RoutingAnalyzer


class InstructionGenerator:
    """Generate refined routing instructions based on analysis."""
    
    BASE_INSTRUCTION = """Select the best model route for the NEXT assistant completion.
Evaluate the unresolved objective, message roles, tool calls, and requested output.
The resident model is a preference, not a mandate. Switch when another route offers
a meaningful quality or capability advantage."""
    
    def __init__(self, analyzer: RoutingAnalyzer):
        self.analyzer = analyzer
    
    def generate(self) -> str:
        """Generate refined routing instructions."""
        insights = self.analyzer.get_insights()
        patterns = self.analyzer.export()["patterns"]
        
        instruction = self.BASE_INSTRUCTION
        
        # Add pattern-based guidance
        if patterns:
            instruction += "\n\nBased on historical performance:"
            for task_type, routes in patterns.items():
                if len(routes) > 1:
                    best_route = max(routes.values(), key=lambda p: p["success_rate"])
                    if best_route["success_rate"] > 0.7:
                        instruction += f"\n- For {task_type} tasks, prefer {best_route['route']} (success rate: {best_route['success_rate']:.0%})"
        
        # Add misroute corrections
        misroutes = self.analyzer.export()["misroutes"]
        if misroutes:
            instruction += "\n\nAvoid these common misroutes:"
            for m in misroutes[:5]:  # Limit to top 5
                instruction += f"\n- {m['task_type']} tasks: route to {m['should_have_routed_to']} instead of {m['routed_to']}"
        
        return instruction
    
    def generate_with_ab_test(self) -> dict:
        """Generate instructions with A/B test configuration."""
        current = self.generate()
        
        # Generate a variant for A/B testing
        variant = self.generate_variant(current)
        
        return {
            "current": current,
            "variant": variant,
            "test_config": {
                "split": 0.5,  # 50/50 split
                "min_samples": 100,
                "significance_level": 0.05,
            },
        }
    
    def generate_variant(self, base: str) -> str:
        """Generate a variant instruction for A/B testing."""
        # Simple variant: emphasize different aspects
        return base + "\n\nAdditional guidance: Prioritize response quality over speed when confidence is high."
```

### Step 3: Add Refinement Endpoints

**proxy.py** — Add refinement endpoints:

```python
from refinement.analysis import RoutingAnalyzer
from refinement.generator import InstructionGenerator

# Initialize refinement components
analyzer = RoutingAnalyzer()
generator = InstructionGenerator(analyzer)

@app.get("/api/refinement/analysis")
async def get_refinement_analysis():
    """Get routing analysis insights."""
    return analyzer.export()

@app.post("/api/refinement/generate")
async def generate_refined_instructions():
    """Generate refined routing instructions."""
    instructions = generator.generate()
    return {
        "instructions": instructions,
        "insights": analyzer.get_insights(),
    }

@app.post("/api/refinement/apply")
async def apply_refined_instructions():
    """Apply refined instructions to the judge model."""
    global ROUTER_SYSTEM_PROMPT
    
    instructions = generator.generate()
    ROUTER_SYSTEM_PROMPT = instructions
    
    logger.info("Applied refined routing instructions")
    return {
        "status": "applied",
        "instructions": instructions,
    }

@app.get("/api/refinement/status")
async def get_refinement_status():
    """Get refinement loop status."""
    return {
        "total_decisions_analyzed": sum(
            sum(p.count for p in routes.values())
            for routes in analyzer.patterns.values()
        ),
        "task_types_tracked": len(analyzer.patterns),
        "misroutes_detected": len(analyzer.misroutes),
        "insights": analyzer.get_insights(),
    }
```

### Step 4: Integrate with Request Flow

**proxy.py** — Update request flow to feed analyzer:

```python
# In proxy_nonstream() and stream_request(), after recording metrics:
async def record_routing_decision(metric: Metric):
    """Feed routing decision to the analyzer."""
    analyzer.add_decision({
        "task_type": metric.task_type,
        "route": metric.route,
        "confidence": metric.router_confidence or 0.0,
        "latency_ms": metric.router_ms or 0.0,
        "success": metric.status_code == 200,
        "prompt": metric.prompt[:100] if metric.prompt else "",
    })
```

### Step 5: Test Refinement Loop

**tests/test_refinement.py**:

```python
import pytest
from refinement.analysis import RoutingAnalyzer
from refinement.generator import InstructionGenerator


class TestRoutingAnalyzer:
    def setup_method(self):
        self.analyzer = RoutingAnalyzer()
    
    def test_add_decision(self):
        self.analyzer.add_decision({
            "task_type": "coding",
            "route": "moe",
            "confidence": 0.8,
            "latency_ms": 100,
            "success": True,
            "prompt": "Write a function",
        })
        
        assert "coding" in self.analyzer.patterns
        assert "moe" in self.analyzer.patterns["coding"]
        assert self.analyzer.patterns["coding"]["moe"].count == 1
    
    def test_detect_misroutes(self):
        # Add decisions with low confidence and high failure
        for _ in range(10):
            self.analyzer.add_decision({
                "task_type": "complex_reasoning",
                "route": "moe",
                "confidence": 0.2,
                "latency_ms": 500,
                "success": False,
                "prompt": "Complex problem",
            })
        
        # Add decisions with high confidence and success for dense
        for _ in range(10):
            self.analyzer.add_decision({
                "task_type": "complex_reasoning",
                "route": "dense",
                "confidence": 0.9,
                "latency_ms": 2000,
                "success": True,
                "prompt": "Complex problem",
            })
        
        misroutes = self.analyzer.detect_misroutes()
        assert len(misroutes) > 0
        assert any(m.task_type == "complex_reasoning" for m in misroutes)
    
    def test_get_insights(self):
        # Add decisions showing clear pattern
        for _ in range(20):
            self.analyzer.add_decision({
                "task_type": "coding",
                "route": "moe",
                "confidence": 0.9,
                "latency_ms": 100,
                "success": True,
                "prompt": "Write code",
            })
        
        insights = self.analyzer.get_insights()
        assert len(insights) > 0


class TestInstructionGenerator:
    def setup_method(self):
        self.analyzer = RoutingAnalyzer()
        self.generator = InstructionGenerator(self.analyzer)
    
    def test_generate_base(self):
        instructions = self.generator.generate()
        assert "Select the best model route" in instructions
    
    def test_generate_with_patterns(self):
        # Add pattern data
        for _ in range(10):
            self.analyzer.add_decision({
                "task_type": "coding",
                "route": "moe",
                "confidence": 0.9,
                "latency_ms": 100,
                "success": True,
                "prompt": "Write code",
            })
        
        instructions = self.generator.generate()
        assert "coding" in instructions.lower()
        assert "moe" in instructions.lower()
    
    def test_generate_with_misroutes(self):
        # Add misroute data
        for _ in range(10):
            self.analyzer.add_decision({
                "task_type": "complex",
                "route": "moe",
                "confidence": 0.2,
                "latency_ms": 500,
                "success": False,
                "prompt": "Complex problem",
            })
        
        self.analyzer.detect_misroutes()
        instructions = self.generator.generate()
        assert "Avoid" in instructions or "misroute" in instructions.lower()
```

### Step 6: Verify Locally

```bash
# Check analysis
curl http://localhost:9000/api/refinement/analysis | python3 -m json.tool

# Generate refined instructions
curl -X POST http://localhost:9000/api/refinement/generate | python3 -m json.tool

# Apply refined instructions
curl -X POST http://localhost:9000/api/refinement/apply

# Check status
curl http://localhost:9000/api/refinement/status | python3 -m json.tool

# Run tests
pytest tests/test_refinement.py -v
```

## Success Criteria

- [ ] Analysis engine tracks routing patterns
- [ ] Misroute detection works
- [ ] Instruction generator produces refined instructions
- [ ] Refinement endpoints work
- [ ] Tests pass for analysis and generation
- [ ] Loop converges (improved accuracy over time)

## Commands Reference

```bash
# Check analysis
curl http://localhost:9000/api/refinement/analysis | python3 -m json.tool

# Generate instructions
curl -X POST http://localhost:9000/api/refinement/generate | python3 -m json.tool

# Apply instructions
curl -X POST http://localhost:9000/api/refinement/apply

# Check status
curl http://localhost:9000/api/refinement/status | python3 -m json.tool

# Run tests
pytest tests/test_refinement.py -v
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No patterns detected | Make more requests first |
| Instructions not changing | Check if enough data collected |
| Misroutes not detected | Lower confidence threshold |
| Loop not converging | Increase min_samples for A/B test |
