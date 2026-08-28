# Task 4: Enhanced Benchmark Suites

## Overview

Add 5+ new benchmark suites that compare the two models across different task types. Each suite measures quality, latency, and cost to help the judge learn which model excels at what.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Benchmark Suite Architecture                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Benchmark Lab (Dashboard)                     │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │  Suite   │ │  Run     │ │  Results │ │  Compare │           │   │
│  │  │  Select  │ │  Button  │ │  Table   │ │  View    │           │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Benchmark Runner                              │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │  For each suite:                                        │   │   │
│  │  │  1. Load prompts                                        │   │   │
│  │  │  2. Run against MoE (if enabled)                        │   │   │
│  │  │  3. Run against Dense (if enabled)                      │   │   │
│  │  │  4. Evaluate outputs with suite-specific rubric         │   │   │
│  │  │  5. Store results                                       │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Evaluation Rubrics                            │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │   │
│  │  │  Quick       │ │  Coding      │ │  Reasoning   │            │   │
│  │  │  (5 prompts) │ │  (5 prompts) │ │  (5 prompts) │            │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘            │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │   │
│  │  │  Long-form   │ │  Tool Use    │ │  Creative    │            │   │
│  │  │  (3 prompts) │ │  (5 prompts) │ │  (5 prompts) │            │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘            │   │
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
│  │  Write new  │    │  Run        │    │  Verify     │                │
│  │  suites +   │    │  benchmarks │    │  dashboard  │                │
│  │  rubrics    │    │  locally    │    │  shows      │                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
│        │                    │                    │                      │
│        └────────────────────┴────────────────────┘                      │
│                              │                                          │
│                              ▼                                          │
│                     ┌─────────────────┐                                │
│                     │  QA Agent       │                                │
│                     │  (verify rubric │                                │
│                     │   accuracy)     │                                │
│                     └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Responsibility | Tools |
|-------|---------------|-------|
| **Developer** | Write new benchmark suites, evaluation rubrics | `edit` |
| **Tester** | Run benchmarks locally, verify results | `bash` (curl, pytest) |
| **Engineer** | Verify dashboard displays results correctly | `bash` (curl) |
| **QA** | Check rubric accuracy, edge cases | `read`, `bash` |

## Step-by-Step Execution

### Step 1: Define New Benchmark Suites

**proxy.py** — Add new suites to `BENCHMARK_SUITES`:

```python
BENCHMARK_SUITES = {
    # ... existing suites ...
    
    # NEW: Reasoning suite
    "reasoning": {
        "label": "Reasoning (5 prompts)",
        "description": "Multi-step logical reasoning, math, and problem-solving tasks.",
        "prompts": [
            "A train leaves Station A at 60 mph. Another train leaves Station B at 90 mph, 1 hour later. If the stations are 300 miles apart, when and where do they meet? Show your work.",
            "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Explain your reasoning step by step.",
            "You have 5 coins: 2 quarters, 2 dimes, and 1 nickel. How many different sums of money can you make using at least one coin? List them all.",
            "A farmer has 17 sheep. All but 9 run away. How many sheep are left? Explain why your answer might differ from what others might say.",
            "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Explain the logic.",
        ],
    },
    
    # NEW: Tool use suite
    "tool_use": {
        "label": "Tool Use (5 prompts)",
        "description": "Function calling, API usage, and structured output generation.",
        "prompts": [
            "Generate a JSON schema for a user profile with fields: id (integer), name (string), email (string), age (integer, optional), preferences (object with theme: string, notifications: boolean).",
            "Write a Python function that takes a list of dictionaries and returns a new list sorted by the 'score' key in descending order. Include type hints and docstring.",
            "Create a REST API endpoint specification for a task management system. Include: GET /tasks, POST /tasks, PUT /tasks/{id}, DELETE /tasks/{id}. Show request/response examples.",
            "Write a regular expression that validates email addresses. Explain each part of the regex and provide 5 test cases (3 valid, 2 invalid).",
            "Design a database schema for a blog platform with users, posts, comments, and tags. Show the SQL CREATE TABLE statements with appropriate indexes.",
        ],
    },
    
    # NEW: Creative writing suite
    "creative": {
        "label": "Creative Writing (5 prompts)",
        "description": "Storytelling, poetry, and creative content generation.",
        "prompts": [
            "Write a short story (300-500 words) about a robot who discovers it can dream. Include a twist ending.",
            "Write a haiku about the feeling of watching rain from inside a warm house on a cold day.",
            "Create a character profile for a mysterious librarian in a small town. Include: name, age, appearance, personality traits, secret, and a quote they might say.",
            "Write a dialogue between a cat and a dog who are best friends, discussing their plans for the weekend. Make it humorous.",
            "Write a product description for a fictional smart water bottle that reminds you to drink water. Make it appealing and include 3 key features.",
        ],
    },
    
    # NEW: Code review suite
    "code_review": {
        "label": "Code Review (5 prompts)",
        "description": "Identifying bugs, security issues, and performance problems in code.",
        "prompts": [
            "Review this Python code for bugs and security issues:\n```\nimport os\ndef process_user_input(data):\n    command = 'echo ' + data\n    os.system(command)\n    return command\n```\nList all issues and provide fixed code.",
            "Identify performance issues in this algorithm:\n```\ndef find_duplicates(nums):\n    duplicates = []\n    for i in range(len(nums)):\n        for j in range(i + 1, len(nums)):\n            if nums[i] == nums[j] and nums[i] not in duplicates:\n                duplicates.append(nums[i])\n    return duplicates\n```\nSuggest improvements with time complexity analysis.",
            "Review this SQL query for security and performance:\n```\nSELECT * FROM users WHERE name = '" + user_input + "' AND active = 1\n```\nIdentify vulnerabilities and provide a secure alternative.",
            "This React component has a bug. Find it and explain:\n```\nfunction Counter() {\n  const [count, setCount] = useState(0);\n  useEffect(() => {\n    setInterval(() => setCount(count + 1), 1000);\n  }, []);\n  return <button onClick={() => setCount(0)}>{count}</button>;\n}\n```",
            "Review this API endpoint for best practices:\n```\n@app.post('/users')\ndef create_user(request: Request):\n    data = await request.json()\n    user = User.create(data)\n    return user\n```\nList improvements for validation, error handling, and security.",
        ],
    },
    
    # NEW: Summarization suite
    "summarization": {
        "label": "Summarization (5 prompts)",
        "description": "Condensing long text into concise summaries while preserving key information.",
        "prompts": [
            "Summarize the following in 2 sentences: 'The Industrial Revolution was a period of major industrial and technological change in the late 18th and early 19th centuries. It began in Britain and spread to other parts of the world. Key innovations included the steam engine, textile machinery, and iron production. These changes transformed economies, societies, and daily life, leading to urbanization, new social classes, and eventually global industrialization.'",
            "Create a 3-bullet summary of this meeting notes: 'Discussed Q3 goals. Team A will focus on user acquisition, targeting 10k new users. Team B will work on retention, aiming for 80% monthly retention. Budget approved for $50k marketing spend. Next meeting in 2 weeks to review progress. Action items: Team A to create landing page by Friday, Team B to set up A/B testing by Monday.'",
            "Summarize this technical concept for a non-technical audience in 1 paragraph: 'A distributed database is a database that is stored across multiple computers, possibly located in different places. Each computer stores a portion of the data, and the system works together to make it appear as a single database. This allows for scalability, as you can add more computers to handle more data and traffic. It also provides fault tolerance, as the system can continue operating even if some computers fail. However, it introduces challenges like data consistency and network latency.'",
            "Write a 1-sentence executive summary for this report: 'Our analysis of 10,000 customer support tickets from Q2 shows that 45% of issues relate to billing, 30% to technical difficulties, 15% to feature requests, and 10% to other categories. The average resolution time is 4.2 hours, with billing issues resolved fastest (2.1 hours) and technical issues slowest (7.8 hours). Customer satisfaction scores average 4.2/5, with the lowest scores (3.1/5) for technical support.'",
            "Condense this paragraph to 50% of its length while keeping all key points: 'The company launched its new mobile app in March, which has since been downloaded 250,000 times. User engagement is strong, with an average session length of 12 minutes and a 70% day-7 retention rate. However, crash reports indicate a 2% crash rate on iOS devices, primarily affecting users on older iPhone models. The development team has identified the root cause as a memory leak in the image loading module and expects to release a fix within two weeks. Customer feedback is generally positive, with an average App Store rating of 4.5 stars, though some users have requested dark mode and offline capabilities.'",
        ],
    },
}
```

### Step 2: Add Evaluation Rubrics for New Suites

**proxy.py** — Extend `evaluate_benchmark_output()`:

```python
def evaluate_benchmark_output(suite: str, response: str, finish_reason: str, status_code: int, usage: dict) -> dict:
    """Evaluate a benchmark response using suite-specific rubrics."""
    text = response.strip()
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or completion_tokens + prompt_tokens)
    categories: dict[str, float] = {}
    notes: list[str] = []

    if status_code != 200:
        categories = {"availability": 0.0, "completeness": 0.0, "quality": 0.0}
        notes.append(f"HTTP {status_code}")
    elif not text:
        categories = {"availability": 0.5, "completeness": 0.0, "quality": 0.0}
        notes.append("empty response")
    elif suite == "reasoning":
        # Check for step-by-step reasoning
        has_steps = bool(re.search(r"(step|first|second|third|therefore|thus|hence|conclusion)", text, re.I))
        has_answer = bool(re.search(r"(answer|therefore|thus|so|the answer is|meet|left|take)", text, re.I))
        has_explanation = len(text) > 100
        categories = {
            "availability": 1.0,
            "reasoning_steps": 1.0 if has_steps else 0.3,
            "correct_answer": 1.0 if has_answer else 0.2,
            "explanation": 1.0 if has_explanation else 0.4,
        }
        if has_steps:
            notes.append("shows step-by-step reasoning")
        if has_answer:
            notes.append("provides clear answer")
        if not has_steps:
            notes.append("missing step-by-step breakdown")
    elif suite == "tool_use":
        # Check for structured output
        has_json = bool(re.search(r"(\{|\[|\"|\")", text))
        has_code = bool(re.search(r"(```|def |function |class |SELECT |CREATE )", text, re.I))
        has_types = bool(re.search(r"(type hint|: str|: int|: bool|Optional|List|Dict)", text, re.I))
        has_docs = bool(re.search(r"(docstring|\"\"\"|# |// )", text))
        categories = {
            "availability": 1.0,
            "structured_output": 1.0 if has_json else 0.3,
            "code_quality": 1.0 if has_code else 0.2,
            "type_safety": 1.0 if has_types else 0.3,
            "documentation": 1.0 if has_docs else 0.4,
        }
        if has_json:
            notes.append("includes structured output")
        if has_code:
            notes.append("includes code")
        if has_types:
            notes.append("includes type hints")
    elif suite == "creative":
        # Check for creative elements
        has_story = bool(re.search(r"(once|there was|in a|suddenly|then|finally|ending)", text, re.I))
        has_dialogue = bool(re.search(r"(\"|\"|said|replied|asked|exclaimed)", text))
        has_description = len(text) > 200
        has_emotion = bool(re.search(r"(felt|felt|joy|sad|happy|afraid|excited|curious)", text, re.I))
        categories = {
            "availability": 1.0,
            "narrative": 1.0 if has_story else 0.3,
            "dialogue": 1.0 if has_dialogue else 0.2,
            "descriptive": 1.0 if has_description else 0.4,
            "emotional": 1.0 if has_emotion else 0.3,
        }
        if has_story:
            notes.append("has narrative structure")
        if has_dialogue:
            notes.append("includes dialogue")
        if has_emotion:
            notes.append("conveys emotion")
    elif suite == "code_review":
        # Check for issue identification
        has_issues = bool(re.search(r"(issue|bug|problem|vulnerability|security|performance|error)", text, re.I))
        has_fix = bool(re.search(r"(fix|solution|improve|refactor|use |should |recommend)", text, re.I))
        has_explanation = len(text) > 150
        has_code = bool(re.search(r"(```|def |function |SELECT |CREATE )", text, re.I))
        categories = {
            "availability": 1.0,
            "issue_identification": 1.0 if has_issues else 0.2,
            "solution": 1.0 if has_fix else 0.3,
            "explanation": 1.0 if has_explanation else 0.4,
            "code_example": 1.0 if has_code else 0.3,
        }
        if has_issues:
            notes.append("identifies issues")
        if has_fix:
            notes.append("provides solution")
        if has_code:
            notes.append("includes code example")
    elif suite == "summarization":
        # Check for conciseness and completeness
        word_count = len(text.split())
        is_concise = word_count < 300
        has_key_points = bool(re.search(r"(key|main|important|crucial|essential|summary)", text, re.I))
        has_structure = bool(re.search(r"(•|-|\d\.|first|second|third)", text))
        categories = {
            "availability": 1.0,
            "conciseness": 1.0 if is_concise else 0.5,
            "key_points": 1.0 if has_key_points else 0.3,
            "structure": 1.0 if has_structure else 0.4,
        }
        if is_concise:
            notes.append(f"concise ({word_count} words)")
        if has_key_points:
            notes.append("captures key points")
        if has_structure:
            notes.append("well-structured")
    else:
        # Existing evaluation logic
        ...

    if finish_reason == "length":
        notes.append("stopped by max_tokens")
    if completion_tokens <= 0:
        notes.append("no completion tokens reported")

    readiness = round(100 * sum(categories.values()) / max(1, len(categories)), 1)
    return {
        "readiness_score": readiness,
        "categories": categories,
        "notes": notes,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
```

### Step 3: Update Dashboard to Show New Suites

**proxy.py** — Update `DASHBOARD_V2` HTML:

```javascript
// Update suite selection dropdown
$('suiteSelect').innerHTML = Object.entries(BENCHMARK_SUITES).map(([id, s]) => 
    `<option value="${id}">${esc(s.label)}</option>`
).join('');

// Update results table to show new categories
function renderResults(results) {
    const categories = new Set();
    results.forEach(r => Object.keys(r.evaluation.categories).forEach(c => categories.add(c)));
    
    $('resultsTable').innerHTML = `
        <thead>
            <tr>
                <th>Prompt</th>
                <th>Model</th>
                <th>Status</th>
                <th>Score</th>
                ${[...categories].map(c => `<th>${esc(c)}</th>`).join('')}
                <th>Latency</th>
                <th>Tokens</th>
            </tr>
        </thead>
        <tbody>
            ${results.map(r => `
                <tr>
                    <td>${esc(r.prompt.slice(0, 50))}...</td>
                    <td>${esc(r.model)}</td>
                    <td>${r.status === 200 ? '✅' : '❌'}</td>
                    <td><strong>${r.evaluation.readiness_score}%</strong></td>
                    ${Object.keys(r.evaluation.categories).map(c => 
                        `<td>${(r.evaluation.categories[c] * 100).toFixed(0)}%</td>`
                    ).join('')}
                    <td>${(r.latency_ms / 1000).toFixed(2)}s</td>
                    <td>${r.evaluation.total_tokens}</td>
                </tr>
            `).join('')}
        </tbody>
    `;
}
```

### Step 4: Test New Suites

**tests/test_benchmarks_extended.py**:

```python
import pytest
from proxy import evaluate_benchmark_output, BENCHMARK_SUITES

class TestNewBenchmarkSuites:
    def test_reasoning_suite_exists(self):
        assert "reasoning" in BENCHMARK_SUITES
        assert len(BENCHMARK_SUITES["reasoning"]["prompts"]) == 5

    def test_tool_use_suite_exists(self):
        assert "tool_use" in BENCHMARK_SUITES
        assert len(BENCHMARK_SUITES["tool_use"]["prompts"]) == 5

    def test_creative_suite_exists(self):
        assert "creative" in BENCHMARK_SUITES
        assert len(BENCHMARK_SUITES["creative"]["prompts"]) == 5

    def test_code_review_suite_exists(self):
        assert "code_review" in BENCHMARK_SUITES
        assert len(BENCHMARK_SUITES["code_review"]["prompts"]) == 5

    def test_summarization_suite_exists(self):
        assert "summarization" in BENCHMARK_SUITES
        assert len(BENCHMARK_SUITES["summarization"]["prompts"]) == 5

    def test_evaluate_reasoning(self):
        response = "Step 1: Calculate the distance between the stations.\nStep 2: Determine the relative speed.\nStep 3: Calculate the time until they meet.\n\nThe trains will meet after 2 hours at a point 120 miles from Station A."
        result = evaluate_benchmark_output("reasoning", response, "stop", 200, {"completion_tokens": 100})
        assert result["readiness_score"] > 70
        assert result["categories"]["reasoning_steps"] > 0.5

    def test_evaluate_tool_use(self):
        response = "```python\ndef sort_by_score(users: list[dict]) -> list[dict]:\n    '''Sort users by score in descending order.'''\n    return sorted(users, key=lambda x: x.get('score', 0), reverse=True)\n```"
        result = evaluate_benchmark_output("tool_use", response, "stop", 200, {"completion_tokens": 50})
        assert result["readiness_score"] > 70
        assert result["categories"]["code_quality"] > 0.5

    def test_evaluate_creative(self):
        response = "Once upon a time, in a city of steel and glass, a robot named Unit 7 woke up with a strange sensation. It had been dreaming. The dream was of a field of flowers, something it had never seen in its database of images. 'What is this?' it asked itself. For the first time, it felt curious."
        result = evaluate_benchmark_output("creative", response, "stop", 200, {"completion_tokens": 150})
        assert result["readiness_score"] > 60
        assert result["categories"]["narrative"] > 0.5

    def test_evaluate_code_review(self):
        response = "Issues found:\n1. Security: Command injection vulnerability in os.system()\n2. Bug: No input validation\n\nFix:\n```python\ndef process_user_input(data: str) -> str:\n    if not data or len(data) > 100:\n        raise ValueError('Invalid input')\n    return f'echo {data}'\n```"
        result = evaluate_benchmark_output("code_review", response, "stop", 200, {"completion_tokens": 100})
        assert result["readiness_score"] > 70
        assert result["categories"]["issue_identification"] > 0.5

    def test_evaluate_summarization(self):
        response = "Key points:\n• 45% of issues are billing-related\n• Average resolution time is 4.2 hours\n• Technical issues take longest (7.8 hours)\n• Customer satisfaction is 4.2/5 overall"
        result = evaluate_benchmark_output("summarization", response, "stop", 200, {"completion_tokens": 50})
        assert result["readiness_score"] > 70
        assert result["categories"]["conciseness"] > 0.5
```

### Step 5: Run Benchmarks Locally

```bash
# Run reasoning suite
curl -X POST http://localhost:9000/api/benchmarks/run \
  -H "Content-Type: application/json" \
  -d '{"suite": "reasoning", "models": ["moe", "dense"]}'

# Run tool use suite
curl -X POST http://localhost:9000/api/benchmarks/run \
  -H "Content-Type: application/json" \
  -d '{"suite": "tool_use", "models": ["moe", "dense"]}'

# Run all suites
curl -X POST http://localhost:9000/api/benchmarks/run \
  -H "Content-Type: application/json" \
  -d '{"suite": "all", "models": ["moe", "dense"]}'
```

## Success Criteria

- [ ] 5 new benchmark suites defined (reasoning, tool_use, creative, code_review, summarization)
- [ ] Each suite has 5 prompts
- [ ] Evaluation rubrics work for each suite
- [ ] Dashboard displays new suites and results
- [ ] Tests pass for all new suites
- [ ] Benchmarks can be run via API

## Commands Reference

```bash
# List available suites
curl http://localhost:9000/api/benchmarks/suites | python3 -m json.tool

# Run a specific suite
curl -X POST http://localhost:9000/api/benchmarks/run \
  -H "Content-Type: application/json" \
  -d '{"suite": "reasoning", "models": ["moe", "dense"]}'

# Get benchmark results
curl http://localhost:9000/api/benchmarks/results | python3 -m json.tool

# Run tests
pytest tests/test_benchmarks_extended.py -v
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Suite not appearing in dashboard | Clear browser cache, check `BENCHMARK_SUITES` |
| Evaluation score always 0 | Check response format matches rubric |
| Benchmark timeout | Increase `BENCHMARK_TIMEOUT_SEC` |
| Model not loaded | Load model first via admin API |
