# Smart Research Lab

Smart Research Lab is a topic-agnostic research-paper generation pipeline for
AI and intelligent-system projects. It accepts a free-form topic, builds a
structured research plan, generates an academic manuscript, audits claims and
formulas, and exports a PDF together with reproducibility artifacts.

The project is designed for fast demonstrations and research planning. Unless
the benchmark is backed by an instrumented experiment, generated latency,
memory, training curves, and quality values are **controlled planning
simulation estimates**, not measured scientific results.

## Features

- Topic normalization for English, Vietnamese, mixed-language, short, and
  LaTeX/XML-like inputs.
- Topic-aware routing with a dynamic fallback for unfamiliar domains.
- Multi-stage pipeline:
  `Planner -> Researcher -> Writer -> Guardrail -> Reviewer`.
- Explicit **Research Gap** and **Proposed Method** sections in every
  manuscript.
- Evidence labels for measured, literature, simulated, and planned values.
- Domain-adaptive datasets, baselines, metrics, benchmark tables, and ablations.
- Academic English manuscript structure with a maximum of four core equations.
- PDF generation with ReportLab and formula rendering through Matplotlib.
- IEEE-style LaTeX artifact, benchmark JSON, guardrail JSON, charts, and ZIP.
- Streamlit interface with Vietnamese controls and English manuscript content.
- PDF preview rendered as page images to avoid local-file iframe restrictions.
- Optional PDF opening in a new browser tab through a JavaScript Blob URL.

## Architecture

```text
User topic
    |
    v
academic_sanitizer.py
    |
    v
FastAPI: backend/main.py
    |
    +--> Planner: domain, research question, hypothesis, baselines
    +--> Researcher: datasets, benchmark matrix, ablations, evidence labels
    +--> Writer: abstract, gap, proposed method, methodology, results
    +--> Guardrail: formula, benchmark, and evidence consistency checks
    +--> Reviewer: novelty, rigor, clarity, weaknesses, revisions
    |
    v
Streamlit: app.py
    |
    +--> PDF preview and browser viewer
    +--> Benchmark tables and charts
    +--> PDF / LaTeX / JSON / ZIP downloads
```

### Runtime modes

| Mode | Requirement | Behavior |
|---|---|---|
| Dynamic fallback | No API key | Generates a deterministic topic-aware research draft and controlled planning estimates |
| Live Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Uses Gemini for planning and writing; guardrails still validate the response |

Live mode does not automatically make benchmark numbers measured. Real
measurements must come from an instrumented experiment and must retain the
appropriate evidence label.

## Requirements

- Python 3.11 or newer
- Windows, Linux, or macOS
- Recommended: a virtual environment
- Optional Gemini API key for live generation

## Installation

### Windows PowerShell

```powershell
git clone https://github.com/trducc/AI-automatically-writes-scientific-papers.git
Set-Location AI-automatically-writes-scientific-papers

py -3.11 -m venv env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the project with the interpreter directly:

```powershell
.\env\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python3.11 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

The API key is optional. Without it, the application uses the dynamic fallback
mode.

```powershell
$env:GEMINI_API_KEY = "your-key"
```

Do not commit API keys, `.env` files, virtual environments, generated PDFs, or
local IDE settings.

## Run the application

Start the FastAPI backend first:

```powershell
.\env\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

In another terminal, start Streamlit:

```powershell
.\env\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

Open <http://localhost:8501>.

The backend health endpoint is available at
<http://localhost:8000/health>. The interactive API schema is available at
<http://localhost:8000/docs>.

## API example

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/pipeline/run `
  -ContentType "application/json" `
  -Body '{"topic":"Adaptive image classification on edge devices"}'
```

The request supports:

```json
{
  "topic": "Research topic",
  "domain": "Optional domain",
  "objective": "Optional research objective",
  "constraints": "Optional constraints",
  "api_key": "Optional per-request key",
  "require_verified_data": false
}
```

## Generated artifacts

After a successful run, the Streamlit session exposes:

| Artifact | Description |
|---|---|
| `paper.pdf` | Rendered academic manuscript |
| `paper.tex` | IEEE-style LaTeX source |
| `benchmark_matrix.json` | Models, metrics, latency, memory, and evidence labels |
| `guardrail_report.json` | Formula and benchmark audit output |
| `charts/` | Benchmark, ablation, and training-curve PNG files |
| `artifacts.zip` | Packaged output files and charts |

The manuscript includes Abstract, Introduction, Research Gap, Related Work,
Proposed Methodology, Experiments, Discussion, Conclusion, and References.

## Validation and tests

Compile the main application:

```powershell
.\env\Scripts\python.exe -m py_compile app.py backend\main.py
```

Run the five-topic end-to-end stress test:

```powershell
.\env\Scripts\python.exe test_pipeline_stress.py
```

Run adversarial topic coverage:

```powershell
.\env\Scripts\python.exe test_adversarial_topics.py
```

The adversarial suite covers short topics, Vietnamese input, mixed language,
LaTeX/XML-like characters, symbols, and unusual domains.

The repository also contains a GitHub Actions smoke workflow at
`.github/workflows/smoke_test.yml`.

## Project layout

```text
app.py                       Streamlit UI and PDF/artifact export
academic_sanitizer.py       Topic normalization and safety cleanup
chart_generator.py           Benchmark and ablation charts
backend/main.py              FastAPI routes and multi-agent orchestration
backend/schemas.py           Typed Pydantic pipeline contracts
backend/guardrail_agent.py   Manuscript and evidence audit
test_pipeline_stress.py      Five-topic E2E artifact test
test_adversarial_topics.py   Ten adversarial-topic test
data/                        Optional local datasets
results/                     Generated experiment artifacts
docs/                        Project documentation assets
```

## Evidence and responsible use

The system can generate a complete research proposal for an unfamiliar topic,
but it cannot guarantee that the proposal is scientifically validated. Before
submission or publication:

1. Run the proposed experiment on real data.
2. Record seeds, hardware, software versions, and measurement protocol.
3. Replace planned or simulated values only with reproducible measurements.
4. Verify every citation against a canonical paper or publisher page.
5. Review safety, ethics, privacy, and domain-specific requirements.

The generated manuscript is a starting point for research, not a substitute
for peer review or experimental validation.

## License

No license file is currently declared in this repository. Treat the code as
all-rights-reserved until a project maintainer adds an explicit license.
