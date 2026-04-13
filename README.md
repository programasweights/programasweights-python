# ProgramAsWeights

Programs as small weight blobs; a fixed interpreter runs them.

## Development

```
pip install -e .
```

```
python test_compile.py
```

```
python test_execute.py
```


## Quickstart

```bash
pip install programasweights
```

```python
import programasweights
f = programasweights.function("/path/to/weights")
print(f("Hello"))
```

- weights = programs; base model = interpreter.
- Deterministic by default (greedy generation).

## 🌐 Web Interface

ProgramAsWeights includes a modern web interface for easy interaction with the system. The web app provides a user-friendly way to compile specifications, test neural programs, and manage models without writing code.

### Features

- **Interactive Model Selection**: Choose from available compiler and interpreter models
- **Natural Language Specifications**: Describe your program in plain English
- **GPT-Powered Examples**: Auto-generate test data using OpenAI's API
- **Real-time Compilation**: Compile specs into neural programs instantly
- **Interactive Testing**: Test compiled programs with custom inputs
- **Model Downloads**: Download compiled models as `.tgz` files
- **Community Sharing**: Publish and discover programs on the leaderboard

### Quick Start

1. **Prerequisites**: Ensure you have Python 3.8+, Node.js 16+, and npm installed

2. **Start the application**:
   ```bash
   cd web-app
   ./start.sh
   ```

3. **Access the interface**:
   - **Web App**: http://localhost:5173
   - **API Docs**: http://localhost:8000/docs

The startup script automatically:
- Installs Python and Node.js dependencies
- Starts the FastAPI backend server
- Starts the React frontend development server

### Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Backend
cd web-app/backend
pip install -r requirements.txt
python run_server.py

# Frontend (in another terminal)
cd web-app/frontend  
npm install
npm run dev
```

### Configuration

Create `web-app/backend/.env` with:
```bash
OPENAI_API_KEY=your_openai_api_key_here  # Optional: for GPT test generation
CHECKPOINT_DIR=../../outputs_1spec/prefix_kv  # Path to trained models
```

### Example Workflow

1. Select "Qwen 2.5 Coder 0.5B" as compiler and "PAW Interpreter" as interpreter
2. Enter specification: *"Parse a string like '(A) cat (B) dog' into a JSON list"*
3. Generate examples with GPT or add manually
4. Click "Compile" to create your neural program
5. Test with input: *"(A) red apple (B) green banana"*
6. Download or publish your compiled model

For detailed documentation, see [`web-app/README.md`](web-app/README.md).

## Train a compiler→KV prefix→interpreter (defaults)

```bash
# installs training extras
pip install -e .[train]

# trains on data/samples_train.jsonl with Qwen defaults
paw-train
```

- Produces checkpoint under `outputs/prefix_kv/checkpoint/`.

## Compile then run (qualitative eval)

```bash
# compile a prompt into a KV-prefix artifact
paw-eval
```

Or programmatically:

```python
import programasweights as paw

prompt = (
    "Parse a string like '(A) ... (B) ... (C) ...' into a JSON list of options. "
    "Be robust to noise: extra spaces, bullets, and phrases like 'both (A) and (B)'."
)

artifact_dir = paw.compile(
    out_dir="outputs/prefix_kv/demo_program",
    spec=prompt,
    checkpoint_dir="outputs/prefix_kv/checkpoint",
)

f = paw.function(artifact_dir, interpreter_name="Qwen/Qwen2.5-Coder-0.5B-Instruct", max_new_tokens=128)
print(f("(A) cat  (B) dog  (C) both (A) and (B) are possible"))
```

- The dummy compiler is still available as `compile_dummy` for tests and demos.

## API

```python
import programasweights
parse_func = programasweights.function(
    "/path/to/weights.safetensors",
    interpreter_name="google/flan-t5-small",
    max_new_tokens=128,
)
output = parse_func("input string")
```

- Accepts `str` or `List[str]` and returns the same shape.
- Aliasing works: `import programasweights as paw`.

## Local development

```bash
pip install -e .[test]
pytest -q
```

```bash
# smoke check
python -c "import programasweights as paw; print(paw.__version__)"
```

- `pip install -e .` installs in editable mode so code changes are picked up without reinstalling.
- If you do not need editable mode: `pip install .` (you must reinstall after changes).

## Notes

- MVP runtime uses a single in-process interpreter (loads the base model once and stays warm).
- Program artifact can be a prompt/prefix (text file) or a KV-prefix directory with `program.json` and `kv_prefix.pt`.
- Device selection: CUDA if available, else CPU. Override with env var `PROGRAMASWEIGHTS_DEVICE`.
- Simple global lock around `generate()` for thread safety.

## Roadmap

- Compiler: spec → weights
- LoRA support
- JSON-constrained decoding
- Server mode (multi-tenant)
