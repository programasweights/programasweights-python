# Feedback and Cases

After you run inference on a program in the web app, you can leave structured feedback so the community understands how the model behaves in practice.

## Submitting feedback

Use the controls shown after a run:

- **Thumbs up** — mark a **success**: the output matched what you needed.
- **Thumbs down** — mark a **limitation**: the program failed or behaved poorly for your use case.

Feedback is optional but encouraged when something notable happens.

## Cases

**Cases** are attached to the program and are **visible to other users** browsing or opening that program. They document real-world outcomes instead of only the static specification.

### Case types

| Type | Typical use |
|------|-------------|
| `success` | The program handled the input well |
| `limitation` | Clear failure mode or systematic weakness |
| `edge_case` | Unusual input that exposes interesting behavior |

You may add a **short description** with any case to explain context, input characteristics, or what you expected versus what you got.

## Why cases matter

Cases help everyone see **what works**, **what breaks**, and **where behavior is subtle**. They complement the written spec with evidence from actual runs.

## Authentication

You must **sign in with GitHub** to submit cases. Anonymous users can run programs where allowed but cannot record feedback as an identified contributor.
