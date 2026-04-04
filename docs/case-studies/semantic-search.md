# Semantic Search Without a Vector Database

Most websites rely on keyword search — PostgreSQL full-text search, Elasticsearch, or similar. These work for exact terms but miss intent: a search for "something that works in the browser" won't match a program described as "runs client-side via WebAssembly." Adding semantic understanding normally requires vector databases, embedding pipelines, and external services. PAW lets you add intent-aware reranking on top of your existing search — no infrastructure changes, no embeddings, no external dependencies.

**Try it live:** Search at [programasweights.com/hub](https://programasweights.com/hub) — try queries like "something that works in the browser" or "not a classifier."

## How we built it

### Attempt 1: Numeric relevance scoring (1–10)

The obvious approach: compile a scorer that rates each search result on a 1–10 scale.

```
Rate the relevance of this search result to the query on a scale of 1-10.
Query: "counting tasks"
```

**Result:** The model clustered everything at 8–10. A search for "counting tasks" gave 9/10 to a sentiment classifier because it vaguely involved processing text. No discrimination.

**Lesson:** Small models can't produce fine-grained numeric scores. They don't have a calibrated sense of what "7 vs 8" means.

### Attempt 2: Binary yes/no

Simplify to binary: is this result relevant or not?

```
Is this search result relevant to the query? Return YES or NO.
```

**Result:** Better discrimination, but couldn't distinguish "perfect match" from "vaguely related." Everything relevant got YES, so the top results were still unordered.

**Lesson:** Binary lacks granularity. You need at least 3–4 buckets to produce a meaningful ranking.

### Attempt 3: Discrete categories

Use 3–4 named categories that map to sort order:

```
Rate how well the candidate matches the query.
Return ONLY one of: exact_match, highly_relevant, somewhat_relevant, not_relevant
```

**Result:** Much better. The model could reliably distinguish "this is exactly what they asked for" from "this is tangentially related." Mapping `exact_match=3, highly_relevant=2, somewhat_relevant=1, not_relevant=0` produced clean rankings.

**Lesson:** Discrete named categories work where numeric scales fail. The names give the model clear semantic anchors.

### Attempt 4: Explicit exclusion rules

A search for "not a classifier" still returned classifiers ranked highly. The model ignored the negation.

**Fix:** Add an explicit rule for exclusions in the spec:

```
If the query excludes something, those candidates are not_relevant.
```

**Result:** Negation queries started working. "Not a classifier" correctly demoted classifiers.

**Lesson:** State rules explicitly. Don't assume the model infers constraints from the query — spell them out in the spec.

## The solution

### Spec template

```python
SCORER_SPEC = """
You are a search matcher. Rate how well the candidate matches the query.
Match all constraints: {constraint_types}.
If the query excludes something, those candidates are not_relevant.

Query: "{query}"

Return ONLY one of: exact_match, highly_relevant, somewhat_relevant, not_relevant
"""

scorer = paw.compile(SCORER_SPEC.format(
    constraint_types="topic, author, and category",
    query="counting tasks by da03",
))
```

Replace `constraint_types` with whatever metadata your search results have (topic, author, date, category, price range, etc.).

### Architecture

```
User query
    │
    ▼
Keyword search (FTS / Elasticsearch / etc.)
    │ returns top N candidates
    ▼
Compile PAW scorer for this query (cached by query text)
    │
    ▼
Score each candidate ──► exact_match / highly_relevant / somewhat / not_relevant
    │                         │
    ▼                         ▼
Map to integers          Sort descending
    │
    ▼
Return reranked results
```

### Candidate formatting

Each candidate is passed to the scorer as a text block. Include the fields that matter for your domain:

```python
SCORE_MAP = {"exact_match": 3, "highly_relevant": 2, "somewhat_relevant": 1, "not_relevant": 0}

def rerank(query: str, candidates: list[dict]) -> list[dict]:
    scorer = paw.compile_and_load(SCORER_SPEC.format(
        constraint_types="topic, author, and category",
        query=query,
    ))

    scored = []
    for c in candidates:
        text = f"Name: {c['name']}\nAuthor: {c['author']}\nDescription: {c['description']}"
        label = scorer(text)
        scored.append((SCORE_MAP.get(label, 0), c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]
```

### Progressive UX

Show keyword results immediately. If the server isn't busy, fire the reranking in the background and update the results once scoring completes. Users see instant results that get refined — no loading spinner.

## Adapting this for your site

1. **Start with your existing search.** PAW reranking is a layer on top — it doesn't replace your keyword search.
2. **Pick your constraint types.** What metadata do your results have? Topic, author, price, date, location?
3. **Compile a scorer** with the template above, replacing `constraint_types` and `query`.
4. **Score your top 10–20 results.** Don't score everything — just the candidates that keyword search already found promising.
5. **Build a test set** of 10–20 queries with known-good rankings. Iterate on the spec until it discriminates well.
6. The scorer is cached by query text — repeated searches are instant.

## Takeaways

- **Discrete categories beat numeric scales** for small models. Use 3–4 named buckets.
- **State constraints and exclusions explicitly** in the spec. The model won't infer them from the query alone.
- **Rerank, don't replace.** Layer PAW scoring on top of existing keyword search for the best of both worlds.
- **Test with adversarial queries** — negations, ambiguous terms, author-specific searches — to find spec weaknesses early.
