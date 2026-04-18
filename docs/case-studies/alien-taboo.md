# Alien Taboo: A Word-Guessing Game on a 0.6B Interpreter

Most of our case studies are about *infrastructure* — log triage, search reranking, agent routing. This one is a game. We wanted to show that a tiny PAW program is enough to power a real, playable, in-browser product, not just internal tooling.

**Try it live:** [programasweights.com/alien](https://programasweights.com/alien) (English) or [programasweights.com/alien-cn](https://programasweights.com/alien-cn) (中文). You describe a secret word, the alien guesses. The alien is one compiled PAW program running on a 0.6B-parameter interpreter — no GPT-4, no API roulette, just one .paw file and a Qwen3 0.6B base model.

## Why this is interesting for PAW

Word-guessing is a textbook *fuzzy function*: easy to describe in one sentence, impossible to write as rules. "Given a player's hint, output the secret word." A keyword matcher would never bridge "fluffy thing that purrs" → `cat`. A retrieval system would force you to enumerate every possible hint. A 32B prompted LLM would work but you can't ship one to a browser-tab user for free.

A 0.6B interpreter with a focused PAW spec is exactly the right size: small enough to run on a single inference server with no per-request cost, expressive enough to recognise the dozens of ways a child might describe "watermelon."

## How we built it

### Attempt 1: One huge spec listing every word

The first instinct was to bake the whole word list into the spec — "the secret word is one of: cat, dog, banana, …" — so the model only ever picks from a known vocabulary.

**Result:** The spec ballooned to thousands of tokens, the compiler timed out, and accuracy *dropped*: the model started biasing toward whichever words appeared earliest in the list regardless of the description.

**Lesson:** Don't dump your domain into the spec. PAW programs work best when the spec teaches a *behaviour* with a few examples, not when it tries to be a database.

### Attempt 2: Free-form generation with a few examples

We replaced the word list with a short instruction and ~15 input/output exemplars covering the diversity of hints (animals, foods, household objects, abstract concepts, multi-word objects).

```
You are playing a word-guessing game. The user describes a common English
word in their own words without saying the word itself. Your job is to
guess the word from the description.

Return ONLY the single word being described. Lowercase. No punctuation,
no explanation, no extra words.

Input: furry animal that meows and purrs
Output: cat

Input: yellow curved fruit, monkeys like it
Output: banana

Input: thing you use to unlock a door, metal, small, has teeth
Output: key

... (≈12 more)
```

**Result:** Compiled cleanly, ran in ~80 ms per guess, and produced sensible answers for almost every test description. This is the version shipping today.

**Lesson:** This is the opposite of what we found in our [site navigation](site-navigation.md) and [semantic search](semantic-search.md) studies — there, generation hallucinated and we reframed as classification. Here, generation works because (a) the answer space is implicit (any English noun) so we couldn't classify even if we wanted to, and (b) the answer is short and over-determined by the input. **Generation works when the output is short and the input is rich.**

### Attempt 3: The actual hard part — curating the word bank

With the alien working on the first realistic hint we threw at it, we expected to ship in an afternoon. Instead we spent days on word selection. The reason:

- **Dead words.** "Sundial" is a word every adult knows, but no one can describe it without saying "sun" or "shadow" or "time," all of which produce the wrong guess. A word that never gets solved isn't unfun for the alien — it's unfun for the *player*. Every dead word in the bank is a session-killing dead end.
- **Fairness.** "Bullfrog" passes the alien's accuracy test but stumps anyone under 12. Mass-appeal games need words a 6-year-old or non-native speaker can describe.
- **Variety.** A 50-word bank gets boring after two sessions. We wanted ≥300 viable words.

The vetting process became its own pipeline:

1. **Generate candidates** with GPT-5.4 (~4000 raw words across 40 themes).
2. **Simulate playthroughs** — for each candidate, prompt GPT-5.4-mini to play the role of a human describing the word; route those descriptions through the actual deployed alien program; keep words solved within ≤8 rounds across ≥4 of 5 random-seed trials.
3. **Filter for commonness** with `wordfreq` (Zipf ≥ 5.0) so kids and ESL players have a fair shot.
4. **Manual pass** by a human (us) on the survivors to remove anything ambiguous, edgy, or culturally narrow.

The final bank is 361 ultra-common English words. The full vetting script lives at [`server/scripts/vet_alien_words.py`](https://github.com/programasweights/website/blob/main/server/scripts/vet_alien_words.py).

**Lesson:** **The PAW program is usually the easy part of the product.** Treat it as one ingredient. The data, the UI, and the fairness work around the program are typically what take the time.

### The multilingual finale: a Chinese version with one new file

After the English game shipped, we wanted a Chinese version. The intuition was that this would mean fine-tuning a Chinese-specific compiler, retraining the 0.6B interpreter on a Chinese corpus, or at minimum prompt-engineering around tokeniser quirks.

What we actually did:

1. Wrote a new spec — `spec_cn.txt` — in Mandarin, with 20 Chinese hint→word examples.
2. Compiled it with the same `paw-4b-qwen3-0.6b` compiler.
3. Pointed a sister `/api/v1/alien-cn/guess` endpoint at the new program ID.

```python
import programasweights as paw

with open("spec_cn.txt", encoding="utf-8") as f:
    spec = f.read()

program = paw.compile(spec, compiler="paw-4b-qwen3-0.6b")
# That's it. program.id is now ready to serve Chinese players.
```

The first compile produced **67.5% per-description accuracy** and **80% at-least-one-of-two accuracy** on a held-out 20-word Chinese test set, comparable to the English program's accuracy on its English test set. We changed exactly zero lines in the SDK or interpreter.

**Lesson:** **Spec language is just data.** PAW's compiler doesn't carry English-specific assumptions; the only language-aware things in the system are (a) your spec, and (b) any output post-processing your client does (we had to extend the server's "first-word extractor" from `[a-z]+` to also accept `[\u4e00-\u9fff]+`). For a multilingual product, this is several orders of magnitude less work than maintaining a separate stack per language.

## The solution at a glance

The entire alien — both languages — is one PAW function per language plus thin glue:

```python
import programasweights as paw

alien_en = paw.function(EN_PROGRAM_ID)   # compiled from spec.txt
alien_cn = paw.function(CN_PROGRAM_ID)   # compiled from spec_cn.txt

def guess(description: str, lang: str) -> str:
    fn = alien_cn if lang == "zh" else alien_en
    return fn(description).strip().lower()
```

Everything else — the timer, the lives, the share card, the leaderboard — is plain product code. No model orchestration, no prompt-management framework, no second LLM behind it.

## Takeaways

- **Generation is the right tool when the output is short and the input is rich.** Classification wins for routing; generation wins for "name this thing." Pick the right one for your task.
- **The PAW program is usually the easy part.** Most product time goes into data quality, UX, and edge cases — exactly as it would with any function in your codebase.
- **Use small models to vet content for small models.** Running GPT-5.4-mini in a multi-trial simulation against your deployed PAW program is faster and cheaper than human playtesting, and it scales to thousands of candidates overnight.
- **Spec language is just data.** A new spec in a new language is a new product, with no SDK or interpreter changes required.
- **A 0.6B interpreter is enough** for tasks where the input over-determines the output. You don't always need 70B; you need the right factoring.
