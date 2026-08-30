"""
Reflection Phrase Extractor - v5.2 (targeted re-run for Weeks 4-7)
====================================================================
WHY THIS FILE EXISTS
---------------------
In the v5.1 run, Weeks 4, 5, 6 and 7 came back at ~97-100% "Missing" on
EVERY component, including "Description of event" - which is Present
94-98% of the time in every other week (1, 2, 3, 8). That pattern (every
single component collapsing to Missing at once, description included)
is the signature of extract_phrases() getting text it couldn't work
with - either:
  (a) entry["reflection"] was empty/near-empty for those weeks in
      reflections.json (extract_phrases() returns all-empty automatically
      if len(reflection_text.strip()) < 50), or
  (b) Ollama was failing/timing out/returning malformed JSON on those
      specific calls and silently falling back to the empty-dict path
      in extract_phrases()'s except-blocks.
This script does NOT change any extraction logic, prompts, thresholds,
or scoring rules from v5.1 - it is byte-for-byte the same pipeline. It
only changes WHICH reflections get processed and HOW results are merged.

WHAT THIS SCRIPT DOES
-----------------------
1. Loads the existing v5_output/phrases_output.json (your v5.1 results).
2. Loads reflections.json fresh from disk.
3. Prints a quick diagnostic: for every Week 4/5/6/7 entry, the raw
   character length of entry["reflection"]. If these are mostly < 50
   chars, the bug is upstream in the reflections.json extraction step
   (step A), not in this script or the v5.1 pipeline - re-running
   extraction here will not fix that, it'll just confirm empty input
   in fabrication/judge logs are correctly seeing nothing to work with.
4. Re-runs extract_phrases() -> verify_phrases_against_text() ->
   judge_phrases() -> score_from_phrases() -> normalization -> gating
   for ONLY the entries whose week is one of:
       "Week 4 Reflection", "Week 5 Reflection",
       "Week 6 Reflection", "Week 7 Reflection"
5. Splices those updated entries back into the full results list,
   keyed on (filename + week), REPLACING the old (broken) entry.
   Weeks 1, 2, 3, and 8 are left completely untouched - their entries
   are carried over as-is from the existing phrases_output.json.
6. Re-saves phrases_output.json (now a mix of untouched weeks 1/2/3/8
   plus freshly re-extracted weeks 4/5/6/7).
7. Rebuilds summary_stats.json from scratch over the FULL, now-corrected
   dataset (same stats logic as v5.1's main()) - so overall totals,
   by-week breakdowns, topic analysis, emotional trend, and profile
   distribution are all recalculated, not just the week 4-7 slice.
8. Appends any new fabrication/soft-match/judge log entries from this
   run onto the existing logs (old entries are kept; nothing from
   weeks 1/2/3/8 is touched or duplicated).

USAGE
-----
    ollama serve                      # if not already running
    python update_weeks_4_7.py

Reads:  v5_output/phrases_output.json (must already exist from v5.1 run)
        ../A/JFC 25 26 Reflections/reflections.json
Writes: v5_output/phrases_output.json   (updated in place)
        v5_output/summary_stats.json    (fully recalculated)
        v5_output/fabrication_log.json  (appended)
        v5_output/soft_match_log.json   (appended)
        v5_output/judge_log.json        (appended)
"""

import json
import re
import time
import inspect
from collections import Counter
from pathlib import Path

try:
    import ollama
except ImportError:
    print("Run: pip install ollama")
    exit(1)

# ── Config (identical to v5.1) ──────────────────────────────────────────────
REFLECTIONS_JSON = Path("../A/JFC 25 26 Reflections/reflections.json")

OUTPUT_DIR       = Path("v5_output")
OUTPUT_FILE      = OUTPUT_DIR / "phrases_output.json"
STATS_FILE       = OUTPUT_DIR / "summary_stats.json"
FABRICATION_FILE = OUTPUT_DIR / "fabrication_log.json"
SOFT_MATCH_FILE  = OUTPUT_DIR / "soft_match_log.json"
JUDGE_FILE       = OUTPUT_DIR / "judge_log.json"
MODEL            = "qwen2.5:7b"
MAX_RETRIES      = 3
DELAY_SECS       = 0.3

ENABLE_LLM_JUDGE = True
JUDGE_MODEL      = "qwen3:14b"

ENABLE_LENGTH_NORMALIZATION = True

ENABLE_HIERARCHY_GATE = True
FOUNDATION_COMPONENTS  = ["description", "evaluation"]
HIGHER_COMPONENTS      = ["alternative", "future_action", "depth"]

JUDGE_COMPONENTS = ["evaluation", "analysis", "alternative", "future_action", "depth"]

# ── THE ONLY REAL BEHAVIOURAL CHANGE IN THIS FILE ───────────────────────────
# Only these weeks get re-extracted. Everything else passes through untouched.
TARGET_WEEKS = {
    "Week 4 Reflection",
    "Week 5 Reflection",
    "Week 6 Reflection",
    "Week 7 Reflection",
}

CONCISE_JUDGE_DEFINITIONS = {
    "evaluation": "A judgement about something SPECIFIC/nameable being effective, ineffective, a mistake, or a success. NOT generic filler ('going fine', 'pretty typical') with no concrete target. NOT the 'because Y' clause of a sentence (that's analysis).",
    "analysis": "Explanation of WHY something happened or what it means, tied to a specific incident. NOT a bare judgement with no cause (that's evaluation). NOT an identity/self-concept shift (that's depth).",
    "alternative": "Consideration of another perspective, approach, or how the situation could have unfolded differently (e.g. another person's viewpoint, a different strategy). NOT simply 'what I'll do next time' (that's future_action).",
    "future_action": "A concrete, specific, nameable planned behaviour change. NOT vague ('try to do better') or continuing current behaviour unchanged ('keep doing what I've been doing', 'see how things go').",
    "depth": "A genuine shift in how the student sees themselves as a professional/engineer/teammate - identity, values, or mindset. NOT a surface-level lesson or action plan (that's analysis/future_action).",
}

TOPIC_CHOICES = [
    "teamwork", "time_management", "communication", "technical_skills",
    "client_management", "professional_conduct", "conflict_resolution",
    "personal_growth", "project_planning", "problem_solving", "leadership", "other",
]

_CHAT_SUPPORTS_THINK = "think" in inspect.signature(ollama.chat).parameters

FABRICATION_LOG_NEW = []
SOFT_MATCH_LOG_NEW = []
JUDGE_LOG_NEW = []

JFC_TIMELINE = {
    "Week 1 Reflection":  {"dates": "1-5 Dec 2025",  "note": "Program induction week. First client meetings. Students orienting to JFC environment."},
    "Week 2 Reflection":  {"dates": "8-12 Dec 2025",  "note": "Early project phase. Teams establishing working norms and beginning technical work."},
    "Week 3 Reflection":  {"dates": "15-19 Dec 2025", "note": "Final week before Christmas break. Pre-holiday period - likely lower effort and engagement."},
    "Week 4 Reflection":  {"dates": "22 Dec 2025 - 5 Jan 2026", "note": "CHRISTMAS BREAK (22 Dec - 6 Jan). Week 4 submitted during or just after break. Expect lower quality and shorter submissions."},
    "Week 5 Reflection":  {"dates": "6-10 Jan 2026",  "note": "Return from break. Re-engagement phase. Students resuming project momentum."},
    "Week 6 Reflection":  {"dates": "13-17 Jan 2026", "note": "Mid-program. Projects in active delivery phase. Increased technical complexity."},
    "Week 7 Reflection":  {"dates": "20-24 Jan 2026", "note": "Late program. Students approaching final deliverables. Increased workload pressure."},
    "Week 8 reflection":  {"dates": "27 Jan - 13 Feb 2026", "note": "Final program week. Program conclusion, final presentations. Students may reflect more deeply on overall experience."},
}

COMPONENT_DEFINITIONS = """
You are analysing student professional development reflections written during an engineering internship.

Extract exact phrases (direct word-for-word quotes) from the reflection that match each component below.
These components are grounded in established reflective frameworks (Gibbs 1988, Moon 2004, Kolb 1984, Rolfe et al. 2001, Schön 1983).

CRITICAL RULE - NO FABRICATION: Every phrase you extract MUST be an exact, word-for-word
substring of the reflection text provided below. Do not paraphrase. Do not combine words from
different sentences. Do not invent a phrase that "sounds like" something the student might have
meant. If you are not certain a phrase appears verbatim in the text, do not include it.

COMPONENT DEFINITIONS:

1. DESCRIPTION
   What it is: A factual account of what happened — the event, task, situation, or experience.
   Grounded in: Gibbs Stage 1 (Description), Kolb Stage 1 (Concrete Experience), Rolfe "What?"
   Look for: Sentences describing tasks completed, meetings attended, problems encountered, or project activities.
   Example phrases: "This week we completed the project charter", "I attended the client meeting", "I was tasked with researching..."
   NOT this: Opinions, feelings, or evaluations — just factual accounts. Also NOT this: vague
   temporal filler with no specific referent, e.g. "this week was another one where I just tried
   to keep things moving along", "it's been a fairly normal week overall" — these name no
   specific task, meeting, or event and should NOT be extracted as description.

2. EMOTIONAL
   What it is: Explicit acknowledgement of feelings or emotional states, ideally with some exploration of why.
   Grounded in: Gibbs Stage 2 (Feelings), Boud, Keogh & Walker (1985) "attending to feelings",
   Carless & Boud (managing affect in feedback literacy), Moon (emotional elements promote long-term behaviour change)
   Look for: Named emotions AND some context or cause. More than just a single word.
   Example phrases: "I felt frustrated because the requirements kept changing", "I was anxious about presenting but became more confident as the week progressed"
   NOT this: Single emotion words with no context e.g. just "I felt happy" with nothing else.

3. EVALUATION
   What it is: A judgement about what worked well or did not work — distinguishing positive from negative outcomes.
   Grounded in: Gibbs Stage 3 (Evaluation), Rolfe "So What?" — significance of the event, Boud
   Keogh & Walker (1985) "Validation"
   Look for: Sentences explicitly identifying something SPECIFIC as effective, ineffective, a
   mistake, a success, or something that could have been better. The thing being judged must be
   nameable — you should be able to say exactly what is being evaluated.
   Example phrases: "Something that could have been handled better was...", "The daily stand-ups worked well because...", "We should have clarified requirements earlier"
   NOT this: Simple descriptions of what happened without a judgement attached. ALSO NOT THIS -
   generic, non-specific filler assessments with no identifiable target, e.g. "things are going
   fine", "progressing at a reasonable pace", "pretty typical", "a fairly standard week, nothing
   really stands out" — these sound evaluative but judge nothing concrete and must NOT be extracted.
   ALSO NOT THIS - if a sentence contains BOTH a judgement AND an explanation of why (e.g.
   "X went wrong because Y"), only extract the judging clause here ("X went wrong"); the
   explanatory "because Y" clause belongs under ANALYSIS instead, not here. In particular,
   any sentence structured as "The reason I did X was that Y" or "I made that call because Y"
   is ANALYSIS, not evaluation - it is explaining a cause, not judging an outcome as good or
   bad. Do NOT extract "reason...was" sentences into EVALUATION.

4. ANALYSIS
   What it is: Explanation of WHY something happened, what it means, or what insight was gained from the experience.
   Grounded in: Gibbs Stage 4 (Analysis), Kolb Stage 3 (Abstract Conceptualisation), Schön
   Reflection-on-Action, Moon deeper levels of reflection, Boud Keogh & Walker (1985)
   "Association"/"Integration"
   Look for: Causal reasoning, lessons learned, realisations about underlying causes, tied to
   THIS SPECIFIC incident.
   Example phrases: "I realised that the confusion arose because we hadn't aligned on scope", "This showed me that early communication prevents rework", "Looking back, the reason this failed was...", "The reason I made that call was that I was optimising for finishing the visible deliverable"
   NOT this: Descriptions or evaluations without explanation of cause or meaning. ALSO NOT THIS -
   do not extract a sentence here if it is really a judgement with no causal clause (that belongs
   in EVALUATION instead). ALSO CRITICAL - do not extract identity-level or self-concept
   statements here (e.g. "this changed how I see my role as an engineer", "I used to think X but
   now I realise Y about myself/my profession"). Statements about a change in WHO THE STUDENT IS
   or how they see their professional identity belong ONLY in TRANSFORMATIVE DEPTH below, never
   in ANALYSIS, even if they also explain a cause. Each phrase should be assigned to exactly ONE
   component, not duplicated across components.

5. ALTERNATIVE INTERPRETATIONS
   What it is: Consideration of other perspectives, approaches, or ways the situation could have unfolded differently.
   Grounded in: Gibbs Stage 4 (could you have done it differently?), Schön (questioning
   assumptions), Moon (higher levels — considering alternative explanations), Badenhorst et al.
   (critical reflective writing), Hatton & Smith (1995) "dialogic reflection"
   Look for: Sentences that consider how OTHERS might see the situation, alternative approaches,
   or counterfactual thinking. This is often introduced by phrases like "from [person]'s
   perspective", "[person] may have seen this differently", "if I put myself in [person]'s
   shoes", "another approach would have been". Read the WHOLE reflection carefully for this
   component specifically — it is frequently under-detected. Check every paragraph for any
   mention of a client, teammate, supervisor, or other party's likely viewpoint, not just the
   most obvious candidate sentence.
   Example phrases: "From the client's perspective...", "Another approach would have been...", "In hindsight, a different strategy could have been...", "My teammate may have seen this differently"
   NOT this: Just saying what you would do next time — that is future action, not alternative interpretation.

6. FUTURE ACTION
   What it is: A concrete plan, intention, or strategy for future behaviour based on what was learned.
   Grounded in: Gibbs Stage 6 (Action Plan), Kolb Stage 4 (Active Experimentation), Rolfe "Now
   What?", Carless & Boud (taking action on feedback), Bain et al. (2002) "Reconstructing"
   Look for: Forward-looking sentences with a SPECIFIC, NAMEABLE behaviour or approach the
   student intends to change or adopt.
   Example phrases: "Moving forward I will...", "My strategy for next week is...", "I plan to improve my communication by...", "I intend to..."
   NOT this: Vague statements like "I will try to do better" with no specific action described.
   ALSO NOT THIS - continuing current behaviour with no actual change, e.g. "I'll just keep doing
   what I've been doing", "I'll see how things go", "I'll continue as normal" - these describe
   NOT changing anything and must NOT be extracted as future action, even though they are
   grammatically forward-looking.

7. TRANSFORMATIVE DEPTH
   What it is: Genuine insight that goes beyond the task — a shift in thinking, professional identity, values, or self-understanding. The rarest and deepest component.
   Grounded in: Moon (Transformative Learning — highest level), Mezirow (1991) "disorienting
   dilemma" / questioning frames of reference, Gibbs Stage 5 (Conclusions), Schön (questioning
   professional assumptions), Badenhorst et al. (engineering identity and critical reflection),
   Hains-Wesson & Young (2017) (STEM students rarely reach this level naturally)
   Look for: Sentences where the student expresses a genuine change in how they see THEMSELVES as
   a professional, engineer, or teammate — not just what they will do differently but who they
   are becoming. This is usually the SAME sentence that also sounds like it could be "analysis" —
   when a sentence has both a causal explanation AND an identity/self-concept claim, it goes here
   in DEPTH, not in ANALYSIS.
   Example phrases: "This experience changed how I see my role as an engineer", "I now question whether...", "I realised my assumptions about teamwork were wrong", "This shifted my understanding of what it means to be professional"
   NOT this: Action plans, evaluations, or surface-level realisations. Must involve identity, values, or fundamental mindset shift.
"""

SYSTEM_PROMPT = """You are a research assistant analysing student engineering internship reflections.
Extract exact word-for-word phrases from the text only. Never fabricate, paraphrase, or invent
phrases that are not an exact substring of the provided text.
Respond ONLY with valid JSON. No preamble, no explanation, no markdown fences, no <think> blocks."""

EXTRACTION_PROMPT = COMPONENT_DEFINITIONS + """

---
Now analyse the following reflection and return a JSON object with these exact keys:
{
  "description": ["exact phrase 1", "exact phrase 2"],
  "emotional": ["exact phrase 1"],
  "evaluation": ["exact phrase 1", "exact phrase 2"],
  "analysis": ["exact phrase 1"],
  "alternative": [],
  "future_action": ["exact phrase 1"],
  "depth": [],
  "topic": "communication"
}

Rules:
- Maximum 3 phrases per component
- Each phrase must appear WORD-FOR-WORD in the reflection text below. Do not paraphrase or invent phrases.
- Return [] for any component that is genuinely absent
- Do NOT include phrases that only partially match a component
- Each phrase should be assigned to exactly ONE component - do not duplicate the same phrase across multiple components
- Check the ALTERNATIVE INTERPRETATIONS component carefully - it is easy to miss
- If two separate qualifying sentences sit right next to each other in the text, extract them as
  TWO separate list entries, not one merged phrase spanning both sentences - each independent
  qualifying sentence should be its own list item, even if adjacent
- "topic" must be the single best-matching PRIMARY topic/theme of the whole reflection, chosen
  from exactly this list: teamwork, time_management, communication, technical_skills,
  client_management, professional_conduct, conflict_resolution, personal_growth,
  project_planning, problem_solving, leadership, other"""

JUDGE_SYSTEM_PROMPT = """You are reviewing already-extracted, already verbatim-verified reflection phrases against their assigned component label. Presumption of validity: KEEP a phrase unless it clearly belongs to a different component or explicitly matches a stated exclusion in its definition. Do not reject for being brief, simple, or "generic" unless that literally matches a stated exclusion. When uncertain, KEEP. Respond ONLY with valid JSON - no preamble, no markdown fences, no think blocks."""

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


# ── Helpers (identical logic to v5.1) ───────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())


def strip_think_blocks(raw: str) -> str:
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def _normalize_for_matching(s: str, strip_quotes: bool = False, fold_case: bool = False) -> str:
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    if strip_quotes:
        s = s.replace('"', "").replace("'", "")
    s = re.sub(r"\s+", " ", s).strip()
    if fold_case:
        s = s.lower()
    return s


def verify_phrases_against_text(phrases: dict, source_text: str, entry_filename: str, entry_week: str) -> dict:
    verified = {}
    for component, phrase_list in phrases.items():
        kept = []
        for phrase in phrase_list:
            phrase_clean = phrase.strip()
            if not phrase_clean:
                continue

            if phrase_clean in source_text:
                kept.append(phrase_clean)
                continue

            norm_phrase = _normalize_for_matching(phrase_clean)
            norm_source = _normalize_for_matching(source_text)
            if norm_phrase in norm_source:
                kept.append(phrase_clean)
                SOFT_MATCH_LOG_NEW.append({
                    "filename": entry_filename, "week": entry_week, "component": component,
                    "phrase": phrase_clean, "match_type": "dash_or_quote_unicode_normalization",
                })
                continue

            nq_phrase = _normalize_for_matching(phrase_clean, strip_quotes=True)
            nq_source = _normalize_for_matching(source_text, strip_quotes=True)
            if nq_phrase in nq_source:
                kept.append(phrase_clean)
                SOFT_MATCH_LOG_NEW.append({
                    "filename": entry_filename, "week": entry_week, "component": component,
                    "phrase": phrase_clean, "match_type": "quote_character_swap",
                })
                continue

            ci_phrase = _normalize_for_matching(phrase_clean, strip_quotes=True, fold_case=True)
            ci_source = _normalize_for_matching(source_text, strip_quotes=True, fold_case=True)
            if ci_phrase in ci_source:
                kept.append(phrase_clean)
                SOFT_MATCH_LOG_NEW.append({
                    "filename": entry_filename, "week": entry_week, "component": component,
                    "phrase": phrase_clean, "match_type": "case_insensitive_recommend_manual_check",
                })
                continue

            FABRICATION_LOG_NEW.append({
                "filename": entry_filename,
                "week": entry_week,
                "component": component,
                "fabricated_phrase": phrase_clean,
            })
        verified[component] = kept
    return verified


def get_phrase_context(phrase: str, source_text: str, window_chars: int = 150) -> str:
    idx = source_text.find(phrase)
    if idx == -1:
        norm_phrase = _normalize_for_matching(phrase, strip_quotes=True, fold_case=True)
        norm_source = _normalize_for_matching(source_text, strip_quotes=True, fold_case=True)
        idx = norm_source.find(norm_phrase)
        if idx == -1:
            return phrase
    start = max(0, idx - window_chars)
    end = min(len(source_text), idx + len(phrase) + window_chars)
    snippet = source_text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(source_text):
        snippet = snippet + "..."
    return snippet


def extract_phrases(reflection_text: str):
    components = ["description","emotional","evaluation","analysis",
                  "alternative","future_action","depth"]
    empty = {c: [] for c in components}

    if len(reflection_text.strip()) < 50:
        return empty, "other"

    prompt = EXTRACTION_PROMPT + "\n\nREFLECTION:\n" + reflection_text[:3500]

    for attempt in range(MAX_RETRIES):
        try:
            chat_kwargs = dict(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                format="json",
                options={"temperature": 0.05}
            )
            if _CHAT_SUPPORTS_THINK:
                chat_kwargs["think"] = False

            response = ollama.chat(**chat_kwargs)
            raw = response.message.content.strip()
            raw = strip_think_blocks(raw)
            raw = raw.replace("```json","").replace("```","").strip()
            parsed = json.loads(raw)
            for c in components:
                if c not in parsed:
                    parsed[c] = []
            topic = parsed.get("topic", "other")
            if topic not in TOPIC_CHOICES:
                topic = "other"
            phrases = {c: parsed[c] for c in components}
            return phrases, topic
        except json.JSONDecodeError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
            else:
                return empty, "other"
        except Exception as e:
            if "Connection refused" in str(e) or "not found" in str(e).lower():
                print(f"\nERROR: Ollama not running or model not found.")
                print(f"Run: ollama serve   and   ollama pull {MODEL}")
                exit(1)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                return empty, "other"
    return empty, "other"


def judge_phrases(reflection_text: str, phrases: dict) -> dict:
    if not ENABLE_LLM_JUDGE:
        return phrases

    to_judge = {c: p for c, p in phrases.items() if p and c in JUDGE_COMPONENTS}
    if not to_judge:
        return phrases

    items = []
    for component, phrase_list in to_judge.items():
        definition = CONCISE_JUDGE_DEFINITIONS[component]
        for phrase in phrase_list:
            context = get_phrase_context(phrase, reflection_text)
            items.append({"component": component, "phrase": phrase, "context": context, "definition": definition})

    lines = []
    for i, item in enumerate(items):
        lines.append(
            f'{i}. Component: {item["component"]}\n'
            f'   Definition: {item["definition"]}\n'
            f'   Context: "{item["context"]}"\n'
            f'   Phrase: "{item["phrase"]}"'
        )
    items_block = "\n\n".join(lines)

    prompt = f"""Review each numbered phrase below against its component definition, using the surrounding context provided. KEEP it if it genuinely satisfies the definition. REJECT only if it clearly belongs to a different component, or explicitly matches a "NOT this" style exclusion in the definition. When uncertain, KEEP.

{items_block}

Return ONLY a JSON object of this exact shape, with exactly one entry per numbered item above, in order:
{{"verdicts": [{{"index": 0, "keep": true, "reason": "..."}}, {{"index": 1, "keep": false, "reason": "..."}}]}}"""

    for attempt in range(MAX_RETRIES):
        try:
            chat_kwargs = dict(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                format="json",
                options={"temperature": 0.05},
            )
            if _CHAT_SUPPORTS_THINK:
                chat_kwargs["think"] = False

            response = ollama.chat(**chat_kwargs)
            raw = response.message.content.strip()
            raw = strip_think_blocks(raw)
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            verdict_list = parsed.get("verdicts", [])
            verdict_by_index = {v.get("index"): v for v in verdict_list if isinstance(v, dict)}

            judged = {c: list(p) for c, p in phrases.items()}
            for c in to_judge:
                judged[c] = []

            for i, item in enumerate(items):
                v = verdict_by_index.get(i)
                keep = True if v is None else v.get("keep", True)
                if keep:
                    judged[item["component"]].append(item["phrase"])
                else:
                    JUDGE_LOG_NEW.append({
                        "component": item["component"],
                        "phrase": item["phrase"],
                        "reason": (v or {}).get("reason", ""),
                    })
            return judged

        except json.JSONDecodeError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
            else:
                return phrases
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                return phrases

    return phrases


def count_phrase_units(phrase: str) -> int:
    pieces = _SENTENCE_SPLIT_RE.split(phrase.strip())
    substantial = [p for p in pieces if len(p.split()) >= 4]
    return max(1, len(substantial))


def score_from_phrases(phrases: dict) -> dict:
    scores = {}
    for component, phrase_list in phrases.items():
        n = sum(count_phrase_units(p) for p in phrase_list if p.strip())
        if n >= 2:
            scores[component] = "Present"
        elif n == 1:
            scores[component] = "Weak"
        else:
            scores[component] = "Missing"
    return scores


def compute_length_normalized_rates(phrases: dict, wc: int) -> dict:
    if wc <= 0:
        return {c: 0.0 for c in phrases}
    rates = {}
    for component, phrase_list in phrases.items():
        n = sum(count_phrase_units(p) for p in phrase_list if p.strip())
        rates[component] = round((n / wc) * 100, 2)
    return rates


def apply_hierarchy_gate(scores: dict) -> dict:
    foundation_ok = all(scores.get(c, "Missing") != "Missing" for c in FOUNDATION_COMPONENTS)

    ungrounded = [
        c for c in HIGHER_COMPONENTS
        if scores.get(c, "Missing") != "Missing" and not foundation_ok
    ]

    higher_present_count = sum(1 for c in HIGHER_COMPONENTS if scores.get(c, "Missing") != "Missing")

    if scores.get("description", "Missing") == "Missing":
        profile = "no_reflective_content"
    elif not foundation_ok:
        profile = "partial_foundation"
    elif higher_present_count == 0:
        profile = "foundational_only"
    elif higher_present_count <= 1:
        profile = "foundational_plus_partial"
    else:
        profile = "full_depth"

    return {
        "foundation_established": foundation_ok,
        "ungrounded_higher_components": ungrounded,
        "profile": profile,
    }


def load_reflections_from_json(json_path: Path) -> list:
    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Ensure the reflections.json file exists.")
        exit(1)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        print(f"ERROR: {json_path} is empty.")
        exit(1)
    processed = []
    for entry in data:
        processed.append({
            "filename": entry.get("filename", "unknown"),
            "week": entry.get("week", "Unknown Week"),
            "sid": entry.get("sid", ""),
            "file_type": entry.get("file_type", ""),
            "reflection": entry.get("reflection", ""),
            "full_text": entry.get("full_text", ""),
        })
    return processed


def load_json_list(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not OUTPUT_FILE.exists():
        print(f"ERROR: {OUTPUT_FILE} not found - this script updates an existing v5.1 run, "
              f"it doesn't start from scratch. Run the original v5.1 script first.")
        exit(1)

    with open(OUTPUT_FILE, encoding="utf-8") as f:
        existing_results = json.load(f)
    print(f"Loaded {len(existing_results)} existing result(s) from {OUTPUT_FILE}")

    all_reflections = load_reflections_from_json(REFLECTIONS_JSON)
    print(f"Loaded {len(all_reflections)} reflection(s) from {REFLECTIONS_JSON}")

    targets = [e for e in all_reflections if e["week"] in TARGET_WEEKS]
    print(f"\n{len(targets)} entries match target weeks {sorted(TARGET_WEEKS)}")

    # Diagnostic: raw source text length per target entry, BEFORE re-extraction.
    # If most of these are near-zero, the problem is upstream in reflections.json
    # (step A), not in the v5.1 extraction pipeline - re-running extraction here
    # will correctly still return empty/Missing, because there's nothing to extract.
    short_count = sum(1 for e in targets if len(e.get("reflection", "").strip()) < 50)
    print(f"Diagnostic: {short_count}/{len(targets)} target entries have reflection text "
          f"under 50 chars (auto-empty in extract_phrases()).")
    if short_count:
        print("  -> If this number is high, check reflections.json itself for those weeks -")
        print("     the source 'reflection' field may be empty/truncated, which would explain")
        print("     the 100% Missing pattern independent of anything in this script.")

    print(f"\nModel: {MODEL}   Judge model: {JUDGE_MODEL}   (think mode disabled: {_CHAT_SUPPORTS_THINK})")

    # Build lookup of existing results by (filename, week) so we can splice in place
    existing_by_key = {r["filename"] + r["week"]: idx for idx, r in enumerate(existing_results)}

    # Snapshot the pre-existing logs once, so checkpoints during the loop can just
    # append this run's NEW entries without re-reading (and double-counting) the
    # growing on-disk file each time.
    _checkpoint_log_base_fabrication = load_json_list(FABRICATION_FILE)
    _checkpoint_log_base_soft = load_json_list(SOFT_MATCH_FILE)
    _checkpoint_log_base_judge = load_json_list(JUDGE_FILE)

    updated_count = 0
    for i, entry in enumerate(targets):
        text = entry.get("reflection", "")
        print(f"[{i+1}/{len(targets)}] {entry['week']} | {entry['filename'][:45]} "
              f"(reflection len={len(text.strip())} chars)")

        phrases, topic = extract_phrases(text)
        phrases = verify_phrases_against_text(phrases, text, entry["filename"], entry["week"])
        phrases = judge_phrases(text, phrases)
        scores  = score_from_phrases(phrases)
        wc      = word_count(text)

        entry["phrases"]    = phrases
        entry["scores"]     = scores
        entry["topic"]      = topic
        entry["word_count"] = wc

        if ENABLE_LENGTH_NORMALIZATION:
            entry["phrases_per_100_words"] = compute_length_normalized_rates(phrases, wc)

        if ENABLE_HIERARCHY_GATE:
            entry["gating"] = apply_hierarchy_gate(scores)

        key = entry["filename"] + entry["week"]
        if key in existing_by_key:
            existing_results[existing_by_key[key]] = entry
        else:
            # Wasn't in the original output at all (e.g. was skipped entirely last run) - append it
            existing_results.append(entry)
            existing_by_key[key] = len(existing_results) - 1

        updated_count += 1
        time.sleep(DELAY_SECS)

        # Checkpoint every 10 re-extracted entries, same cadence as v5.1's main(),
        # so a crash/interrupt partway through the 359-entry run doesn't lose
        # everything - phrases_output.json always reflects the latest saved state.
        if updated_count % 10 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=2)
            # Also checkpoint the logs so a crash doesn't lose this run's fabrication/
            # soft-match/judge entries: original on-disk contents (snapshotted once
            # before the loop) plus everything logged so far this run.
            with open(FABRICATION_FILE, "w", encoding="utf-8") as f:
                json.dump(_checkpoint_log_base_fabrication + FABRICATION_LOG_NEW, f, ensure_ascii=False, indent=2)
            with open(SOFT_MATCH_FILE, "w", encoding="utf-8") as f:
                json.dump(_checkpoint_log_base_soft + SOFT_MATCH_LOG_NEW, f, ensure_ascii=False, indent=2)
            with open(JUDGE_FILE, "w", encoding="utf-8") as f:
                json.dump(_checkpoint_log_base_judge + JUDGE_LOG_NEW, f, ensure_ascii=False, indent=2)
            print(f"  ...checkpoint saved ({updated_count}/{len(targets)} done)")

    print(f"\nRe-extracted and spliced in {updated_count} entries for weeks {sorted(TARGET_WEEKS)}.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=2)
    print(f"Saved updated {OUTPUT_FILE}")

    # Merge and save logs: original on-disk contents (snapshotted before the loop
    # started) plus this run's new entries. Using the snapshot rather than re-reading
    # the files here avoids double-counting anything already written by the mid-run
    # checkpoints above. (Weeks 1/2/3/8 are untouched by this run either way.)
    fabrication_log = _checkpoint_log_base_fabrication + FABRICATION_LOG_NEW
    soft_match_log  = _checkpoint_log_base_soft + SOFT_MATCH_LOG_NEW
    judge_log       = _checkpoint_log_base_judge + JUDGE_LOG_NEW

    with open(FABRICATION_FILE, "w", encoding="utf-8") as f:
        json.dump(fabrication_log, f, ensure_ascii=False, indent=2)
    with open(SOFT_MATCH_FILE, "w", encoding="utf-8") as f:
        json.dump(soft_match_log, f, ensure_ascii=False, indent=2)
    with open(JUDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(judge_log, f, ensure_ascii=False, indent=2)

    print(f"\nThis run: {len(FABRICATION_LOG_NEW)} new fabrication(s), "
          f"{len(SOFT_MATCH_LOG_NEW)} new soft-match(es), {len(JUDGE_LOG_NEW)} new judge rejection(s).")

    # ── Recalculate ALL stats over the full, now-corrected dataset ─────────────
    print(f"\nRebuilding {STATS_FILE} over all {len(existing_results)} reflections...\n")

    components = ["description","emotional","evaluation","analysis",
                  "alternative","future_action","depth"]
    component_labels = {
        "description":   "Description of event",
        "emotional":     "Emotional response / feelings",
        "evaluation":    "Evaluation (good/bad)",
        "analysis":      "Analysis / making sense",
        "alternative":   "Alternative interpretations",
        "future_action": "Future action / what next",
        "depth":         "Transformative depth / insight",
    }

    total = len(existing_results)
    stats = {
        "total_reflections": total,
        "model_used": MODEL,
        "fabricated_phrases_dropped": len(fabrication_log),
        "soft_matches_normalized": len(soft_match_log),
        "judge_rejections": len(judge_log),
        "frameworks_used": [
            "Gibbs (1988)", "Moon (2004)", "Kolb (1984)",
            "Rolfe et al. (2001)", "Schon (1983)",
            "Carless & Boud (2018)", "Badenhorst et al. (2020)",
            "Hains-Wesson & Young (2017)", "Boud, Keogh & Walker (1985)",
            "Mezirow (1991)", "Hatton & Smith (1995)"
        ],
        "components": {},
        "by_week": {},
        "topic_analysis": {},
    }

    print("=" * 65)
    print(f"OVERALL NORMALISED RESULTS — {total} reflections (weeks 4-7 re-extracted)")
    print("=" * 65)

    for c in components:
        counts = Counter(r["scores"].get(c,"Missing") for r in existing_results)
        p, w, m = counts["Present"], counts["Weak"], counts["Missing"]
        has_phrase = sum(1 for r in existing_results if len(r["phrases"].get(c,[])) > 0)

        stats["components"][c] = {
            "label":              component_labels[c],
            "present_n":          p,
            "weak_n":             w,
            "missing_n":          m,
            "present_pct":        round(p / total * 100, 1),
            "weak_pct":           round(w / total * 100, 1),
            "missing_pct":        round(m / total * 100, 1),
            "has_any_phrase_pct": round(has_phrase / total * 100, 1),
        }

        print(f"\n{component_labels[c]}")
        print(f"  Present : {p:3d}  ({p/total*100:5.1f}%)")
        print(f"  Weak    : {w:3d}  ({w/total*100:5.1f}%)")
        print(f"  Missing : {m:3d}  ({m/total*100:5.1f}%)")

    print("\n" + "=" * 65)
    print("BY WEEK — NORMALISED COMPONENT SCORES + WORD COUNT")
    print("=" * 65)

    week_groups = {}
    for r in existing_results:
        w = r["week"]
        week_groups.setdefault(w, []).append(r)

    for wk in sorted(week_groups.keys()):
        group    = week_groups[wk]
        n        = len(group)
        timeline = JFC_TIMELINE.get(wk, {"dates": "unknown", "note": ""})
        avg_wc   = round(sum(r.get("word_count", 0) for r in group) / n, 1)

        print(f"\n{wk}  [{timeline['dates']}]  n={n}  avg_words={avg_wc}")
        print(f"  Context: {timeline['note']}")

        week_data = {
            "n":         n,
            "dates":     timeline["dates"],
            "context":   timeline["note"],
            "avg_words": avg_wc,
            "components": {}
        }

        for c in components:
            counts = Counter(r["scores"].get(c,"Missing") for r in group)
            p, w2, m = counts["Present"], counts["Weak"], counts["Missing"]
            week_data["components"][c] = {
                "present_pct": round(p / n * 100, 1),
                "weak_pct":    round(w2 / n * 100, 1),
                "missing_pct": round(m / n * 100, 1),
            }
            label_short = component_labels[c].split("/")[0].strip()[:20]
            print(f"  {label_short:<22} Present={p/n*100:4.1f}%  Weak={w2/n*100:4.1f}%  Missing={m/n*100:4.1f}%")

        stats["by_week"][wk] = week_data

    print("\n" + "=" * 65)
    print("AVERAGE WORD COUNT BY WEEK")
    print("=" * 65)
    for wk in sorted(week_groups.keys()):
        group  = week_groups[wk]
        avg_wc = round(sum(r.get("word_count",0) for r in group) / len(group), 1)
        note   = JFC_TIMELINE.get(wk, {}).get("note","")[:60]
        print(f"  {wk}: {avg_wc:6.1f} words   [{note}]")

    print("\n" + "=" * 65)
    print("TOPIC ANALYSIS — WHAT ARE STUDENTS REFLECTING ON?")
    print("=" * 65)

    topic_counter = Counter(r.get("topic","other") for r in existing_results)
    stats["topic_analysis"]["overall"] = dict(topic_counter.most_common())

    for topic, count in topic_counter.most_common():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {topic:<25} {count:4d}  ({pct:5.1f}%)  {bar}")

    print("\nTopic distribution by week:")
    topic_by_week = {}
    for wk in sorted(week_groups.keys()):
        group = week_groups[wk]
        tc = Counter(r.get("topic","other") for r in group)
        top3 = ", ".join(f"{t}({c})" for t, c in tc.most_common(3))
        print(f"  {wk}: {top3}")
        topic_by_week[wk] = dict(tc)
    stats["topic_analysis"]["by_week"] = topic_by_week

    print("\n" + "=" * 65)
    print("EMOTIONAL RESPONSE TREND BY WEEK (normalised)")
    print("=" * 65)
    for wk in sorted(week_groups.keys()):
        group = week_groups[wk]
        n = len(group)
        counts = Counter(r["scores"].get("emotional","Missing") for r in group)
        p, w2, m = counts["Present"], counts["Weak"], counts["Missing"]
        note = JFC_TIMELINE.get(wk, {}).get("note","")[:55]
        print(f"  {wk}: Present={p/n*100:.0f}% Weak={w2/n*100:.0f}% Missing={m/n*100:.0f}%")
        print(f"    → {note}")

    if ENABLE_HIERARCHY_GATE:
        print("\n" + "=" * 65)
        print("REFLECTION PROFILES (hierarchy gate - soft flag, not a hard cap)")
        print("=" * 65)
        profile_counter = Counter(r.get("gating", {}).get("profile", "unknown") for r in existing_results)
        stats["profile_distribution"] = dict(profile_counter.most_common())
        for profile, count in profile_counter.most_common():
            print(f"  {profile:<28} {count:4d}  ({count/total*100:5.1f}%)")

        ungrounded_count = sum(1 for r in existing_results if r.get("gating", {}).get("ungrounded_higher_components"))
        stats["reflections_with_ungrounded_higher_components"] = ungrounded_count
        if ungrounded_count:
            print(f"\n  ⚠ {ungrounded_count} reflection(s) scored a higher-order component "
                  f"without an established foundation (flagged, not altered - see each entry's "
                  f"'gating' field in {OUTPUT_FILE}).")

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*65}")
    print(f"Saved: {OUTPUT_FILE}       (weeks 4-7 re-extracted, weeks 1/2/3/8 unchanged)")
    print(f"Saved: {STATS_FILE}        (fully recalculated over all {total} reflections)")
    print(f"Saved: {FABRICATION_FILE}  ({len(fabrication_log)} total, {len(FABRICATION_LOG_NEW)} new)")
    print(f"Saved: {SOFT_MATCH_FILE}  ({len(soft_match_log)} total, {len(SOFT_MATCH_LOG_NEW)} new)")
    print(f"Saved: {JUDGE_FILE}  ({len(judge_log)} total, {len(JUDGE_LOG_NEW)} new)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()