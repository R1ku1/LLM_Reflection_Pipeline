"""
Reflection Phrase Extractor (Ollama Local Version) - v5.1
===========================================================
Changes from v5 (efficiency-only, no intended change to results):
  - JUDGE SCOPE NARROWED: the LLM-judge pass now only reviews the five
    components where mislabeling was actually observed (evaluation,
    analysis, alternative, future_action, depth). description/emotional
    phrases pass through unjudged - they were not a source of "real text,
    wrong label" errors, so reviewing them was pure overhead.
  - JUDGE PROMPT SLIMMED: the judge no longer receives the full reflection
    text or the full component-definitions block with six few-shot
    examples. Each phrase is now sent with a short (~150 char either side)
    context window located around it (see get_phrase_context()) plus a
    one-line definition - fabrication is already ruled out by this stage,
    so a label check doesn't need the whole document or full framework
    citations.
  - TOPIC CLASSIFICATION MERGED INTO EXTRACTION: extract_phrases() now
    also returns the reflection's topic in the same JSON response/model
    call, removing the previously-separate extract_topic() invocation per
    reflection. extract_topic() is left in the file as a standalone
    fallback/manual-rerun function but main() no longer calls it.
  - Per-phrase judge calls were considered and rejected: one larger judge
    call per reflection is still cheaper than N small ones per reflection.

Changes from v4:
  - LLM-AS-JUDGE PASS: after phrase extraction + verbatim verification, a
    second model call reviews every surviving phrase against its component's
    full definition (including the exclusion criteria) and can reject
    phrases that are verbatim-real but don't actually qualify - e.g. generic
    filler wrongly labelled EVALUATION, or a phrase assigned to the wrong
    component. Rejections are logged to judge_log.json with the model's
    stated reason, so this is auditable rather than a silent second filter.
    This is the mitigation discussed for the "real text, wrong label"
    over-generalisation failure mode (as distinct from fabrication, which
    verify_phrases_against_text() already handles).
  - LENGTH NORMALIZATION: raw phrase counts are no longer the only measure
    reported. Each component now also gets a phrases_per_100_words figure,
    since raw counts conflate "wrote more" with "reflected more deeply" on
    longer reflections. The existing Present/Weak/Missing thresholds are
    UNCHANGED (kept for continuity with the Semester 1 baseline) - the
    normalized figure is added alongside, not used to override scoring yet,
    pending calibration against the human-coded validation subsample.
  - ADJACENT-PHRASE FIX: scoring no longer just counts list length. Each
    extracted phrase is checked for whether it's actually two+ complete
    sentences merged into one string (the bug flagged in the meeting, where
    two separate qualifying sentences sitting next to each other got
    extracted as one continuous phrase and under-counted as a result).
    Sentence-splitting is used to count "genuine instances" for scoring,
    while the original merged phrase is still stored for readability. The
    extraction prompt also now explicitly instructs the model to split
    adjacent qualifying sentences into separate list entries.
  - HIERARCHICAL / SEQUENTIAL GATING: added a SOFT gate (per the "pilot
    before committing" decision) rather than a hard cap. Description and
    Evaluation are treated as the foundation. If a reflection scores a
    higher-order component (Alternative, Future Action, Depth) as Present
    or Weak while the foundation is largely absent, this is flagged as
    "ungrounded_higher_component" rather than silently accepted - raw
    scores are NOT changed, so this can be evaluated on the validation
    subsample before deciding whether to make it a hard gate. Each
    reflection also now gets an overall "profile" label (e.g.
    "foundational_only", "full_depth") as a first step toward the
    actionable-feedback redesign discussed separately.

Changes from v3 (carried forward):
  - Switched default model to qwen3:14b (larger model to try to fix the
    recall miss on ALTERNATIVE and the EVALUATION/ANALYSIS/DEPTH boundary
    confusion observed with qwen2.5:7b on the sample reflections).
  - qwen3 models support a "thinking" mode that can wrap output in
    <think>...</think> blocks. This is handled two ways for safety:
      1) think=False is passed to ollama.chat() if the installed ollama
         python client supports it (auto-detected, falls back cleanly
         if not).
      2) Any <think>...</think> content is stripped via regex before
         JSON parsing regardless, so this is safe even on older clients.
  - Added verify_phrases_against_text(): every extracted phrase is now
    checked as an exact substring of the reflection text. Any phrase
    that is NOT a verbatim match (i.e. fabricated/paraphrased by the
    model, despite instructions not to) is dropped before scoring, and
    logged to fabrication_log.json with the offending phrase, the
    component, and the source file. This fixes the fabricated-quote
    issue found in 2.txt (an "alternative" phrase that did not actually
    appear in the source text).
  - Tightened COMPONENT_DEFINITIONS based on specific errors observed
    on the sample reflections:
      * EVALUATION: added explicit exclusion of generic non-judgement
        filler ("reasonable pace", "pretty typical", "nothing stands
        out") which was previously being extracted as evaluation.
      * FUTURE_ACTION: added explicit exclusion of "keep doing what
        I've been doing" / "see how things go" - continuing current
        behaviour is not a specific new action.
      * ANALYSIS vs EVALUATION: added an explicit instruction that a
        single sentence should be split — the judging clause goes to
        EVALUATION, the causal/explanatory clause goes to ANALYSIS —
        rather than the whole sentence being filed under one or the
        other.
      * ANALYSIS vs DEPTH: added an explicit instruction that identity-
        level claims ("this changed how I see my role", "I used to
        think X but now I...") belong to DEPTH only, never duplicated
        into ANALYSIS.
      * Reinforced the no-fabrication rule directly in the extraction
        prompt rules, not just the system prompt.
  - Reads input reflections from reflections.json (already extracted
    reflections of the actual JFC data) instead of a folder of .txt files.
    Output is saved to a new v5_output/ directory to avoid overwriting
    old stats.

Usage:
    ollama pull qwen3:14b
    pip install ollama
    Ensure reflections.json exists at ../A/JFC 25 26 Reflections/reflections.json
    python code.py

Output (saved to v5_output/):
    phrases_output.json     - full results: extracted phrases, raw scores,
                              length-normalized rates, gating flags, profile
    summary_stats.json      - normalised component stats + word counts + topics
                              + profile distribution across the dataset
    fabrication_log.json    - any phrases the model produced that were NOT
                              verbatim in the source text (should ideally be empty)
    soft_match_log.json     - kept phrases that needed quote/case normalization
    judge_log.json          - phrases the LLM-judge pass rejected, with reasons
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

# ── Config ─────────────────────────────────────────────────────────────────────
# Path to the already-extracted reflections JSON (relative to code.py in ../B/)
REFLECTIONS_JSON = Path("../A/JFC 25 26 Reflections/reflections.json")

# Output directory (new, to avoid overwriting old stats)
OUTPUT_DIR       = Path("v5_output")
OUTPUT_FILE      = OUTPUT_DIR / "phrases_output.json"
STATS_FILE       = OUTPUT_DIR / "summary_stats.json"
FABRICATION_FILE = OUTPUT_DIR / "fabrication_log.json"
SOFT_MATCH_FILE  = OUTPUT_DIR / "soft_match_log.json"
JUDGE_FILE       = OUTPUT_DIR / "judge_log.json"
MODEL            = "qwen2.5:7b"
MAX_RETRIES      = 3
DELAY_SECS       = 0.3

# ── New v5 toggles ──────────────────────────────────────────────────────────
# All three are deliberately separate ON/OFF switches so any one of them can
# be disabled independently while comparing before/after output, rather than
# being baked in as an all-or-nothing change.

ENABLE_LLM_JUDGE = True     # second-pass model review of extracted phrases
JUDGE_MODEL      = "qwen3:14b"    

ENABLE_LENGTH_NORMALIZATION = True   # add phrases_per_100_words fields (reporting only - does not change Present/Weak/Missing thresholds yet)

ENABLE_HIERARCHY_GATE = True   # flag (not hard-cap) higher components scored without a foundation present
FOUNDATION_COMPONENTS  = ["description", "evaluation"]
HIGHER_COMPONENTS      = ["alternative", "future_action", "depth"]

# ── v5.1 efficiency changes ──────────────────────────────────────────────────
# Three changes aimed at cutting judge-pass and topic-classification cost
# without changing the substance of what gets kept/rejected:
#
# 1) Only the components where extraction actually confuses labels get a
#    judge review. DESCRIPTION and EMOTIONAL are reliably easy to identify
#    (a factual account; a named emotion + cause) - the errors observed on
#    the sample reflections were all EVALUATION/ANALYSIS/ALTERNATIVE/
#    FUTURE_ACTION/DEPTH boundary confusion, so those are the only ones
#    worth a second model call.
# 2) The judge sees a short window of surrounding text per phrase instead
#    of the full ~3500-char reflection, plus a one-line definition instead
#    of the full framework citations + exclusion prose + six worked
#    examples. Fabrication is already ruled out by this point, so the judge
#    only needs enough context to check the label, not the whole document.
# 3) Topic classification is folded into the extraction call (same model,
#    same read of the reflection) instead of being a separate model
#    invocation per reflection - extract_topic() is kept below as a
#    standalone fallback but main() no longer calls it.

JUDGE_COMPONENTS = ["evaluation", "analysis", "alternative", "future_action", "depth"]

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

# Auto-detect whether the installed ollama client supports the `think` kwarg
# (added for reasoning models like qwen3/deepseek-r1). If not supported, we
# just skip passing it - the regex strip below still protects us either way.
_CHAT_SUPPORTS_THINK = "think" in inspect.signature(ollama.chat).parameters

# Global fabrication tracker - collects any extracted phrase that was not
# found verbatim in its source reflection, across the whole run.
FABRICATION_LOG = []

# Collects phrases that were kept but only matched after quote/dash
# normalization, quote-character swapping, or case-folding - i.e. genuine
# quotes that needed loosened matching, not fabrications. Worth a quick skim
# but these are NOT dropped from the results.
SOFT_MATCH_LOG = []

# Collects phrases the LLM-judge pass rejected: verbatim-real (passed
# fabrication check) but judged not to genuinely satisfy the component
# definition once checked against the exclusion criteria a second time.
JUDGE_LOG = []

# ── JFC Program Timeline ───────────────────────────────────────────────────────
JFC_TIMELINE = {
    "Week 1 Reflection":  {"dates": "1–5 Dec 2025",  "note": "Program induction week. First client meetings. Students orienting to JFC environment."},
    "Week 2 Reflection":  {"dates": "8–12 Dec 2025",  "note": "Early project phase. Teams establishing working norms and beginning technical work."},
    "Week 3 Reflection":  {"dates": "15–19 Dec 2025", "note": "Final week before Christmas break. Pre-holiday period — likely lower effort and engagement."},
    "Week 4 Reflection":  {"dates": "22 Dec 2025 – 5 Jan 2026", "note": "CHRISTMAS BREAK (22 Dec – 6 Jan). Week 4 submitted during or just after break. Expect lower quality and shorter submissions."},
    "Week 5 Reflection":  {"dates": "6–10 Jan 2026",  "note": "Return from break. Re-engagement phase. Students resuming project momentum."},
    "Week 6 Reflection":  {"dates": "13–17 Jan 2026", "note": "Mid-program. Projects in active delivery phase. Increased technical complexity."},
    "Week 7 Reflection":  {"dates": "20–24 Jan 2026", "note": "Late program. Students approaching final deliverables. Increased workload pressure."},
    "Week 8 reflection":  {"dates": "27 Jan – 13 Feb 2026", "note": "Final program week. Program conclusion, final presentations. Students may reflect more deeply on overall experience."},
}

# ── Tightened Component Definitions (v4) ──────────────────────────────────────
# Grounded in: Gibbs (1988), Moon (2004), Kolb (1984), Rolfe et al. (2001),
# Schön (1983), Carless & Boud (2018), Badenhorst et al. (2020),
# Hains-Wesson & Young (2017), Boud, Keogh & Walker (1985), Hatton & Smith (1995)

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

# ── Topic analysis prompt ──────────────────────────────────────────────────────
TOPIC_PROMPT_BASE = """You are analysing a student engineering internship reflection.
Identify the PRIMARY topic or theme this student is reflecting on.
Choose the single best match from this list:
- teamwork
- time_management
- communication
- technical_skills
- client_management
- professional_conduct
- conflict_resolution
- personal_growth
- project_planning
- problem_solving
- leadership
- other

Return ONLY a JSON object like: {"topic": "teamwork"}

REFLECTION:
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())


def strip_think_blocks(raw: str) -> str:
    """
    Safety net for reasoning models (qwen3, deepseek-r1, etc.) that may emit
    <think>...</think> reasoning before the actual JSON answer, regardless of
    whether think=False was successfully passed to ollama.chat().
    """
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def load_reflections_from_json(json_path: Path) -> list:
    """
    Load already-extracted reflections from reflections.json.
    Each entry has: week, filename, sid, file_type, reflection, full_text
    We use the 'reflection' field as the text to analyse.
    """
    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Ensure the reflections.json file exists.")
        exit(1)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print(f"ERROR: {json_path} is empty.")
        exit(1)

    # Validate expected structure and map to the internal format
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

    print(f"Loaded {len(processed)} reflection(s) from {json_path}")
    return processed


def _normalize_for_matching(s: str, strip_quotes: bool = False, fold_case: bool = False) -> str:
    """
    Normalizes text for verbatim-matching purposes only (the ORIGINAL phrase
    is still what gets stored/scored - this function is never used to alter
    output, only to decide whether a phrase counts as a genuine match).
    """
    replacements = {
        "\u2018": "'", "\u2019": "'",   # curly single quotes -> straight
        "\u201c": '"', "\u201d": '"',   # curly double quotes -> straight
        "\u2013": "-", "\u2014": "-",   # en/em dash -> hyphen
        "\u00a0": " ",                   # nbsp -> space
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    if strip_quotes:
        # Handles the model swapping " for ' (or vice versa) when re-quoting
        # a span that itself contains a quoted sub-phrase, e.g. the source
        # has `"informed."` but the model emits `'informed.'` - same words,
        # different quote character. Stripping both for the match check
        # (NOT for the stored/output phrase) avoids a false fabrication flag.
        s = s.replace('"', "").replace("'", "")
    s = re.sub(r"\s+", " ", s).strip()
    if fold_case:
        s = s.lower()
    return s


def verify_phrases_against_text(phrases: dict, source_text: str, entry_filename: str) -> dict:
    """
    Verifies every extracted phrase is genuinely present in the source text,
    using progressively looser (but still meaningful) matching passes so that
    cosmetic differences (quote-mark style, sentence-initial recapitalization
    from mid-sentence extraction) aren't mistaken for fabrication:

      1. Exact substring match (best - stored as-is)
      2. Match after normalizing dash/quote unicode variants to straight ASCII
      3. Match after also stripping all quote/apostrophe characters (handles
         the model swapping ' for " when re-quoting a nested quotation)
      4. Match after also folding case (handles "The reason..." vs
         "the reason..." when a phrase is extracted starting mid-sentence)

    Passes 2-4 are logged to SOFT_MATCH_LOG for transparency (worth a manual
    skim, but not evidence of fabrication) and the phrase IS kept, using the
    model's original (not normalized) text. Only phrases that fail ALL four
    passes are treated as fabricated: dropped, and logged to FABRICATION_LOG.
    """
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
                SOFT_MATCH_LOG.append({
                    "filename": entry_filename, "component": component,
                    "phrase": phrase_clean, "match_type": "dash_or_quote_unicode_normalization",
                })
                continue

            nq_phrase = _normalize_for_matching(phrase_clean, strip_quotes=True)
            nq_source = _normalize_for_matching(source_text, strip_quotes=True)
            if nq_phrase in nq_source:
                kept.append(phrase_clean)
                SOFT_MATCH_LOG.append({
                    "filename": entry_filename, "component": component,
                    "phrase": phrase_clean, "match_type": "quote_character_swap",
                })
                continue

            ci_phrase = _normalize_for_matching(phrase_clean, strip_quotes=True, fold_case=True)
            ci_source = _normalize_for_matching(source_text, strip_quotes=True, fold_case=True)
            if ci_phrase in ci_source:
                kept.append(phrase_clean)
                SOFT_MATCH_LOG.append({
                    "filename": entry_filename, "component": component,
                    "phrase": phrase_clean, "match_type": "case_insensitive_recommend_manual_check",
                })
                continue

            FABRICATION_LOG.append({
                "filename": entry_filename,
                "component": component,
                "fabricated_phrase": phrase_clean,
            })
        verified[component] = kept
    return verified


def get_phrase_context(phrase: str, source_text: str, window_chars: int = 150) -> str:
    """
    Returns a short window of text surrounding a phrase, instead of handing
    the judge the entire reflection. By the time this is called the phrase
    has already passed verify_phrases_against_text(), so it's a genuine
    verbatim (or near-verbatim, quote/case-normalized) quote - this just
    locates it and grabs some context on either side. Falls back to the
    bare phrase if it can't be located for some reason.
    """
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
    """
    Returns (phrases_dict, topic). v5.1: topic classification now rides
    along in the same extraction call/response instead of a separate
    extract_topic() invocation per reflection - the model already reads
    the full reflection for phrase extraction, so this removes one whole
    extra model call per reflection with no change to what gets extracted.
    """
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
                chat_kwargs["think"] = False  # disable qwen3-style reasoning mode if supported

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


JUDGE_SYSTEM_PROMPT = """You are reviewing already-extracted, already verbatim-verified reflection phrases against their assigned component label. Presumption of validity: KEEP a phrase unless it clearly belongs to a different component or explicitly matches a stated exclusion in its definition. Do not reject for being brief, simple, or "generic" unless that literally matches a stated exclusion. When uncertain, KEEP. Respond ONLY with valid JSON - no preamble, no markdown fences, no think blocks."""


def judge_phrases(reflection_text: str, phrases: dict) -> dict:
    """
    Second-pass LLM-as-judge review, run AFTER verify_phrases_against_text().
    Fabrication is already ruled out at this point - every phrase here is a
    genuine verbatim quote from the reflection. What this pass catches is
    the "real text, wrong label" failure mode: a phrase that truly exists,
    but doesn't actually satisfy the component's definition once checked
    against the exclusion criteria a second time. Every rejection is logged
    with the model's stated reason, so this is auditable, not a silent
    extra filter.

    v5.1 scope/cost changes (results-preserving):
      - Only JUDGE_COMPONENTS (evaluation, analysis, alternative,
        future_action, depth) are sent for review. description/emotional
        phrases pass straight through unchanged - these two were not the
        source of the "real text, wrong label" errors this pass was built
        to catch, so reviewing them added cost without catching anything.
      - Each phrase is sent with a short surrounding-text window
        (get_phrase_context) instead of the full ~3500-char reflection,
        and a one-line definition instead of the full component
        definitions block - fabrication and full-document context are not
        needed for a label check.

    If ENABLE_LLM_JUDGE is False, there's nothing to judge, or the judge
    call fails for any reason, this returns the phrases UNCHANGED - the
    judge pass is a strictly additive safety net, never a point of failure.
    """
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
                keep = True if v is None else v.get("keep", True)  # missing verdict -> fail-safe keep
                if keep:
                    judged[item["component"]].append(item["phrase"])
                else:
                    JUDGE_LOG.append({
                        "component": item["component"],
                        "phrase": item["phrase"],
                        "reason": (v or {}).get("reason", ""),
                    })
            return judged

        except json.JSONDecodeError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
            else:
                return phrases  # fail-safe: judge unavailable, keep original phrases
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                return phrases  # fail-safe: judge unavailable, keep original phrases

    return phrases


def extract_topic(reflection_text: str) -> str:
    """
    Standalone topic classifier. As of v5.1, main() no longer calls this -
    topic is returned directly by extract_phrases() in the same model call.
    Kept here for manual/standalone use (e.g. reclassifying topic alone
    without re-running phrase extraction).
    """
    if len(reflection_text.strip()) < 50:
        return "other"
    prompt = TOPIC_PROMPT_BASE + reflection_text[:2000]
    for attempt in range(MAX_RETRIES):
        try:
            chat_kwargs = dict(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
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
            return parsed.get("topic", "other")
        except:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
    return "other"


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def count_phrase_units(phrase: str) -> int:
    """
    A single extracted "phrase" can actually be two or more complete
    sentences that were merged into one string by the model (the
    adjacent-phrase bug flagged in the meeting - two genuinely separate
    qualifying sentences sitting next to each other got extracted as one
    continuous span and under-counted as a result).

    This splits on sentence boundaries (period/!/? followed by a capital
    letter) and counts each resulting piece that looks like a real clause
    (4+ words) as a separate unit for SCORING purposes only. The original,
    unsplit phrase is still what gets stored/displayed - this function only
    affects the Present/Weak/Missing count.
    """
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


def compute_length_normalized_rates(phrases: dict, word_count: int) -> dict:
    """
    Reports phrases-per-100-words per component, alongside (not replacing)
    the existing raw-count-based Present/Weak/Missing score. This is purely
    additive - a longer reflection naturally has more raw material for two
    qualifying phrases regardless of actual reflective depth, so this field
    is what lets that be checked once the human-coded validation subsample
    is available, rather than assuming the raw threshold is fine as-is.
    """
    if word_count <= 0:
        return {c: 0.0 for c in phrases}
    rates = {}
    for component, phrase_list in phrases.items():
        n = sum(count_phrase_units(p) for p in phrase_list if p.strip())
        rates[component] = round((n / word_count) * 100, 2)
    return rates


def apply_hierarchy_gate(scores: dict) -> dict:
    """
    SOFT gate only (per the "pilot before committing to a hard cap"
    decision) - does not change any Present/Weak/Missing value. Flags any
    higher-order component (alternative, future_action, depth) that scored
    Present or Weak while the foundation (description AND evaluation) is
    not solidly established, and assigns an overall reflection "profile"
    label. This is the first step toward the actionable-feedback redesign
    (structured report instead of a flat 7-flag table) discussed separately.
    """
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_reflections_from_json(REFLECTIONS_JSON)

    print(f"Model: {MODEL}  (think mode disabled: {_CHAT_SUPPORTS_THINK})")

    results = []
    components = ["description","emotional","evaluation","analysis",
                  "alternative","future_action","depth"]

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} already processed")

    processed = {r["filename"] + r["week"] for r in results}

    for i, entry in enumerate(data):
        key = entry["filename"] + entry["week"]
        if key in processed:
            continue

        text = entry.get("reflection", "")

        print(f"[{i+1}/{len(data)}] {entry['week']} | {entry['filename'][:45]}")

        phrases, topic = extract_phrases(text)           # v5.1: topic now comes from the same call
        phrases = verify_phrases_against_text(phrases, text, entry["filename"])
        phrases = judge_phrases(text, phrases)          # v5: LLM-as-judge second pass (v5.1: scoped + slimmed)
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

        results.append(entry)

        if len(results) % 10 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(DELAY_SECS)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save fabrication log regardless of whether it's empty, so an empty file
    # is itself evidence the run was clean.
    with open(FABRICATION_FILE, "w", encoding="utf-8") as f:
        json.dump(FABRICATION_LOG, f, ensure_ascii=False, indent=2)
    with open(SOFT_MATCH_FILE, "w", encoding="utf-8") as f:
        json.dump(SOFT_MATCH_LOG, f, ensure_ascii=False, indent=2)
    with open(JUDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(JUDGE_LOG, f, ensure_ascii=False, indent=2)

    if FABRICATION_LOG:
        print(f"\n⚠ WARNING: {len(FABRICATION_LOG)} fabricated (non-verbatim, even after "
              f"normalization) phrase(s) were extracted and dropped. See {FABRICATION_FILE}.")
    else:
        print(f"\nNo fabricated phrases detected this run. {FABRICATION_FILE} is empty.")

    if SOFT_MATCH_LOG:
        print(f"ℹ {len(SOFT_MATCH_LOG)} phrase(s) only matched after quote/case normalization "
              f"(genuine quotes, just cosmetic differences - not fabrication). See {SOFT_MATCH_FILE} "
              f"if you want to spot-check them.")

    if ENABLE_LLM_JUDGE:
        if JUDGE_LOG:
            print(f"ℹ LLM-judge pass rejected {len(JUDGE_LOG)} phrase(s) that were verbatim-real but "
                  f"didn't satisfy their component's definition on second review. See {JUDGE_FILE}.")
        else:
            print(f"\nLLM-judge pass: no rejections this run. {JUDGE_FILE} is empty.")

    print(f"\nExtraction complete. Building stats...\n")

    component_labels = {
        "description":   "Description of event",
        "emotional":     "Emotional response / feelings",
        "evaluation":    "Evaluation (good/bad)",
        "analysis":      "Analysis / making sense",
        "alternative":   "Alternative interpretations",
        "future_action": "Future action / what next",
        "depth":         "Transformative depth / insight",
    }

    total = len(results)
    stats = {
        "total_reflections": total,
        "model_used": MODEL,
        "fabricated_phrases_dropped": len(FABRICATION_LOG),
        "soft_matches_normalized": len(SOFT_MATCH_LOG),
        "judge_rejections": len(JUDGE_LOG),
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
    print(f"OVERALL NORMALISED RESULTS — {total} reflections")
    print("=" * 65)

    for c in components:
        counts = Counter(r["scores"].get(c,"Missing") for r in results)
        p, w, m = counts["Present"], counts["Weak"], counts["Missing"]
        has_phrase = sum(1 for r in results if len(r["phrases"].get(c,[])) > 0)

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
    for r in results:
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

    topic_counter = Counter(r.get("topic","other") for r in results)
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
        profile_counter = Counter(r.get("gating", {}).get("profile", "unknown") for r in results)
        stats["profile_distribution"] = dict(profile_counter.most_common())
        for profile, count in profile_counter.most_common():
            print(f"  {profile:<28} {count:4d}  ({count/total*100:5.1f}%)")

        ungrounded_count = sum(1 for r in results if r.get("gating", {}).get("ungrounded_higher_components"))
        stats["reflections_with_ungrounded_higher_components"] = ungrounded_count
        if ungrounded_count:
            print(f"\n  ⚠ {ungrounded_count} reflection(s) scored a higher-order component "
                  f"without an established foundation (flagged, not altered - see each entry's "
                  f"'gating' field in {OUTPUT_FILE}).")

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*65}")
    print(f"Saved: {OUTPUT_FILE}       (phrases + scores per reflection)")
    print(f"Saved: {STATS_FILE}        (normalised stats, word counts, topics)")
    print(f"Saved: {FABRICATION_FILE}  (any dropped non-verbatim phrases - should be empty or near-empty)")
    print(f"Saved: {SOFT_MATCH_FILE}  (kept phrases that needed quote/case normalization to verify)")
    print(f"Saved: {JUDGE_FILE}  (phrases rejected by the LLM-judge pass, with reasons)")
    print(f"{'='*65}")
    print("\nNEXT STEPS (Spot-check):")
    print("  1. Open phrases_output.json")
    print("  2. Find entries where scores['depth'] == 'Present'")
    print("  3. Manually verify each extracted phrase is genuinely transformative")
    print("  4. Note how many you verified and how many were correct — report this")
    print("  5. Check fabrication_log.json - ideally it should be empty")
    print("  6. Check judge_log.json - skim a sample of rejections to confirm the judge's")
    print("     reasoning holds up on manual read, not just fabrication_log.json")
    print("  7. Check phrases_per_100_words in phrases_output.json against raw scores on a")
    print("     sample to see whether length normalization would change any classifications")


if __name__ == "__main__":
    main()