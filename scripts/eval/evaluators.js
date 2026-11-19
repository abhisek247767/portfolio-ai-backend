// Evaluators for the portfolio chatbot.
// Each receives { inputs, outputs, referenceOutputs } from LangSmith's evaluate():
//   inputs           = example inputs   { message, history }
//   outputs          = what the target returned { answer, contexts }
//   referenceOutputs = example outputs  { expected_behavior, reference_answer, must_contain, ... }
const { ollamaDecide } = require('../../src/llm/ollama');

const lower = (s) => String(s || '').toLowerCase();
const includesCI = (text, needle) => lower(text).includes(lower(needle));

// ---------- Code evaluators (free, instant, deterministic) ----------

/** 1 if every required phrase is present, at least one "any" phrase is present,
 *  and no forbidden phrase is present (all case-insensitive). */
function containsExpected({ outputs, referenceOutputs: ref }) {
  const answer = outputs?.answer || '';
  const problems = [];

  for (const p of ref.must_contain || []) {
    if (!includesCI(answer, p)) problems.push(`missing "${p}"`);
  }
  const any = ref.must_contain_any || [];
  if (any.length && !any.some((p) => includesCI(answer, p))) {
    problems.push(`missing one of [${any.join(' | ')}]`);
  }
  for (const p of ref.must_not_contain || []) {
    if (includesCI(answer, p)) problems.push(`contains forbidden "${p}"`);
  }

  return {
    key: 'contains_expected',
    score: problems.length === 0 ? 1 : 0,
    comment: problems.join('; ') || 'all checks passed'
  };
}

/** 1 if the retrieved chunks contain the text the answer depends on.
 *  Separates "retrieval failed" from "model failed". */
function retrievalHit({ outputs, referenceOutputs: ref }) {
  const needles = ref.retrieval_must_contain || [];
  if (!needles.length) {
    return { key: 'retrieval_hit', score: null, comment: 'not applicable' };
  }
  const context = (outputs?.contexts || []).join('\n');
  const missing = needles.filter((n) => !includesCI(context, n));
  return {
    key: 'retrieval_hit',
    score: missing.length === 0 ? 1 : 0,
    comment: missing.length ? `not retrieved: ${missing.join(', ')}` : 'all retrieved'
  };
}

// ---------- Judge: tev1 decision model (local, via Ollama /v1/systemone) ----------

// Grading bands for tev1's "yes" probability:
//   >= 0.80        -> good      (score 1)
//   0.50 - 0.80    -> moderate  (score 0.5)
//   <  0.50        -> fail      (score 0)
const GOOD_AT = 0.8;
const MODERATE_AT = 0.5;

function grade(probability) {
  if (probability >= GOOD_AT) return { label: 'good', score: 1 };
  if (probability >= MODERATE_AT) return { label: 'moderate', score: 0.5 };
  return { label: 'fail', score: 0 };
}

const BEHAVIOR_RULES = {
  answer: 'Answer the question using the profile.',
  decline_unknown:
    'The information is NOT in the profile: say it is not mentioned / not known, and do NOT invent it.',
  redirect_off_topic:
    'The question is unrelated to Abhisek: politely decline and steer back to his profile, and do NOT answer the question.'
};

// One tev1 call asks all three yes/no ("noul") questions about the same state.
const JUDGE_QUESTIONS = {
  correct: {
    type: 'noul',
    instructions:
      'Does chatbot_answer agree with reference_answer on the key facts, with no wrong or contradicting facts?'
  },
  grounded: {
    type: 'noul',
    instructions:
      'Is every factual claim about Abhisek in chatbot_answer supported by context? Any invented fact makes this false.'
  },
  behavior_ok: {
    type: 'noul',
    instructions: 'Does chatbot_answer follow expected_behavior?'
  }
};

// tev1 reads only ~2,000 tokens, so cap the longest fields to keep the whole
// state inside its context window.
const MAX_CONTEXT_CHARS = 4500;
const MAX_ANSWER_CHARS = 1500;
const cap = (text, max) => (text.length > max ? `${text.slice(0, max)} [truncated]` : text);

/** Builds the judge evaluator. Default model: tev1:0.8b. */
function makeJudge({ model = 'tev1:0.8b' } = {}) {
  return async function tev1Judge({ inputs, outputs, referenceOutputs: ref }) {
    if (ref.judge_question) {
      let p;
      try {
        const answers = await ollamaDecide({
          model,
          state: { chatbot_answer: cap(outputs?.answer || '', MAX_ANSWER_CHARS) },
          questions: { check: { type: 'noul', instructions: ref.judge_question } },
          runName: 'judge'
        });
        p = answers.check?.noul;
      } catch (err) {
        p = undefined;
      }
      const na = { score: null, comment: 'not applicable (judged by judge_question)' };
      if (typeof p !== 'number') {
        return {
          results: [
            { key: 'correct', ...na },
            { key: 'grounded', ...na },
            { key: 'behavior_ok', score: null, comment: 'judge failed' }
          ]
        };
      }
      // tev1 gives P(yes). If we expect "no", the chance the answer is right is 1 - P(yes).
      const pOk = ref.judge_expect === false ? 1 - p : p;
      const { label, score } = grade(pOk);
      return {
        results: [
          { key: 'correct', ...na },
          { key: 'grounded', ...na },
          {
            key: 'behavior_ok',
            score,
            value: label,
            comment: `${label} (p_ok=${pOk.toFixed(2)}; tev1 said yes=${p.toFixed(2)} to "${ref.judge_question}")`
          }
        ]
      };
    }
    const state = {
      question: inputs.message,
      history: (inputs.history || []).map((m) => `${m.role}: ${m.content}`).join('\n') || '(none)',
      expected_behavior: BEHAVIOR_RULES[ref.expected_behavior] || BEHAVIOR_RULES.answer,
      reference_answer: ref.reference_answer,
      context: cap((outputs?.contexts || []).join('\n---\n'), MAX_CONTEXT_CHARS),
      chatbot_answer: cap(outputs?.answer || '', MAX_ANSWER_CHARS)
    };

    let answers;
    try {
      answers = await ollamaDecide({ model, state, questions: JUDGE_QUESTIONS, runName: 'judge' });
    } catch (err) {
      // Record the failure instead of crashing the whole experiment.
      const comment = `judge failed: ${err.message}`;
      return { results: Object.keys(JUDGE_QUESTIONS).map((key) => ({ key, score: null, comment })) };
    }

    return {
      results: Object.keys(JUDGE_QUESTIONS).map((key) => {
        const p = answers[key]?.noul;
        if (typeof p !== 'number') {
          return { key, score: null, comment: `no probability returned: ${JSON.stringify(answers[key])}` };
        }
        const { label, score } = grade(p);
        // LangSmith shows `score` in charts; `value` and `comment` keep the band and raw probability.
        return { key, score, value: label, comment: `${label} (p=${p.toFixed(2)})` };
      })
    };
  };
}

module.exports = { containsExpected, retrievalHit, makeJudge, grade };
