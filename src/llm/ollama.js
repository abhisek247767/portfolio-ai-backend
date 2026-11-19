// Minimal client for a local Ollama server (https://ollama.com).
// Used by the evaluation runner, both to answer questions and to judge answers.
const { traceable } = require('langsmith/traceable');

const OLLAMA_URL = (process.env.OLLAMA_URL || 'http://localhost:11434').replace(/\/$/, '');

async function ollamaChatRaw({ model, messages, options = {}, format, timeoutMs = 300000 }) {
  let res;
  try {
    res = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, messages, options, format, stream: false }),
      signal: AbortSignal.timeout(timeoutMs)
    });
  } catch (err) {
    throw new Error(`Cannot reach Ollama at ${OLLAMA_URL} (${err.message}). Is "ollama serve" running?`);
  }

  if (!res.ok) {
    const body = await res.text();
    if (res.status === 404) {
      throw new Error(`Ollama model "${model}" not found. Run: ollama pull ${model}`);
    }
    throw new Error(`Ollama error ${res.status}: ${body.slice(0, 300)}`);
  }

  const data = await res.json();
  return {
    text: data.message?.content ?? '',
    // Token counts so LangSmith can show usage for local models too.
    usage_metadata: {
      input_tokens: data.prompt_eval_count ?? 0,
      output_tokens: data.eval_count ?? 0,
      total_tokens: (data.prompt_eval_count ?? 0) + (data.eval_count ?? 0)
    }
  };
}

/**
 * Traced chat call. Returns the reply text.
 * @param {string} model     e.g. "qwen2.5:3b"
 * @param {Array}  messages  [{ role: 'system'|'user'|'assistant', content }]
 */
function ollamaChat({ model, messages, options, format, runName = 'ollama-chat' }) {
  const traced = traceable(ollamaChatRaw, {
    name: runName,
    run_type: 'llm',
    metadata: { ls_provider: 'ollama', ls_model_name: model },
    processOutputs: (out) => ({ content: out.text, usage_metadata: out.usage_metadata })
  });
  return traced({ model, messages, options, format }).then((out) => out.text);
}

async function ollamaDecideRaw({ model, state, questions, timeoutMs = 120000 }) {
  let res;
  try {
    res = await fetch(`${OLLAMA_URL}/v1/systemone`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, state, questions }),
      signal: AbortSignal.timeout(timeoutMs)
    });
  } catch (err) {
    throw new Error(`Cannot reach Ollama at ${OLLAMA_URL} (${err.message}). Is "ollama serve" running?`);
  }
  const body = await res.text();
  if (!res.ok) {
    if (res.status === 404 && /model/i.test(body)) {
      throw new Error(`Ollama model "${model}" not found. Run: ollama pull ${model}`);
    }
    throw new Error(`Ollama systemone error ${res.status}: ${body.slice(0, 300)}`);
  }
  const data = JSON.parse(body);
  if (!data.answers) throw new Error(`Unexpected systemone response: ${body.slice(0, 200)}`);
  return data.answers;
}

/**
 * Decision-model call (e.g. tev1) via Ollama's /v1/systemone endpoint.
 * Decision models don't chat: they take a `state` (data to judge) and named
 * `questions`, and return per-question answers/probabilities.
 * @returns {Promise<object>} answers keyed by question name, e.g. { grounded: { noul: 0.93 } }
 */
function ollamaDecide({ model, state, questions, runName = 'ollama-decide' }) {
  const traced = traceable(ollamaDecideRaw, {
    name: runName,
    run_type: 'llm',
    metadata: { ls_provider: 'ollama', ls_model_name: model }
  });
  return traced({ model, state, questions });
}

module.exports = { ollamaChat, ollamaDecide, OLLAMA_URL };
