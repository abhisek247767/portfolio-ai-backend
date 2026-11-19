// Runs one LangSmith experiment over the "portfolio-qa" dataset.
//
// Usage:
//   npm run eval                                   # qwen2.5:3b answers, tev1:0.8b judges
//   npm run eval -- --model qwen2.5:7b
//   npm run eval -- --provider gemini              # production Gemini model from .env
//   npm run eval -- --judge-model tev1             # larger 4B tev1 judge
//   npm run eval -- --limit 2                      # quick smoke test on 2 questions
//   npm run eval -- --repetitions 2                # run each question twice
//
// Retrieval always uses the real pipeline (data/embeddings.json + Gemini query
// embeddings), so only the answering model changes between experiments.
require('dotenv').config();
const path = require('path');
const { Client } = require('langsmith');
const { evaluate } = require('langsmith/evaluation');
const { initializeRag, retrieveChunks, buildPrompt } = require('../../src/rag');
const { generateChatCompletion } = require('../../src/geminiClient');
const { ollamaChat, OLLAMA_URL } = require('../../src/llm/ollama');
const { systemPrompt, generationConfig } = require('../../src/config/generationConfig');
const { containsExpected, retrievalHit, makeJudge } = require('./evaluators');
const datasetFile = require('./dataset.json');

// ---------- CLI args ----------
function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const provider = arg('provider', 'ollama'); // ollama | gemini
const model = provider === 'gemini' ? process.env.GEMINI_CHAT_MODEL : arg('model', 'qwen2.5:3b');
const judgeModel = arg('judge-model', 'tev1:0.8b');
const limit = parseInt(arg('limit', '0'), 10) || 0;
const repetitions = parseInt(arg('repetitions', '1'), 10) || 1;
// CPU-only Ollama can't usefully run requests in parallel.
const concurrency = parseInt(arg('concurrency', provider === 'gemini' ? '2' : '1'), 10);

if (!['ollama', 'gemini'].includes(provider)) {
  console.error(`Unknown --provider "${provider}". Use ollama or gemini.`);
  process.exit(1);
}

// ---------- Answering model ----------
async function answerWithModel(prompt) {
  if (provider === 'gemini') {
    // Exactly what production does (same function /chat uses).
    return generateChatCompletion(prompt);
  }
  // Same prompt shape and sampling settings as production, sent to the local model.
  return ollamaChat({
    model,
    runName: 'answer',
    messages: [{ role: 'user', content: `${systemPrompt}\n\nUser Question: ${prompt}` }],
    options: {
      temperature: generationConfig.temperature,
      top_p: generationConfig.topP,
      top_k: generationConfig.topK,
      num_predict: generationConfig.maxOutputTokens
    }
  });
}

// The function LangSmith calls once per example.
async function target(inputs) {
  const chunks = await retrieveChunks(inputs.message);
  const prompt = buildPrompt(inputs.message, inputs.history, chunks);
  const answer = await answerWithModel(prompt);
  return {
    answer,
    contexts: chunks.map((c) => c.text),
    retrieved: chunks.map((c) => ({ id: c.id, score: Number(c.score.toFixed(4)) }))
  };
}

// ---------- Preflight ----------
async function ensureOllamaModels(models) {
  let tags;
  try {
    tags = await (await fetch(`${OLLAMA_URL}/api/tags`)).json();
  } catch {
    throw new Error(`Cannot reach Ollama at ${OLLAMA_URL}. Start it with "ollama serve".`);
  }
  const installed = new Set((tags.models || []).flatMap((m) => [m.name, m.model]));
  const missing = models.filter((m) => !installed.has(m) && !installed.has(`${m}:latest`));
  if (missing.length) {
    throw new Error(`Missing Ollama model(s): ${missing.join(', ')}. Run: ${missing.map((m) => `ollama pull ${m}`).join(' && ')}`);
  }
}

async function main() {
  if (!process.env.LANGSMITH_API_KEY) throw new Error('LANGSMITH_API_KEY is not set in .env');

  await ensureOllamaModels(provider === 'ollama' ? [model, judgeModel] : [judgeModel]);

  const pdfPath = process.env.PROFILE_PDF_PATH
    ? path.resolve(process.env.PROFILE_PDF_PATH)
    : path.join(__dirname, '..', '..', 'data', 'abhisek_profile.pdf');
  await initializeRag(pdfPath);

  const client = new Client();
  let data = datasetFile.name;
  if (limit > 0) {
    const examples = [];
    for await (const ex of client.listExamples({ datasetName: datasetFile.name })) examples.push(ex);
    examples.sort((a, b) => String(a.metadata?.slug).localeCompare(String(b.metadata?.slug)));
    data = examples.slice(0, limit);
  }

  console.log(
    `Running experiment: answer=${provider}/${model} judge=ollama/${judgeModel} ` +
      `examples=${limit || 'all'} repetitions=${repetitions} concurrency=${concurrency}`
  );

  const results = await evaluate(target, {
    data,
    client,
    evaluators: [containsExpected, retrievalHit, makeJudge({ model: judgeModel })],
    experimentPrefix: `${model}--judge-${judgeModel}`.replace(/[^\w.-]+/g, '-'),
    description: `Answers by ${provider}/${model}, judged by ollama/${judgeModel}`,
    metadata: {
      answer_provider: provider,
      answer_model: model,
      judge_model: judgeModel,
      embedding_model: process.env.GEMINI_EMBEDDING_MODEL,
      top_k: 5
    },
    maxConcurrency: concurrency,
    numRepetitions: repetitions
  });

  // ---------- Summary in the terminal ----------
  const totals = {};
  for (const row of results.results) {
    for (const r of row.evaluationResults.results) {
      if (r.score === null || r.score === undefined) continue;
      totals[r.key] ??= { sum: 0, n: 0 };
      totals[r.key].sum += Number(r.score);
      totals[r.key].n += 1;
    }
  }
  console.log('\nScores (higher is better):');
  for (const [key, { sum, n }] of Object.entries(totals)) {
    console.log(`  ${key.padEnd(18)} ${((sum / n) * 100).toFixed(0).padStart(3)}%  (${sum}/${n})`);
  }
  console.log('  (judge scores: good = 1, moderate = 0.5, fail = 0)');
  console.log(`\nExperiment: ${results.experimentName}`);
  console.log('Open smith.langchain.com -> Datasets & Experiments -> portfolio-qa to inspect and compare.');
}

main().catch((err) => {
  console.error('Eval failed:', err.message || err);
  process.exit(1);
});
