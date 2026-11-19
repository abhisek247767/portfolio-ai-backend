const path = require('path');
const { loadPdfText } = require('./pdfLoader');
const { RecursiveCharacterTextSplitter } = require('@langchain/textsplitters');
const {
  loadOrCreateVectorStore,
  getTopKRelevantChunks
} = require('./vectorStore');
const { generateChatCompletion } = require('./geminiClient');
const { traceable } = require('langsmith/traceable');
const { langsmithClient } = require('./tracing');

let embeddingsStore = null;
let initialized = false;

function isInitialized() {
  return initialized;
}

const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 200;

// Splits on paragraphs -> lines -> words, so chunks don't cut mid-sentence.
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP
});

async function splitIntoChunks(text) {
  const parts = await splitter.splitText(text);
  return parts.map((p) => p.replace(/[ \t]+/g, ' ').trim()).filter(Boolean);
}

async function initializeRag(pdfFilePath, { force = false } = {}) {
  try {
    const absolutePdfPath = path.resolve(pdfFilePath);
    const pdfText = await loadPdfText(absolutePdfPath);

    const chunks = await splitIntoChunks(pdfText);

    embeddingsStore = await loadOrCreateVectorStore(
      chunks,
      {
        source: path.basename(absolutePdfPath),
        splitter: 'RecursiveCharacterTextSplitter',
        chunkSize: CHUNK_SIZE,
        chunkOverlap: CHUNK_OVERLAP
      },
      { force }
    );
    initialized = true;
  } catch (error) {
    console.error('Error initializing RAG:', error);
    throw new Error('Failed to initialize RAG');
  }
}

const TOP_K = 5;

// Shared by /chat and the evaluation runner, so evals test the real pipeline.
async function retrieveChunks(message, k = TOP_K) {
  if (!initialized || !embeddingsStore) {
    throw new Error('RAG not initialized yet');
  }
  return getTopKRelevantChunks(embeddingsStore, message, k);
}

function buildPrompt(message, history, topChunks) {
  const contextText = topChunks
    .map((c, idx) => `Chunk ${idx + 1}:\n${c.text}`)
    .join('\n\n');

  const conversationHistory = Array.isArray(history) ? history : [];

  return `
Context about Abhisek:
${contextText}

Conversation so far:
${conversationHistory
  .map(
    (m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`
  )
  .join('\n')}

New user question:
${message}
`.trim();
}

async function getAnswerForQuery(message, history) {
  const topChunks = await retrieveChunks(message);
  const prompt = buildPrompt(message, history, topChunks);
  return generateChatCompletion(prompt);
}

module.exports = {
  initializeRag,
  getAnswerForQuery: traceable(getAnswerForQuery, {
    name: 'portfolio-chat',
    run_type: 'chain',
    client: langsmithClient,
    metadata: { chatModel: process.env.GEMINI_CHAT_MODEL },
    processInputs: (inputs) => ({ message: inputs.args?.[0], history: inputs.args?.[1] })
  }),
  isInitialized,
  retrieveChunks,
  buildPrompt
};


