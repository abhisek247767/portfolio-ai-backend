const path = require('path');
const { loadPdfText } = require('./pdfLoader');
const { chunkText } = require('./textUtils');
const {
  loadOrCreateEmbeddingsStore,
  getTopKRelevantChunks
} = require('./embeddingsStore');
const { generateChatCompletion } = require('./geminiClient');

let embeddingsStore = null;
let initialized = false;

function isInitialized() {
  return initialized;
}

async function initializeRag(pdfFilePath) {
  try {
  const absolutePdfPath = path.resolve(pdfFilePath);
  const pdfText = await loadPdfText(absolutePdfPath);

    const chunks = chunkText(pdfText, 1000, 200);

    embeddingsStore = await loadOrCreateEmbeddingsStore(chunks);
    initialized = true;
  } catch (error) {
    console.error('Error initializing RAG:', error);
    throw new Error('Failed to initialize RAG');
  }
}

async function getAnswerForQuery(message, history) {
  if (!initialized || !embeddingsStore) {
    throw new Error('RAG not initialized yet');
  }

  const topChunks = await getTopKRelevantChunks(embeddingsStore, message, 5);

  const contextText = topChunks
    .map((c, idx) => `Chunk ${idx + 1}:\n${c.text}`)
    .join('\n\n');

  const systemInstruction =
    'You are a helpful AI assistant that answers questions about Abhisek Roy using ONLY the provided context. ' +
    'If the answer is not clearly contained in the context, say you do not know and suggest the user ask something related to the provided information.';

  const conversationHistory = Array.isArray(history) ? history : [];

  const prompt = `
${systemInstruction}

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

Please answer in a friendly, concise way suitable for a personal portfolio website visitor.
  `.trim();

  const reply = await generateChatCompletion(prompt);
  return reply;
}

module.exports = {
  initializeRag,
  getAnswerForQuery,
  isInitialized
};


