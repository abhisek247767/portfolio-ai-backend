const fs = require('fs');
const path = require('path');
const { getEmbeddingForText } = require('./geminiClient');

const DATA_DIR = path.join(__dirname, '..', 'data');
const STORE_FILE = path.join(DATA_DIR, 'embeddings.json');

function ensureDataDir() {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
}

async function loadOrCreateEmbeddingsStore(chunks) {
  ensureDataDir();

  if (fs.existsSync(STORE_FILE)) {
    const raw = fs.readFileSync(STORE_FILE, 'utf-8');
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed.chunks) && parsed.chunks.length > 0) {
      return parsed;
    }
  }

  const store = {
    createdAt: new Date().toISOString(),
    chunks: []
  };

  for (let i = 0; i < chunks.length; i++) {
    const text = chunks[i];
    const embedding = await getEmbeddingForText(text);
    store.chunks.push({
      id: i,
      text,
      embedding
    });
  }

  fs.writeFileSync(STORE_FILE, JSON.stringify(store, null, 2), 'utf-8');
  return store;
}

function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (!normA || !normB) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function getTopKRelevantChunks(store, query, k = 5) {
  const queryEmbedding = await getEmbeddingForText(query);

  const scored = store.chunks.map((chunk) => ({
    ...chunk,
    score: cosineSimilarity(queryEmbedding, chunk.embedding)
  }));

  scored.sort((a, b) => b.score - a.score);

  return scored.slice(0, k);
}

module.exports = {
  loadOrCreateEmbeddingsStore,
  getTopKRelevantChunks
};


