const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { GoogleGenerativeAIEmbeddings } = require('@langchain/google-genai');
const { MemoryVectorStore } = require('@langchain/classic/vectorstores/memory');
const { Document } = require('@langchain/core/documents');

const DATA_DIR = path.join(__dirname, '..', 'data');
const STORE_FILE = path.join(DATA_DIR, 'embeddings.json');

// Bump when the cache format or embedding logic changes, to force a rebuild.
const STORE_VERSION = 3;

// Gemini embeds documents and queries differently for better retrieval.
const DOC_TASK_TYPE = 'RETRIEVAL_DOCUMENT';
const QUERY_TASK_TYPE = 'RETRIEVAL_QUERY';

function createEmbeddings(taskType) {
  return new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: process.env.GEMINI_EMBEDDING_MODEL,
    taskType
  });
}

// Hash of everything that determines the vectors. If the PDF text, chunk
// settings or embedding model change, the hash changes -> rebuild.
function computeFingerprint(chunks, meta) {
  return crypto
    .createHash('sha256')
    .update(JSON.stringify({ v: STORE_VERSION, ...meta, chunks }))
    .digest('hex');
}

function readCache() {
  if (!fs.existsSync(STORE_FILE)) return null;
  try {
    return JSON.parse(fs.readFileSync(STORE_FILE, 'utf-8'));
  } catch (err) {
    console.warn('Embeddings cache is unreadable, rebuilding:', err.message);
    return null;
  }
}

const hasVector = (v) => Array.isArray(v) && v.length > 0;

function isCacheValid(cache, fingerprint) {
  return (
    cache &&
    cache.fingerprint === fingerprint &&
    Array.isArray(cache.chunks) &&
    cache.chunks.length > 0 &&
    cache.chunks.every((c) => hasVector(c.embedding))
  );
}

// Write to a temp file, then rename over the real one. If the process crashes
// mid-write, only the .tmp file is broken; embeddings.json stays intact.
function writeCacheAtomic(cache) {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
  const tmp = `${STORE_FILE}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(cache, null, 2), 'utf-8');
  fs.renameSync(tmp, STORE_FILE);
}

/**
 * Builds a MemoryVectorStore for the given chunks. Reuses the vectors in
 * data/embeddings.json when the fingerprint matches (no Gemini calls);
 * otherwise embeds the chunks and rewrites the cache.
 */
async function loadOrCreateVectorStore(chunks, meta = {}, { force = false } = {}) {
  const fullMeta = {
    embeddingModel: process.env.GEMINI_EMBEDDING_MODEL || null,
    docTaskType: DOC_TASK_TYPE,
    ...meta
  };
  const fingerprint = computeFingerprint(chunks, fullMeta);

  // The store's own embeddings object is used to embed incoming queries.
  const store = new MemoryVectorStore(createEmbeddings(QUERY_TASK_TYPE));
  const docs = chunks.map(
    (text, id) => new Document({ pageContent: text, metadata: { id, source: fullMeta.source } })
  );

  const cache = readCache();
  if (!force && isCacheValid(cache, fingerprint)) {
    await store.addVectors(cache.chunks.map((c) => c.embedding), docs);
    console.log(`Embeddings cache up to date (${docs.length} chunks).`);
    return store;
  }

  console.log(force ? 'Forced rebuild of embeddings...' : 'Source changed; rebuilding embeddings...');

  const vectors = await createEmbeddings(DOC_TASK_TYPE).embedDocuments(chunks);
  // LangChain returns [] for a failed batch instead of throwing; never cache that.
  if (vectors.length !== chunks.length || !vectors.every(hasVector)) {
    throw new Error('Embedding failed for one or more chunks; cache not written');
  }

  await store.addVectors(vectors, docs);

  writeCacheAtomic({
    version: STORE_VERSION,
    fingerprint,
    ...fullMeta,
    createdAt: new Date().toISOString(),
    chunks: chunks.map((text, id) => ({ id, text, embedding: vectors[id] }))
  });
  console.log(`Embeddings rebuilt (${docs.length} chunks).`);
  return store;
}

async function getTopKRelevantChunks(store, query, k = 5) {
  const results = await store.similaritySearchWithScore(query, k);
  return results.map(([doc, score]) => ({
    id: doc.metadata.id,
    text: doc.pageContent,
    score
  }));
}

module.exports = {
  loadOrCreateVectorStore,
  getTopKRelevantChunks
};
