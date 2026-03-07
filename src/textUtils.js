function chunkText(text, chunkSize = 1000, overlap = 200) {
  const normalized = text.replace(/\s+/g, ' ').trim();
  if (!normalized) return [];

  const chunks = [];
  let start = 0;

  while (start < normalized.length) {
    const end = Math.min(start + chunkSize, normalized.length);
    const chunk = normalized.slice(start, end);
    chunks.push(chunk);

    if (end === normalized.length) break;
    start = end - overlap;
    if (start < 0) start = 0;
  }

  return chunks;
}

module.exports = {
  chunkText
};


