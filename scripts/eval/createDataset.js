// Syncs scripts/eval/dataset.json to a LangSmith dataset.
// dataset.json is the source of truth; each example is matched by its `slug`
// (stored in example metadata), so re-running is safe:
//   new slug      -> created
//   changed slug  -> updated
//   removed slug  -> deleted from LangSmith
//
// Usage: npm run eval:dataset
require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { Client } = require('langsmith');

const DATASET_FILE = path.join(__dirname, 'dataset.json');

function toExample(e) {
  return {
    inputs: e.inputs,
    outputs: e.outputs,
    metadata: { slug: e.slug, category: e.category },
    split: e.category
  };
}

// Key-order-independent comparison (LangSmith returns keys in its own order),
// so unchanged examples aren't re-uploaded.
function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map((k) => [k, canonical(value[k])]));
  }
  return value;
}
const sameJson = (a, b) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));

function validate(file) {
  const slugs = new Set();
  for (const e of file.examples) {
    if (!e.slug || !e.category || !e.inputs?.message || !e.outputs?.expected_behavior) {
      throw new Error(`Example is missing slug/category/inputs.message/outputs.expected_behavior: ${JSON.stringify(e).slice(0, 120)}`);
    }
    if (slugs.has(e.slug)) throw new Error(`Duplicate slug: ${e.slug}`);
    slugs.add(e.slug);
  }
}

async function main() {
  if (!process.env.LANGSMITH_API_KEY) {
    throw new Error('LANGSMITH_API_KEY is not set in .env');
  }

  const file = JSON.parse(fs.readFileSync(DATASET_FILE, 'utf-8'));
  validate(file);

  const client = new Client();

  const dataset = (await client.hasDataset({ datasetName: file.name }))
    ? await client.readDataset({ datasetName: file.name })
    : await client.createDataset(file.name, { description: file.description });

  const existing = new Map();
  for await (const ex of client.listExamples({ datasetId: dataset.id })) {
    if (ex.metadata?.slug) existing.set(ex.metadata.slug, ex);
  }

  const toCreate = [];
  let updated = 0;
  let unchanged = 0;

  for (const e of file.examples) {
    const want = toExample(e);
    const have = existing.get(e.slug);
    existing.delete(e.slug);

    if (!have) {
      toCreate.push({ ...want, dataset_id: dataset.id });
    } else if (
      !sameJson(have.inputs, want.inputs) ||
      !sameJson(have.outputs, want.outputs) ||
      have.metadata?.category !== e.category
    ) {
      await client.updateExample({ id: have.id, ...want, dataset_id: dataset.id });
      updated++;
    } else {
      unchanged++;
    }
  }

  if (toCreate.length) await client.createExamples(toCreate);

  // Whatever is left in `existing` was removed from dataset.json.
  for (const ex of existing.values()) await client.deleteExample(ex.id);

  console.log(
    `Dataset "${file.name}": ${toCreate.length} created, ${updated} updated, ` +
      `${existing.size} deleted, ${unchanged} unchanged (${file.examples.length} total).`
  );
}

main().catch((err) => {
  console.error('Dataset sync failed:', err.message || err);
  process.exit(1);
});
