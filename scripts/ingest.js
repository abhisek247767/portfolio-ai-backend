// Rebuilds data/embeddings.json from the profile PDF, without starting the server.
// Usage:
//   npm run ingest                      # rebuild only if the PDF changed
//   npm run ingest -- --force           # always rebuild
//   npm run ingest -- path/to/file.pdf  # use a different PDF
require('dotenv').config();
const path = require('path');
const { initializeRag } = require('../src/rag');

const args = process.argv.slice(2);
const force = args.includes('--force');
const pdfArg = args.find((a) => !a.startsWith('--'));

const pdfPath = pdfArg
  ? path.resolve(pdfArg)
  : process.env.PROFILE_PDF_PATH
    ? path.resolve(process.env.PROFILE_PDF_PATH)
    : path.join(__dirname, '..', 'data', 'abhisek_profile.pdf');

console.log(`Ingesting ${pdfPath}${force ? ' (forced)' : ''}`);

initializeRag(pdfPath, { force })
  .then(() => console.log('Ingest complete.'))
  .catch((err) => {
    console.error('Ingest failed:', err.message || err);
    process.exit(1);
  });
