const { Client } = require('langsmith');
const { createAnonymizer } = require('langsmith/anonymizer');

const anonymizer = createAnonymizer([
  { pattern: /[\w.+-]+@[\w-]+\.[\w.]+/g, replace: '<email>' },
  { pattern: /\+\d{1,3}\s*-?\s*\d{10}\b|\b\d{10}\b/g, replace: '<phone>' }
]);

const langsmithClient = new Client({ anonymizer });

module.exports = { langsmithClient };
