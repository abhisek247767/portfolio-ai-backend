require("dotenv").config();
const { GoogleGenAI } = require("@google/genai");
const { systemPrompt, generationConfig } = require("./config/generationConfig");

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_CHAT_MODEL = process.env.GEMINI_CHAT_MODEL;
const GEMINI_EMBEDDING_MODEL = process.env.GEMINI_EMBEDDING_MODEL;

if (!GEMINI_API_KEY || !GEMINI_CHAT_MODEL || !GEMINI_EMBEDDING_MODEL) {
  console.warn("One or more required GEMINI environment variables are not set.");
}

const ai = new GoogleGenAI({
  apiKey: GEMINI_API_KEY,
});

async function getEmbeddingForText(text) {
  try {
    const response = await ai.models.embedContent({
      model: GEMINI_EMBEDDING_MODEL,
      contents: text,
    });

    return response.embeddings.values;
  } catch (error) {
    console.error("Error generating embedding:", error);
    throw error;
  }
}




async function generateChatCompletion(userMessage) {
  try {
    const response = await ai.models.generateContent({
      model: GEMINI_CHAT_MODEL,

      contents: [
        {
          role: "user",
          parts: [
            {
              text: `${systemPrompt}\n\nUser Question: ${userMessage}`,
            },
          ],
        },
      ],

      generationConfig,
    });

    return response.text;
  } catch (error) {
    console.error("Error generating chat completion:", error);
    return `- Apologies, something went wrong.
- Please try again in a moment.
- If the issue persists, feel free to contact Abhisek directly.`;
  }
}

module.exports = {
  getEmbeddingForText,
  generateChatCompletion,
};


