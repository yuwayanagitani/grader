# 🧠 AI Grader for Anki (AI Grade Assigner)

**AI Grader** (a.k.a **AI Grade Assigner**) is an Anki add-on that uses AI to automatically **evaluate your flashcard answers** and assign a grade, helping you optimize review timing and identify spots for improvement.

---

## 🔗 AnkiWeb Page

This add-on is officially published on **AnkiWeb**:

👉 https://ankiweb.net/shared/info/1150481237

Installing from AnkiWeb is recommended for the easiest setup and **automatic updates**.

---

## 🎯 Overview

AI Grader enhances your review workflow by:

- 📝 Using AI to **grade card responses** based on your typed answer  
- 📊 Providing consistent, context-aware grading beyond manual selection  
- 🔄 Supporting batch grading for cards selected in the Browser  
- ⚙️ Allowing configuration of grading criteria and output fields

Instead of relying solely on manual buttons (Again/Hard/Good/Easy), this add-on can **suggest grades** using language models, ideal for self-study, language labs, or complex open-ended answer cards.

---

## 🚀 How It Works

1. During review or in the Browser, trigger **AI Grader**.  
2. The add-on sends your card question + your answer to the AI provider.  
3. The model returns a suggested grade (e.g., “Good”, “Hard”).  
4. The grade can be written to your Anki fields or used to inform your manual selection.

This can help you:

- Get instant feedback when you’re unsure about your recall strength.  
- Review more consistently across study sessions.  
- Spot patterns in mistakes and adjust your focus.

---

## 📦 Installation

### ⬇️ From AnkiWeb (Recommended)

1. Open **Anki**  
2. Go to **Tools → Add-Ons → Browse & Install**  
3. Search for **AI Grade Assigner**  
4. Install and **restart Anki**

### 📁 Manual Installation (GitHub)

1. Clone or download this repository  
2. Place it into:
   `Anki2/addons21/anki-ai-grader`  
3. Restart **Anki**

---

## 🔑 API Key Setup

AI Grader needs an API key for the selected AI provider.

| Provider | Environment Variable |
|----------|----------------------|
| OpenAI | `OPENAI_API_KEY` |
| Gemini | `GEMINI_API_KEY` |

API keys can be set via:
- System environment variables, or  
- The add-on **Settings / Config** dialog

---

## ⚙️ Configuration

Open:

**Tools → Add-Ons → AI Grader → Config**

Key options include:

- Provider selection (OpenAI / Gemini)  
- Model name  
- Which fields to read for question/answer  
- Output destination (graded field, tag appending, note update)  
- Grading style and thresholds

---

## 🧪 Usage

### Review Mode

While reviewing:

1. Answer the card as usual  
2. Trigger **AI Grader** from the Reviewer menu  
3. View the suggested grade  
4. Accept or manually override

### Batch Grading

In the Browser:

1. Select multiple cards  
2. Run **AI Grader: Grade selected cards**  
3. Suggested grades will be written to configured fields or tags

This is useful when you want to retrospectively evaluate a deck or track performance over time.

---

## ⚠️ Notes on Privacy

Card contents (questions and your answers) are sent to external AI services for grading.  
Avoid sending **sensitive or personal information** unless you understand the provider’s privacy policy.

---

## 🛠 Troubleshooting

| Issue | Solution |
|-------|----------|
| “No current card” | Run during review or select cards in Browser |
| Grades not applied | Check field mapping in Config |
| API errors | Verify API key and internet connection |
| Slow response | Choose a lighter AI model |

---

## 📜 License

MIT License

---

## 🔧 Related Add-Ons

These add-ons form an AI-powered ecosystem for Anki:

- **AI Card Explainer** — Generates natural explanations  
- **AI Card Translator** — Translates cards on the fly  
- **AI Card Splitter** — Breaks complex cards into pieces  
- **HTML Exporter for Anki** — Export cards to HTML / PDF

Together they enhance your study flow with **AI-assisted quality, understanding, and review automation**.
