# AI Type Grader — Configuration (config.md)

This add-on can grade **typed answers** (0–100%) using **OpenAI**, **Gemini**, or a **local** fallback scorer.

To keep the order stable in Anki’s Config editor, this version supports **numbered keys** like `01. enabled`.
It also accepts the **same keys without numbers** (backward compatible), e.g. `enabled`.

---

## Where to edit

Anki → **Tools** → **Add-ons** → **AI Type Grader** → **Config**

---

## Default config (numbered keys)

```json
{
  "01. enabled": true,

  "02. provider": "auto",
  "03. provider_auto_order": ["openai", "gemini"],

  "10. openai_api_key": "",
  "11. openai_model": "gpt-4o-mini",
  "12. openai_api_url": "https://api.openai.com/v1/responses",

  "20. gemini_api_key": "",
  "21. gemini_model": "gemini-2.5-flash-lite",
  "22. gemini_api_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",

  "30. answer_field": "Back",

  "31. easy_min": 80,
  "32. good_min": 60,
  "33. hard_min": 40,

  "40. show_tooltip": true,
  "41. show_on_card": true,

  "50. auto_answer": true,

  "60. timeout_sec": 20,
  "61. max_output_tokens": 16
}
```

---

## Key reference

### 01. enabled
Turns the add-on ON/OFF.

- `true`: enabled  
- `false`: disabled (no grading, no badge, no ease override)

---

### 02. provider
Which scorer to use.

- `"openai"`: OpenAI only (falls back to local if OpenAI fails)
- `"gemini"`: Gemini only (falls back to local if Gemini fails)
- `"auto"`: try providers in `03. provider_auto_order` (skips providers that have no API key)
- `"local"`: **local-only** (no external requests)

---

### 03. provider_auto_order
Only used when `02. provider` is `"auto"`.

Example: try Gemini first, then OpenAI:

```json
"03. provider_auto_order": ["gemini", "openai"]
```

---

## OpenAI settings

### 10. openai_api_key
OpenAI API key.

- If empty, the add-on will try the environment variable:
  - `OPENAI_API_KEY`

### 11. openai_model
Example: `gpt-4o-mini`

### 12. openai_api_url
Default: `https://api.openai.com/v1/responses`

---

## Gemini settings

### 20. gemini_api_key
Gemini API key.

- If empty, the add-on will try environment variables:
  - `GEMINI_API_KEY`
  - `GOOGLE_API_KEY`

### 21. gemini_model
Default: `gemini-2.5-flash-lite`

### 22. gemini_api_url
Default:
`https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`

`{model}` is replaced with `21. gemini_model`.

---

## Scoring settings

### 30. answer_field
The **note field name** used as the “model answer” (default: `Back`).

If your note type stores the model answer elsewhere, change this to match your field name.

---

## Thresholds: score → ease (Again/Hard/Good/Easy)

### 31. easy_min
Score ≥ this → **Easy**

### 32. good_min
Score ≥ this → **Good**

### 33. hard_min
Score ≥ this → **Hard**

Score < `33. hard_min` → **Again**

---

## UI

### 40. show_tooltip
Shows a tooltip after grading (score + provider name).

### 41. show_on_card
Shows a score badge on the **back** of the card.

---

## Behavior

### 50. auto_answer
If `true`, pressing **Enter/Space** uses the AI score to choose the default ease button.

If `false`, Anki’s normal default behavior is used.

---

## Networking

### 60. timeout_sec
HTTP timeout in seconds.

### 61. max_output_tokens
Max tokens for the AI response.

Because the model should output **only a number**, this can be small (e.g., 8–32).

---

## Notes

- This add-on grades **only when you typed an answer** (typedAnswer is not empty).
- In `provider: "auto"`, a provider without an API key is skipped automatically.
- If AI fails (network/API error), the add-on falls back to a local similarity score.
