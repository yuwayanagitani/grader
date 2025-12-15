# Anki AI Grader

AnkiWeb: https://ankiweb.net/shared/by-author/2117859718

An add-on that scores typed-answer cards (e.g., “Type in the Answer”) using AI or a local fallback, then shows the result as a badge and/or tooltip.

## Features
- AI or local string-similarity grading of typed answers
- Shows score/badge inline during review
- Configurable scoring thresholds and feedback templates
- Option to write score into a custom field

## Requirements
- Optional AI provider API key for advanced grading
- Local fallback uses fuzzy matching (no external calls)

## Installation
1. Tools → Add-ons → Open Add-ons Folder.
2. Put the add-on directory under `addons21/`.
3. Restart Anki.

## Usage
- Configure fields and model in Tools → AI Grader → Settings.
- Enable grading for selected note types or fields.

## Configuration
`config.json`:
- provider (ai/local)
- feedback_template
- score_field (optional)

Example:
```json
{
  "provider": "local",
  "score_field": "GraderScore",
  "thresholds": { "pass": 0.8 }
}
```

## Issues & Support
When reporting problems include the note type, sample question/answers, and Anki version.

## License
See LICENSE.
