from __future__ import annotations

import os
import re
import json
import html
import urllib.request
import urllib.error
from typing import Optional, Tuple, Dict, Any, List
from difflib import SequenceMatcher

from anki.cards import Card
from aqt import gui_hooks, mw
from aqt.reviewer import Reviewer
from aqt.utils import tooltip
from aqt.qt import (
    Qt,
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QGroupBox,
    QSizePolicy,
)



# =============================================================================
# AI Type Grader (OpenAI + Gemini + Local)
# - Grades typed answers (0-100) via OpenAI Responses API or Gemini generateContent
# - Falls back to local similarity scoring on any error
# - Shows a badge on card back + tooltip
# - Optionally overrides Enter/Space default ease based on the score
#
# Config keys can be numbered (e.g., "01. enabled") to keep ordering stable in
# Anki's Config editor. This add-on accepts BOTH numbered and unnumbered keys.
# =============================================================================


# ===== Global state =====

# Last AI-recommended ease (1-4)
last_ai_ease: Optional[int] = None

# Config cache
_config_cache: Optional[Dict[str, Any]] = None

# Preserve original Reviewer._defaultEase safely (avoid stacking patches on reload)
if hasattr(Reviewer, "_ai_type_grader_original_default_ease"):
    _original_default_ease = getattr(Reviewer, "_ai_type_grader_original_default_ease")
else:
    _original_default_ease = Reviewer._defaultEase
    setattr(Reviewer, "_ai_type_grader_original_default_ease", _original_default_ease)


# =============================================================================
# Config
# =============================================================================

def _invalidate_config_cache(*_args, **_kwargs) -> None:
    global _config_cache
    _config_cache = None


def _cfg_get(cfg: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Return first existing key among keys, else default."""
    for k in keys:
        if k in cfg:
            return cfg.get(k)
    return default


def get_config() -> Dict[str, Any]:
    """
    Load add-on config (cached). Cache is cleared when Anki updates add-on config.
    Accepts both numbered keys and plain keys.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    cfg_raw = mw.addonManager.getConfig(__name__) or {}

    # --- Key aliases (numbered -> plain) ---
    enabled = bool(_cfg_get(cfg_raw, ["01. enabled", "enabled"], True))

    provider = str(_cfg_get(cfg_raw, ["02. provider", "provider"], "") or "").strip().lower()
    if provider not in ("openai", "gemini", "auto", "local", ""):
        provider = "auto"

    auto_order = _cfg_get(cfg_raw, ["03. provider_auto_order", "provider_auto_order"], ["openai", "gemini"])
    if not isinstance(auto_order, list) or not auto_order:
        auto_order = ["openai", "gemini"]
    auto_order = [str(x).strip().lower() for x in auto_order if str(x).strip().lower() in ("openai", "gemini")]
    if not auto_order:
        auto_order = ["openai", "gemini"]

    # OpenAI keys (backward-compat: "api_key")
    openai_key = str(_cfg_get(cfg_raw, ["10. openai_api_key", "openai_api_key", "api_key"], "") or "")
    openai_model = str(_cfg_get(cfg_raw, ["11. openai_model", "openai_model"], "gpt-4o-mini") or "gpt-4o-mini")
    openai_api_url = str(
        _cfg_get(cfg_raw, ["12. openai_api_url", "openai_api_url"], "https://api.openai.com/v1/responses")
        or "https://api.openai.com/v1/responses"
    )

    # Gemini keys
    gemini_key = str(_cfg_get(cfg_raw, ["20. gemini_api_key", "gemini_api_key"], "") or "")
    gemini_model = str(_cfg_get(cfg_raw, ["21. gemini_model", "gemini_model"], "gemini-2.5-flash-lite") or "gemini-2.5-flash-lite")
    gemini_api_url = str(
        _cfg_get(
            cfg_raw,
            ["22. gemini_api_url", "gemini_api_url"],
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        )
        or "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )

    # Answer field
    answer_field = str(_cfg_get(cfg_raw, ["30. answer_field", "answer_field"], "Back") or "Back")

    # Thresholds
    easy_min = int(_cfg_get(cfg_raw, ["31. easy_min", "easy_min"], 80))
    good_min = int(_cfg_get(cfg_raw, ["32. good_min", "good_min"], 60))
    hard_min = int(_cfg_get(cfg_raw, ["33. hard_min", "hard_min"], 40))

    # UI
    show_tooltip = bool(_cfg_get(cfg_raw, ["40. show_tooltip", "show_tooltip"], True))
    show_on_card = bool(_cfg_get(cfg_raw, ["41. show_on_card", "show_on_card"], True))

    # Behavior
    auto_answer = bool(_cfg_get(cfg_raw, ["50. auto_answer", "auto_answer"], True))

    # Networking
    timeout_sec = int(_cfg_get(cfg_raw, ["60. timeout_sec", "timeout_sec"], 20))
    max_output_tokens = int(_cfg_get(cfg_raw, ["61. max_output_tokens", "max_output_tokens"], 16))

    # Default provider if not specified:
    if provider == "":
        provider = "openai" if (openai_key or os.getenv("OPENAI_API_KEY", "")) else "auto"

    _config_cache = {
        "enabled": enabled,

        "provider": provider,
        "provider_auto_order": auto_order,

        # OpenAI
        "openai_api_key": openai_key,
        "openai_model": openai_model,
        "openai_api_url": openai_api_url,

        # Gemini
        "gemini_api_key": gemini_key,
        "gemini_model": gemini_model,
        "gemini_api_url": gemini_api_url,

        # Scoring
        "answer_field": answer_field,
        "easy_min": easy_min,
        "good_min": good_min,
        "hard_min": hard_min,

        # UI
        "show_tooltip": show_tooltip,
        "show_on_card": show_on_card,

        # Behavior
        "auto_answer": auto_answer,

        # Networking
        "timeout_sec": timeout_sec,
        "max_output_tokens": max_output_tokens,
    }
    return _config_cache


# Clear cache when config is updated in Add-ons UI (supported in modern Anki)
try:
    mw.addonManager.setConfigUpdatedAction(__name__, _invalidate_config_cache)  # type: ignore[attr-defined]
except Exception:
    pass


# =============================================================================
# Text normalization helpers
# =============================================================================

_RE_TAG = re.compile(r"<[^>]+>")
_RE_WS = re.compile(r"\s+")


def strip_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = _RE_TAG.sub("", s)
    s = html.unescape(s)
    return s


def normalize_text(s: str) -> str:
    s = strip_html(s)
    s = s.replace("\u00a0", " ")
    s = _RE_WS.sub(" ", s).strip()
    return s


# =============================================================================
# Grading (OpenAI HTTP + Gemini HTTP + Local fallback)
# =============================================================================

def grade_answer_openai_http(
    user: str,
    correct: str,
    api_key: str,
    model: str,
    api_url: str,
    timeout_sec: int,
    max_output_tokens: int,
) -> int:
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OpenAI API key is not set (openai_api_key / api_key / OPENAI_API_KEY).")

    prompt = (
        "Rate how similar the student answer is to the model answer.\n"
        "Return a number 0-100 only.\n\n"
        f"Model: {correct}\n"
        f"Student: {user}\n"
    )

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    body = {"model": model, "input": prompt, "max_output_tokens": max_output_tokens, "temperature": 0}

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            resp_bytes = resp.read()
    except urllib.error.HTTPError as e:
        err_txt = e.read().decode("utf-8", "ignore")
        raise RuntimeError(f"OpenAI HTTP {e.code}: {err_txt}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenAI network error: {e}") from e

    try:
        j = json.loads(resp_bytes.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"OpenAI: failed to parse JSON: {e}") from e

    text: Optional[str] = None
    try:
        out0 = j["output"][0]

        ot = out0.get("output_text")
        if isinstance(ot, dict):
            content = ot.get("content")
            if isinstance(content, list) and content:
                tobj = content[0].get("text")
                if isinstance(tobj, dict) and "value" in tobj:
                    text = tobj["value"]

        if text is None:
            content2 = out0.get("content")
            if isinstance(content2, list) and content2:
                part = content2[0]
                t = part.get("text")
                if isinstance(t, dict) and "value" in t:
                    text = t["value"]
                elif isinstance(t, str):
                    text = t

        if text is None:
            def _walk(obj) -> Optional[str]:
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k == "text" and isinstance(v, str) and v.strip():
                            return v
                        got = _walk(v)
                        if got:
                            return got
                elif isinstance(obj, list):
                    for it in obj:
                        got = _walk(it)
                        if got:
                            return got
                return None

            text = _walk(j)

        if text is None:
            raise KeyError("no text field found")

    except Exception as e:
        raise RuntimeError(f"OpenAI: could not extract text from response: {e}") from e

    text = str(text).strip()
    m = re.search(r"\d+", text)
    if not m:
        raise RuntimeError(f"OpenAI: output did not contain an integer score: {text!r}")

    score = int(m.group())
    return max(0, min(100, score))


def grade_answer_gemini_http(
    user: str,
    correct: str,
    api_key: str,
    model: str,
    api_url_tpl: str,
    timeout_sec: int,
    max_output_tokens: int,
) -> int:
    key = api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError("Gemini API key is not set (gemini_api_key / GEMINI_API_KEY / GOOGLE_API_KEY).")

    prompt = (
        "Rate how similar the student answer is to the model answer.\n"
        "Return a number 0-100 only.\n\n"
        f"Model: {correct}\n"
        f"Student: {user}\n"
    )

    api_url = api_url_tpl.format(model=model)

    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_output_tokens, "temperature": 0},
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            resp_bytes = resp.read()
    except urllib.error.HTTPError as e:
        err_txt = e.read().decode("utf-8", "ignore")
        raise RuntimeError(f"Gemini HTTP {e.code}: {err_txt}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Gemini network error: {e}") from e

    try:
        j = json.loads(resp_bytes.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Gemini: failed to parse JSON: {e}") from e

    text: Optional[str] = None
    try:
        candidates = j.get("candidates")
        if isinstance(candidates, list) and candidates:
            c0 = candidates[0]
            content = c0.get("content", {})
            parts = content.get("parts", [])
            if isinstance(parts, list) and parts:
                p0 = parts[0]
                if isinstance(p0, dict) and isinstance(p0.get("text"), str):
                    text = p0["text"]

        if text is None:
            def _walk(obj) -> Optional[str]:
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k == "text" and isinstance(v, str) and v.strip():
                            return v
                        got = _walk(v)
                        if got:
                            return got
                elif isinstance(obj, list):
                    for it in obj:
                        got = _walk(it)
                        if got:
                            return got
                return None

            text = _walk(j)

        if text is None:
            raise KeyError("no text field found")

    except Exception as e:
        raise RuntimeError(f"Gemini: could not extract text from response: {e}") from e

    text = str(text).strip()
    m = re.search(r"\d+", text)
    if not m:
        raise RuntimeError(f"Gemini: output did not contain an integer score: {text!r}")

    score = int(m.group())
    return max(0, min(100, score))


def grade_answer_local(user: str, correct: str) -> int:
    if not user and not correct:
        return 100
    if not user or not correct:
        return 0
    ratio = SequenceMatcher(None, user.strip(), correct.strip()).ratio()
    return int(ratio * 100)


def score_to_ease(score: int, cfg: Dict[str, Any]) -> int:
    if score >= cfg["easy_min"]:
        return 4
    if score >= cfg["good_min"]:
        return 3
    if score >= cfg["hard_min"]:
        return 2
    return 1


def _provider_candidates(cfg: Dict[str, Any]) -> List[str]:
    provider = cfg.get("provider", "auto")
    if provider == "local":
        return []
    if provider in ("openai", "gemini"):
        return [provider]
    return list(cfg.get("provider_auto_order", ["openai", "gemini"]))


def _background_grade(typed: str, correct: str, cfg: Dict[str, Any]) -> Tuple[int, str, Optional[str]]:
    score: Optional[int] = None
    source = "local"
    error_message: Optional[str] = None

    typed_n = normalize_text(typed)
    correct_n = normalize_text(correct)

    for provider in _provider_candidates(cfg):
        try:
            if provider == "openai":
                if not (cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")):
                    continue
                score = grade_answer_openai_http(
                    typed_n, correct_n,
                    cfg["openai_api_key"],
                    cfg["openai_model"],
                    cfg["openai_api_url"],
                    cfg["timeout_sec"],
                    cfg["max_output_tokens"],
                )
                source = "OpenAI"
                error_message = None
                break

            if provider == "gemini":
                if not (cfg.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")):
                    continue
                score = grade_answer_gemini_http(
                    typed_n, correct_n,
                    cfg["gemini_api_key"],
                    cfg["gemini_model"],
                    cfg["gemini_api_url"],
                    cfg["timeout_sec"],
                    cfg["max_output_tokens"],
                )
                source = "Gemini"
                error_message = None
                break

        except Exception as e:
            error_message = str(e)
            if cfg.get("provider") in ("openai", "gemini"):
                break

    if score is None:
        score = grade_answer_local(typed_n, correct_n)
        source = "local (AI error)" if error_message else "local"

    return score, source, error_message


# =============================================================================
# UI: score badge
# =============================================================================

def _ensure_score_css(reviewer: Reviewer) -> None:
    js = r"""
    (function(){
        if (document.getElementById('ai-score-style')) return;

        var style = document.createElement('style');
        style.id = 'ai-score-style';
        style.innerHTML = `
        #ai-score-wrapper {
            width: 100%;
            margin-top: 12px;
            text-align: center;
        }
        #ai-score {
            padding: 8px 12px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 1.0em;
            display: inline-block;
        }
        .ai-pending {
            color: #555555;
            background-color: #eeeeee;
            border: 1px dashed #999999;
            font-weight: 500;
        }
        .ai-easy {
            color: #0a427a;
            background-color: #d6ebff;
            border: 2px solid #0a427a;
        }
        .ai-good {
            color: #155724;
            background-color: #d4edda;
            border: 2px solid #28a745;
        }
        .ai-hard {
            color: #856404;
            background-color: #fff3cd;
            border: 2px solid #856404;
        }
        .ai-again {
            color: #721c24;
            background-color: #f8d7da;
            border: 2px solid #721c24;
        }
        `;
        document.head.appendChild(style);
    })();
    """
    reviewer.web.eval(js)


def show_pending_on_card(reviewer: Reviewer) -> None:
    _ensure_score_css(reviewer)
    js = r"""
    (function(){
        var wrapper = document.getElementById('ai-score-wrapper');
        if (!wrapper) {
            wrapper = document.createElement('div');
            wrapper.id = 'ai-score-wrapper';
            document.body.appendChild(wrapper);
        }
        var el = document.getElementById('ai-score');
        if (!el) {
            el = document.createElement('div');
            el.id = 'ai-score';
            wrapper.appendChild(el);
        }
        el.className = 'ai-pending';
        el.textContent = '⚙ AI grading…';
        wrapper.style.display = 'block';
    })();
    """
    reviewer.web.eval(js)


def show_score_on_card(reviewer: Reviewer, score: int, cfg: Dict[str, Any]) -> None:
    if score >= cfg["easy_min"]:
        cls = "ai-easy"
    elif score >= cfg["good_min"]:
        cls = "ai-good"
    elif score >= cfg["hard_min"]:
        cls = "ai-hard"
    else:
        cls = "ai-again"

    _ensure_score_css(reviewer)

    js = f"""
    (function(){{
        var wrapper = document.getElementById('ai-score-wrapper');
        if (!wrapper) {{
            wrapper = document.createElement('div');
            wrapper.id = 'ai-score-wrapper';
            document.body.appendChild(wrapper);
        }}
        var el = document.getElementById('ai-score');
        if (!el) {{
            el = document.createElement('div');
            el.id = 'ai-score';
            wrapper.appendChild(el);
        }}
        el.className = '{cls}';
        el.textContent = 'Score: {score}%';
        wrapper.style.display = 'block';
    }})();
    """
    reviewer.web.eval(js)


def hide_score_on_card(reviewer: Reviewer) -> None:
    js = """
    (function(){
        var wrapper = document.getElementById('ai-score-wrapper');
        if (wrapper) wrapper.style.display = 'none';
    })();
    """
    reviewer.web.eval(js)


# =============================================================================
# Hooks
# =============================================================================

def on_reviewer_did_show_answer(card: Card) -> None:
    global last_ai_ease

    cfg = get_config()
    if not cfg["enabled"]:
        last_ai_ease = None
        return

    reviewer = mw.reviewer
    if reviewer is None:
        last_ai_ease = None
        return

    typed = getattr(reviewer, "typedAnswer", None)
    if not typed or not str(typed).strip():
        last_ai_ease = None
        hide_score_on_card(reviewer)
        return

    note = card.note()
    answer_field = cfg["answer_field"]
    if answer_field not in note:
        if cfg["show_tooltip"]:
            tooltip(f"AI grading: field '{answer_field}' not found.")
        last_ai_ease = None
        hide_score_on_card(reviewer)
        return

    correct = note[answer_field]
    card_id = card.id

    if cfg["show_on_card"]:
        show_pending_on_card(reviewer)

    def _do_in_background() -> Tuple[int, str, Optional[str]]:
        return _background_grade(str(typed), str(correct), cfg)

    def _on_done(future) -> None:
        global last_ai_ease

        try:
            score, source, error_message = future.result()
        except Exception as e:
            cfg_local = get_config()
            if cfg_local.get("show_tooltip", True):
                tooltip(f"AI grading error: {e}")
            last_ai_ease = None
            reviewer_now = mw.reviewer
            if reviewer_now is not None:
                hide_score_on_card(reviewer_now)
            return

        reviewer_now = mw.reviewer
        if reviewer_now is None or reviewer_now.card is None:
            return
        if reviewer_now.card.id != card_id:
            return

        cfg_local = get_config()
        if not cfg_local["enabled"]:
            last_ai_ease = None
            hide_score_on_card(reviewer_now)
            return

        if cfg_local["show_tooltip"]:
            if error_message and source.startswith("local"):
                msg = error_message
                if len(msg) > 300:
                    msg = msg[:300] + "…"
                tooltip(f"{source}: {score}% ({msg})")
            else:
                tooltip(f"{source} score: {score}%")

        if cfg_local["show_on_card"]:
            show_score_on_card(reviewer_now, score, cfg_local)

        last_ai_ease = score_to_ease(score, cfg_local)

    mw.taskman.run_in_background(_do_in_background, _on_done)


def on_reviewer_did_show_question(card: Card) -> None:
    global last_ai_ease
    last_ai_ease = None
    reviewer = mw.reviewer
    if reviewer is not None:
        hide_score_on_card(reviewer)


# =============================================================================
# Default ease override (Enter / Space)
# =============================================================================

def _ai_default_ease(self: Reviewer) -> int:
    cfg = get_config()
    if not cfg.get("enabled", True):
        return _original_default_ease(self)

    if not cfg.get("auto_answer", True):
        return _original_default_ease(self)

    global last_ai_ease
    if last_ai_ease is not None:
        return last_ai_ease

    return _original_default_ease(self)


# =============================================================================
# Register hooks + patch
# =============================================================================

gui_hooks.reviewer_did_show_answer.append(on_reviewer_did_show_answer)
gui_hooks.reviewer_did_show_question.append(on_reviewer_did_show_question)

Reviewer._defaultEase = _ai_default_ease


# =============================================================================
# Config Dialog (opened from Tools -> Add-ons -> Config)
# =============================================================================

_DEFAULT_NUMBERED_CONFIG: Dict[str, Any] = {
    "01. enabled": True,
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
    "40. show_tooltip": True,
    "41. show_on_card": True,
    "50. auto_answer": True,
    "60. timeout_sec": 20,
    "61. max_output_tokens": 16,
}


def _to_numbered_config(cfg_plain: Dict[str, Any]) -> Dict[str, Any]:
    """Internal plain keys -> numbered keys (keeps ordering stable)."""
    return {
        "01. enabled": bool(cfg_plain.get("enabled", True)),
        "02. provider": str(cfg_plain.get("provider", "auto")),
        "03. provider_auto_order": list(cfg_plain.get("provider_auto_order", ["openai", "gemini"])),

        "10. openai_api_key": str(cfg_plain.get("openai_api_key", "")),
        "11. openai_model": str(cfg_plain.get("openai_model", "gpt-4o-mini")),
        "12. openai_api_url": str(cfg_plain.get("openai_api_url", "https://api.openai.com/v1/responses")),

        "20. gemini_api_key": str(cfg_plain.get("gemini_api_key", "")),
        "21. gemini_model": str(cfg_plain.get("gemini_model", "gemini-2.5-flash-lite")),
        "22. gemini_api_url": str(cfg_plain.get(
            "gemini_api_url",
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        )),

        "30. answer_field": str(cfg_plain.get("answer_field", "Back")),
        "31. easy_min": int(cfg_plain.get("easy_min", 80)),
        "32. good_min": int(cfg_plain.get("good_min", 60)),
        "33. hard_min": int(cfg_plain.get("hard_min", 40)),

        "40. show_tooltip": bool(cfg_plain.get("show_tooltip", True)),
        "41. show_on_card": bool(cfg_plain.get("show_on_card", True)),

        "50. auto_answer": bool(cfg_plain.get("auto_answer", True)),

        "60. timeout_sec": int(cfg_plain.get("timeout_sec", 20)),
        "61. max_output_tokens": int(cfg_plain.get("max_output_tokens", 16)),
    }


class AiTypeGraderConfigDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI Type Grader - Settings")
        self.setMinimumWidth(560)

        # always read fresh
        _invalidate_config_cache()
        cfg = get_config()

        root = QVBoxLayout(self)

        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        self.setSizeGripEnabled(True)

        info = QLabel(
            "Configure provider, scoring thresholds, and UI.\n"
            "API keys are stored in Anki's add-on config (meta.json)."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        tabs = QTabWidget(self)
        root.addWidget(tabs)

        # ---- Tab: General
        tab_general = QWidget()
        tabs.addTab(tab_general, "General")
        gl = QVBoxLayout(tab_general)

        # ---- General (top form) : QWidgetで包んで「縦に伸びすぎ」を防ぐ
        general_top = QWidget()
        general_top.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        form_g = QFormLayout(general_top)
        form_g.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form_g.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form_g.setHorizontalSpacing(12)
        form_g.setVerticalSpacing(10)
        gl.addWidget(general_top)

        self.enabled_cb = QCheckBox()
        self.enabled_cb.setChecked(bool(cfg.get("enabled", True)))
        form_g.addRow("Enable add-on", self.enabled_cb)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["auto", "openai", "gemini", "local"])
        self.provider_combo.setCurrentText(str(cfg.get("provider", "auto")))
        self.provider_combo.setMinimumWidth(220)
        form_g.addRow("Provider", self.provider_combo)

        # ---- Auto order box
        auto_box = QGroupBox("Auto provider order (when Provider = auto)")
        auto_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        auto_form = QFormLayout(auto_box)
        auto_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        auto_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        auto_form.setHorizontalSpacing(12)
        auto_form.setVerticalSpacing(10)

        self.auto_first = QComboBox()
        self.auto_second = QComboBox()
        for w in (self.auto_first, self.auto_second):
            w.addItems(["openai", "gemini"])
            w.setMinimumWidth(220)

        order = cfg.get("provider_auto_order", ["openai", "gemini"])
        if not isinstance(order, list):
            order = ["openai", "gemini"]
        order = [str(x).strip().lower() for x in order if str(x).strip().lower() in ("openai", "gemini")]
        if len(order) < 2:
            order = (order + ["openai", "gemini"])[:2]
        if order[0] == order[1]:
            order = ["openai", "gemini"]

        self.auto_first.setCurrentText(order[0])
        self.auto_second.setCurrentText(order[1])

        auto_form.addRow("1st", self.auto_first)
        auto_form.addRow("2nd", self.auto_second)
        gl.addWidget(auto_box)
        gl.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ---- Tab: OpenAI
        tab_openai = QWidget()
        tabs.addTab(tab_openai, "OpenAI")
        ol = QVBoxLayout(tab_openai)
        ol.setAlignment(Qt.AlignmentFlag.AlignTop)

        openai_box = QGroupBox("OpenAI settings")
        openai_form = QFormLayout(openai_box)

        openai_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        openai_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        openai_form.setHorizontalSpacing(12)
        openai_form.setVerticalSpacing(10)

        self.openai_key = QLineEdit()
        self.openai_key.setEchoMode(QLineEdit.EchoMode.Password)

        # ★ 追記：cfg反映＋見た目（Geminiと揃える）
        self.openai_key.setText(str(cfg.get("openai_api_key", "")))
        self.openai_key.setMinimumWidth(360)
        self.openai_key.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.openai_key_toggle = QPushButton("Show")
        self.openai_key_toggle.setFixedWidth(80)

        # ★ 追記：Show/Hide を動かす
        self.openai_key_toggle.clicked.connect(self._toggle_openai_key)

        key_w = QWidget()
        key_l = QHBoxLayout(key_w)
        key_l.setContentsMargins(0, 0, 0, 0)
        key_l.setSpacing(8)
        key_l.addWidget(self.openai_key, 1)
        key_l.addWidget(self.openai_key_toggle, 0)

        openai_form.addRow("API key", key_w)

        self.openai_model = QLineEdit(str(cfg.get("openai_model", "gpt-4o-mini")))
        self.openai_model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        openai_form.addRow("Model", self.openai_model)

        self.openai_url = QLineEdit(str(cfg.get("openai_api_url", "https://api.openai.com/v1/responses")))
        self.openai_url.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        openai_form.addRow("API URL", self.openai_url)

        ol.addWidget(openai_box)

        # ---- Tab: Gemini
        tab_gemini = QWidget()
        tabs.addTab(tab_gemini, "Gemini")
        ml = QVBoxLayout(tab_gemini)
        ml.setAlignment(Qt.AlignmentFlag.AlignTop)

        gemini_box = QGroupBox("Gemini settings")
        gemini_form = QFormLayout(gemini_box)

        gemini_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        gemini_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        gemini_form.setHorizontalSpacing(12)
        gemini_form.setVerticalSpacing(10)

        self.gemini_key = QLineEdit()
        self.gemini_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_key.setText(str(cfg.get("gemini_api_key", "")))
        self.gemini_key.setMinimumWidth(360)
        self.gemini_key.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.gemini_key_toggle = QPushButton("Show")
        self.gemini_key_toggle.setFixedWidth(80)
        self.gemini_key_toggle.clicked.connect(self._toggle_gemini_key)

        gkey_w = QWidget()
        gkey_l = QHBoxLayout(gkey_w)
        gkey_l.setContentsMargins(0, 0, 0, 0)
        gkey_l.setSpacing(8)
        gkey_l.addWidget(self.gemini_key, 1)
        gkey_l.addWidget(self.gemini_key_toggle, 0)

        gemini_form.addRow("API key", gkey_w)

        self.gemini_model = QLineEdit(str(cfg.get("gemini_model", "gemini-2.5-flash-lite")))
        self.gemini_model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        gemini_form.addRow("Model", self.gemini_model)

        self.gemini_url = QLineEdit(str(cfg.get(
            "gemini_api_url",
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        )))
        self.gemini_url.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        gemini_form.addRow("API URL", self.gemini_url)

        ml.addWidget(gemini_box)

        # ---- Tab: Scoring & UI
        tab_scoring = QWidget()
        tabs.addTab(tab_scoring, "Scoring & UI")
        sl = QVBoxLayout(tab_scoring)
        sl.setAlignment(Qt.AlignmentFlag.AlignTop)

        scoring_box = QGroupBox("Scoring")
        scoring_form = QFormLayout(scoring_box)

        self.answer_field = QLineEdit(str(cfg.get("answer_field", "Back")))
        scoring_form.addRow("Answer field name", self.answer_field)

        self.easy_min = QSpinBox()
        self.easy_min.setRange(0, 100)
        self.easy_min.setValue(int(cfg.get("easy_min", 80)))
        scoring_form.addRow("Easy min", self.easy_min)

        self.good_min = QSpinBox()
        self.good_min.setRange(0, 100)
        self.good_min.setValue(int(cfg.get("good_min", 60)))
        scoring_form.addRow("Good min", self.good_min)

        self.hard_min = QSpinBox()
        self.hard_min.setRange(0, 100)
        self.hard_min.setValue(int(cfg.get("hard_min", 40)))
        scoring_form.addRow("Hard min", self.hard_min)

        sl.addWidget(scoring_box)

        ui_box = QGroupBox("UI / Behavior")
        ui_form = QFormLayout(ui_box)

        self.show_tooltip = QCheckBox("Show tooltip")
        self.show_tooltip.setChecked(bool(cfg.get("show_tooltip", True)))
        ui_form.addRow(self.show_tooltip)

        self.show_on_card = QCheckBox("Show badge on card back")
        self.show_on_card.setChecked(bool(cfg.get("show_on_card", True)))
        ui_form.addRow(self.show_on_card)

        self.auto_answer = QCheckBox("Override Enter/Space answer button automatically")
        self.auto_answer.setChecked(bool(cfg.get("auto_answer", True)))
        ui_form.addRow(self.auto_answer)

        sl.addWidget(ui_box)

        net_box = QGroupBox("Networking")
        net_form = QFormLayout(net_box)

        self.timeout_sec = QSpinBox()
        self.timeout_sec.setRange(1, 300)
        self.timeout_sec.setValue(int(cfg.get("timeout_sec", 20)))
        net_form.addRow("Timeout (sec)", self.timeout_sec)

        self.max_output_tokens = QSpinBox()
        self.max_output_tokens.setRange(1, 4096)
        self.max_output_tokens.setValue(int(cfg.get("max_output_tokens", 16)))
        net_form.addRow("Max output tokens", self.max_output_tokens)

        sl.addWidget(net_box)

        # ---- Buttons
        btn_row = QHBoxLayout()
        root.addLayout(btn_row)

        self.reset_btn = QPushButton("Reset to defaults")
        self.reset_btn.clicked.connect(self._reset_to_defaults)
        btn_row.addWidget(self.reset_btn)

        btn_row.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.provider_combo.currentTextChanged.connect(self._refresh_enabled_state)
        self._refresh_enabled_state()

    def _toggle_openai_key(self) -> None:
        if self.openai_key.echoMode() == QLineEdit.EchoMode.Password:
            self.openai_key.setEchoMode(QLineEdit.EchoMode.Normal)
            self.openai_key_toggle.setText("Hide")
        else:
            self.openai_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.openai_key_toggle.setText("Show")

    def _toggle_gemini_key(self) -> None:
        if self.gemini_key.echoMode() == QLineEdit.EchoMode.Password:
            self.gemini_key.setEchoMode(QLineEdit.EchoMode.Normal)
            self.gemini_key_toggle.setText("Hide")
        else:
            self.gemini_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.gemini_key_toggle.setText("Show")

    def _refresh_enabled_state(self) -> None:
        provider = self.provider_combo.currentText().strip().lower()
        auto_enabled = (provider == "auto")
        self.auto_first.setEnabled(auto_enabled)
        self.auto_second.setEnabled(auto_enabled)

    def _reset_to_defaults(self) -> None:
        d = dict(_DEFAULT_NUMBERED_CONFIG)

        # General
        self.enabled_cb.setChecked(bool(d["01. enabled"]))
        self.provider_combo.setCurrentText(str(d["02. provider"]))
        order = list(d["03. provider_auto_order"])
        self.auto_first.setCurrentText(order[0])
        self.auto_second.setCurrentText(order[1])

        # OpenAI
        self.openai_key.setText(str(d["10. openai_api_key"]))
        self.openai_model.setText(str(d["11. openai_model"]))
        self.openai_url.setText(str(d["12. openai_api_url"]))

        # Gemini
        self.gemini_key.setText(str(d["20. gemini_api_key"]))
        self.gemini_model.setText(str(d["21. gemini_model"]))
        self.gemini_url.setText(str(d["22. gemini_api_url"]))

        # Scoring/UI/Net
        self.answer_field.setText(str(d["30. answer_field"]))
        self.easy_min.setValue(int(d["31. easy_min"]))
        self.good_min.setValue(int(d["32. good_min"]))
        self.hard_min.setValue(int(d["33. hard_min"]))
        self.show_tooltip.setChecked(bool(d["40. show_tooltip"]))
        self.show_on_card.setChecked(bool(d["41. show_on_card"]))
        self.auto_answer.setChecked(bool(d["50. auto_answer"]))
        self.timeout_sec.setValue(int(d["60. timeout_sec"]))
        self.max_output_tokens.setValue(int(d["61. max_output_tokens"]))

        self._refresh_enabled_state()

    def accept(self) -> None:
        # Build plain cfg (same as get_config() returns)
        provider = self.provider_combo.currentText().strip().lower()

        first = self.auto_first.currentText().strip().lower()
        second = self.auto_second.currentText().strip().lower()
        if first == second:
            # deterministic fix
            second = "gemini" if first == "openai" else "openai"

        plain: Dict[str, Any] = {
            "enabled": self.enabled_cb.isChecked(),
            "provider": provider,
            "provider_auto_order": [first, second],

            "openai_api_key": self.openai_key.text().strip(),
            "openai_model": self.openai_model.text().strip() or "gpt-4o-mini",
            "openai_api_url": self.openai_url.text().strip() or "https://api.openai.com/v1/responses",

            "gemini_api_key": self.gemini_key.text().strip(),
            "gemini_model": self.gemini_model.text().strip() or "gemini-2.5-flash-lite",
            "gemini_api_url": self.gemini_url.text().strip()
                or "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",

            "answer_field": self.answer_field.text().strip() or "Back",
            "easy_min": int(self.easy_min.value()),
            "good_min": int(self.good_min.value()),
            "hard_min": int(self.hard_min.value()),

            "show_tooltip": self.show_tooltip.isChecked(),
            "show_on_card": self.show_on_card.isChecked(),
            "auto_answer": self.auto_answer.isChecked(),

            "timeout_sec": int(self.timeout_sec.value()),
            "max_output_tokens": int(self.max_output_tokens.value()),
        }

        # Basic sanity (keep thresholds ordered if user messed them up)
        # easy >= good >= hard
        e = plain["easy_min"]
        g = plain["good_min"]
        h = plain["hard_min"]
        e = max(0, min(100, e))
        g = max(0, min(100, g))
        h = max(0, min(100, h))
        if e < g:
            e = g
        if g < h:
            g = h
        plain["easy_min"], plain["good_min"], plain["hard_min"] = e, g, h

        numbered = _to_numbered_config(plain)
        mw.addonManager.writeConfig(__name__, numbered)
        _invalidate_config_cache()

        super().accept()


def _open_config_dialog(*args, **kwargs) -> None:
    # Anki may pass a parent dialog/window; fall back to mw
    parent = None
    if args:
        parent = args[0]
    if parent is None:
        parent = mw
    dlg = AiTypeGraderConfigDialog(parent)
    dlg.exec()


# Register: make Add-ons -> Config open our dialog (no Tools menu button needed)
try:
    mw.addonManager.setConfigAction(__name__, _open_config_dialog)  # type: ignore[attr-defined]
except Exception:
    pass
