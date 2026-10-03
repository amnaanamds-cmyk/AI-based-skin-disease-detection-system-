"""Virtual dermatology assistant.

Every message first passes a deterministic red-flag check, so urgent
symptoms are escalated even if the language model is unavailable. Answers
come from a curated knowledge base (offline, always available). When Claude
API credentials are configured, Claude writes the reply, grounded in the
retrieved knowledge and the user's latest analysis.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

CLAUDE_MODEL = os.getenv("DERMAAI_ASSISTANT_MODEL", "claude-opus-5-5")

RED_FLAGS = [
    ("emergency", r"(can'?t|cannot|difficult(y)?|trouble|hard to) breath|throat (is )?(swelling|closing)|swollen (lips|tongue|face)|anaphyla",
     "Swelling of the face, lips or throat or difficulty breathing can be a severe allergic reaction. Call emergency services now."),
    ("emergency", r"(rash|blister|peeling).*(fever|eyes|mouth)|(fever|eyes|mouth).*(rash|blister|peeling)|skin (is )?(peeling off|sloughing)",
     "A spreading rash with fever, blisters or sores in the eyes or mouth needs emergency care today."),
    ("urgent", r"(spreading|hot|red).*(painful|swollen).*(skin|leg|arm)|cellulitis|red streak",
     "Hot, painful, spreading redness can be a skin infection (cellulitis). See a doctor today."),
    ("urgent", r"mole.*(bleed|grow|chang|bigger|itch|crust|dark)|(bleed|grow|chang|bigger|itch|crust|dark).*mole|new (mole|spot).*(after|over) (30|40|50)",
     "A mole that is changing, growing, itching or bleeding should be checked by a dermatologist within 2 weeks."),
    ("urgent", r"(sore|wound|spot|lesion).*(not|won'?t|doesn'?t|never) heal",
     "A sore that does not heal within 3-4 weeks should be examined by a doctor; it can be a skin cancer."),
]

KNOWLEDGE = [
    ("ABCDE rule for moles", "abcde mole check melanoma signs asymmetry border colour diameter evolving",
     "Check moles with ABCDE: Asymmetry (halves don't match), Border (irregular, notched or blurred), Colour (several shades, "
     "black, blue, red or white), Diameter (larger than 6 mm), Evolving (changing in size, shape, colour, or new itching or "
     "bleeding). Any one of these is a reason to see a dermatologist. Evolution matters most, so photograph moles monthly."),
    ("Melanoma", "melanoma skin cancer dangerous deadly survival early detection",
     "Melanoma is the most dangerous skin cancer, but 5-year survival is above 99% when it is caught early. Warning signs "
     "are a new or changing mole, or an 'ugly duckling' that looks different from your other moles. Only a dermatologist "
     "with dermoscopy or a biopsy can diagnose it."),
    ("Basal cell carcinoma", "basal cell carcinoma bcc pearly shiny bump sore bleeding face nose",
     "Basal cell carcinoma usually looks like a pearly or shiny bump, a pink patch, or a sore that bleeds, crusts and does "
     "not heal, often on the face. It grows slowly and rarely spreads, but should be treated early."),
    ("Actinic keratosis", "actinic keratosis rough scaly patch sun damage precancer",
     "Actinic keratoses are rough, scaly, sandpaper-like patches on sun-exposed skin. They are pre-cancerous and easily "
     "treated by a dermatologist with freezing, creams or light therapy."),
    ("Sunscreen", "sunscreen spf sun protection uv how much reapply daily",
     "Use a broad-spectrum SPF 30 or higher every day, SPF 50 for long outdoor exposure. Use two finger-lengths for face "
     "and neck, apply 15 minutes before going out, and reapply every 2 hours and after swimming or sweating. Add shade, a "
     "hat and sunglasses between 10:00 and 16:00."),
    ("Acne routine", "acne pimples breakouts spots blackheads whiteheads routine treatment",
     "For mild acne: a gentle cleanser twice daily, a non-comedogenic moisturiser, daily sunscreen, and one active such as "
     "adapalene 0.1%, benzoyl peroxide 2.5-5% or salicylic acid. Give it 8-12 weeks. Don't pick or squeeze. See a "
     "dermatologist for painful nodules, scarring or acne that doesn't improve; prescription options work well."),
    ("Retinoids", "retinol retinoid tretinoin adapalene how to use purging irritation",
     "Start retinoids 2-3 nights a week on dry skin with a pea-sized amount, then build up. Expect some dryness or 'purging' "
     "for 4-6 weeks. Always use sunscreen, and avoid same-night AHA/BHA. Do not use retinoids in pregnancy."),
    ("Pregnancy-safe skincare", "pregnant pregnancy breastfeeding safe skincare avoid",
     "In pregnancy, avoid retinoids (retinol, tretinoin, adapalene) and oral isotretinoin. Generally considered safe: "
     "azelaic acid, niacinamide, vitamin C, mineral sunscreen, and low-strength glycolic or lactic acid. Check with your "
     "doctor or midwife."),
    ("Hyperpigmentation", "dark spots pigmentation melasma uneven tone post inflammatory marks",
     "Dark spots fade with strict daily sunscreen plus actives such as vitamin C, niacinamide, azelaic acid or a retinoid, "
     "usually over 8-12 weeks. Melasma (symmetrical patches, often hormonal) needs tinted mineral sunscreen against "
     "visible light and often prescription treatment."),
    ("Rosacea and redness", "rosacea redness flushing red face sensitive burning",
     "Persistent central facial redness with flushing, visible vessels or bumps may be rosacea. Avoid triggers (heat, "
     "alcohol, spicy food, harsh products), use gentle fragrance-free products and mineral sunscreen. Azelaic acid helps, "
     "and a dermatologist can prescribe more."),
    ("Eczema", "eczema atopic dermatitis itchy dry patches flare",
     "Eczema causes dry, itchy, inflamed patches. Moisturise generously at least twice daily with a fragrance-free cream "
     "or ointment, take short lukewarm showers, and avoid soap and irritants. See a doctor if it is widespread, weeping or "
     "infected, or affects sleep."),
    ("Psoriasis", "psoriasis thick scaly plaques silver scales elbows knees scalp",
     "Psoriasis causes well-defined, thick, red plaques with silvery scale, often on elbows, knees and scalp. It is a "
     "chronic immune condition; many effective treatments exist, so see a doctor for diagnosis."),
    ("Dry skin", "dry skin flaky tight dehydrated moisturizer winter",
     "For dry skin: a cream cleanser, ceramide or urea moisturiser on damp skin, short lukewarm showers, and a humidifier "
     "in winter. Avoid foaming cleansers and alcohol-heavy toners."),
    ("Oily skin", "oily skin shine greasy pores sebum",
     "For oily skin: a gentle gel cleanser twice daily, an oil-free moisturiser (skipping it can make oiliness worse), "
     "niacinamide, and salicylic acid a few times a week. A matte sunscreen helps control shine."),
    ("Fungal skin infections", "fungal ringworm tinea athlete foot itchy ring rash",
     "Ring-shaped, itchy, scaly rashes with a clearer centre are often fungal (tinea). Over-the-counter antifungal creams "
     "(clotrimazole, terbinafine) for 2-4 weeks usually work. See a doctor if it spreads or affects scalp or nails."),
    ("Sunburn", "sunburn burnt red painful peeling after sun",
     "Cool the skin, use aloe or a fragrance-free moisturiser, drink water and take an over-the-counter painkiller if "
     "needed. Seek care for blistering over a large area, fever, chills or confusion. Every sunburn raises skin-cancer risk."),
    ("Taking good photos", "photo picture camera how to take image quality app",
     "Use daylight without flash, hold the camera parallel to the skin, and fill most of the frame with the spot and a "
     "little surrounding skin. Tap to focus. For tracking, keep the same distance and lighting and add a coin for scale."),
    ("How DermaAI works", "how does app work accuracy ai model trust reliable diagnosis",
     "DermaAI checks image quality, estimates probabilities for 7 lesion types with a deep-learning model, shows where it "
     "looked, and runs an ABCDE analysis. It is a decision-support tool, not a diagnosis: it can be wrong, and a "
     "dermatologist should examine any lesion that worries you."),
    ("Patch testing", "patch test new product reaction allergy irritation",
     "Before using a new product on your face, apply a small amount behind the ear or on the jaw for 3 days. Stop if you "
     "get redness, itching or burning. Introduce one new product at a time, about 2 weeks apart."),
]

SYSTEM_PROMPT = """You are DermaAI's virtual dermatology assistant inside a skin-health app.

Your job is to give clear, practical, evidence-based skin and skincare information and to help people decide how urgently to see a doctor.

Ground your answer in the reference notes and the user's latest app results when they are relevant. If the notes don't cover the question, use mainstream dermatology knowledge and say when something is uncertain.

You cannot examine the skin or make a diagnosis. Say so when someone asks what a lesion "is", and point them to the app's analysis and to a dermatologist. Never tell someone that a spot is definitely harmless. When anything sounds worrying (a changing or bleeding mole, a sore that doesn't heal, a rapidly spreading rash, signs of infection or allergic reaction), say plainly how soon they should see a doctor.

Do not prescribe prescription-only medicines or doses. You may describe over-the-counter options and what a doctor might offer.

Keep answers short: 2-5 sentences or a brief list, in plain language at the reading level of a general audience. Answer in the language the user writes in."""


def check_red_flags(text: str) -> dict | None:
    t = text.lower()
    hits = [(lvl, msg) for lvl, pat, msg in RED_FLAGS if re.search(pat, t)]
    if not hits:
        return None
    level = "emergency" if any(l == "emergency" for l, _ in hits) else "urgent"
    return {"level": level, "messages": [m for _, m in hits]}


@lru_cache(maxsize=1)
def _index():
    # Topic words (title + keywords) and answer text are scored separately so that a
    # passing mention in another answer doesn't outrank the note that is about the topic.
    fields = []
    for docs in ([f"{t} {kw}" for t, kw, _ in KNOWLEDGE], [a for _, _, a in KNOWLEDGE]):
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        fields.append((vec, vec.fit_transform(docs)))
    return fields


def retrieve(query: str, k: int = 3, min_score: float = 0.08) -> list[dict]:
    (tv, tm), (av, am) = _index()
    sims = 0.75 * cosine_similarity(tv.transform([query]), tm)[0] + 0.25 * cosine_similarity(av.transform([query]), am)[0]
    order = sims.argsort()[::-1][:k]
    return [{"title": KNOWLEDGE[i][0], "text": KNOWLEDGE[i][2], "score": round(float(sims[i]), 3)}
            for i in order if sims[i] >= min_score]


def _llm_enabled() -> bool:
    flag = os.getenv("DERMAAI_ASSISTANT_LLM", "auto").lower()
    if flag in {"0", "false", "off"}:
        return False
    if flag in {"1", "true", "on"}:
        return True
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


@lru_cache(maxsize=1)
def _client():
    import anthropic

    return anthropic.Anthropic(max_retries=1, timeout=60.0)


def _context_block(sources: list[dict], context: dict | None) -> str:
    parts = ["<reference_notes>"]
    parts += [f"## {s['title']}\n{s['text']}" for s in sources] or ["(no matching notes)"]
    parts.append("</reference_notes>")
    if context:
        parts.append("<latest_app_results>")
        parts += [f"{k}: {v}" for k, v in context.items() if v not in (None, "", [])]
        parts.append("</latest_app_results>")
    return "\n".join(parts)


def _ask_claude(message: str, history: list[dict], sources: list[dict], context: dict | None) -> str | None:
    import anthropic

    turns = [{"role": h["role"], "content": str(h["content"])[:4000]}
             for h in history[-10:] if h.get("role") in {"user", "assistant"} and h.get("content")]
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    turns.append({"role": "user", "content": f"{_context_block(sources, context)}\n\n{message}"})
    try:
        response = _client().beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=turns,
            output_config={"effort": "low"},
            # On a safety decline, the API re-runs the request on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.APIConnectionError as e:
        log.warning("assistant: Claude API unreachable: %s", e)
        return None
    except anthropic.APIStatusError as e:
        log.warning("assistant: Claude API error %s: %s", e.status_code, e.message)
        return None
    if response.stop_reason == "refusal":
        return None
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text or None


RESULT_WORDS = re.compile(r"\b(my|last|latest)\b.*\b(result|analysis|score|report|routine|check|scan)\b", re.I)
TRIAGE_ADVICE = {
    "high": "This means you should see a dermatologist urgently, within days.",
    "moderate": "This means a dermatologist should look at it within the next few weeks.",
    "low": "This means low concern. Keep photographing it monthly and watch for change.",
    "retake": "The photo quality was too poor. Please retake it in daylight, in focus.",
}


def explain_results(context: dict) -> str | None:
    """Plain-language summary of the user's latest app results (offline mode)."""
    parts = []
    if context.get("lesion_triage"):
        parts.append(f"Your last lesion check was rated {context['lesion_triage']} priority; the most likely match was "
                     f"{context.get('lesion_top_prediction', 'unknown')} (combined malignancy probability "
                     f"{context.get('lesion_malignancy_probability', 'n/a')}). "
                     + TRIAGE_ADVICE.get(str(context["lesion_triage"]), ""))
        if str(context.get("model_demo_mode", "")).startswith("yes"):
            parts.append("Note: the app is in demo mode, so that result is not meaningful.")
    if context.get("lesion_change"):
        parts.append(f"Your before/after comparison showed: {context['lesion_change']}")
    if context.get("skin_health_score") is not None:
        concerns = context.get("skin_concerns") or "no marked concerns"
        parts.append(f"Your skin health score is {context['skin_health_score']}/100 ({concerns}).")
    if context.get("routine_actives"):
        parts.append(f"Your routine's active ingredients are {context['routine_actives']}. Introduce them one at a time "
                     "and give them 8-12 weeks.")
    return " ".join(parts) or None


def chat(message: str, history: list[dict] | None = None, context: dict | None = None) -> dict:
    message = message.strip()[:2000]
    flags = check_red_flags(message)
    sources = retrieve(message)
    reply, engine = None, "knowledge-base"
    if _llm_enabled():
        reply = _ask_claude(message, history or [], sources, context)
        if reply:
            engine = "claude"
    if not reply and context and RESULT_WORDS.search(message):
        reply = explain_results(context)
    if not reply and flags and not sources:
        reply = "Please get help now rather than relying on this app."
    if not reply:
        if sources:
            reply = sources[0]["text"]
            if len(sources) > 1 and sources[1]["score"] >= 0.6 * sources[0]["score"]:
                reply += f"\n\nRelated: {sources[1]['title']}. {sources[1]['text']}"
        else:
            reply = ("I can help with moles and skin cancer warning signs, acne, pigmentation, redness, eczema, "
                     "sun protection and building a skincare routine. Could you rephrase or add more detail? "
                     "For anything that worries you about a specific spot, please see a dermatologist.")
    if flags:
        reply = "⚠️ " + " ".join(flags["messages"]) + "\n\n" + reply
    return {
        "reply": reply,
        "escalation": flags,
        "sources": [{"title": s["title"], "score": s["score"]} for s in sources],
        "engine": engine,
        "disclaimer": "Information only, not a diagnosis. In an emergency call your local emergency number.",
    }
