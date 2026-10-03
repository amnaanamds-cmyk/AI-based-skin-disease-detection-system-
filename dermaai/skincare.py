"""Personalised skincare routine and product recommendations.

Rules follow mainstream dermatology guidance (daily SPF, a gentle base
routine, at most one or two actives introduced slowly, pregnancy-safe
alternatives). Products are generic archetypes described by their key
ingredients, not brands, so recommendations stay unbiased.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SKIN_TYPES = ["oily", "dry", "combination", "normal", "sensitive"]
CONCERN_LIST = ["acne", "redness", "pigmentation", "aging", "texture", "dryness", "dullness", "oiliness"]
BUDGETS = ["low", "mid", "high"]

# Photo concern -> routine concern, with the score at which it is adopted.
PHOTO_TO_CONCERN = {"blemishes": "acne", "redness": "redness", "pigmentation": "pigmentation",
                    "texture": "texture", "shine": "oiliness"}

ACTIVES = {
    "niacinamide": {"targets": {"oiliness", "acne", "redness", "pigmentation", "texture"}, "when": "AM",
                    "how": "Niacinamide 4-5% serum after cleansing.", "pregnancy_safe": True, "gentle": True},
    "vitamin_c": {"targets": {"pigmentation", "dullness", "aging"}, "when": "AM",
                  "how": "Vitamin C (L-ascorbic acid 10-15% or a gentler derivative) before sunscreen.",
                  "pregnancy_safe": True, "gentle": False},
    "retinoid": {"targets": {"aging", "acne", "texture", "pigmentation"}, "when": "PM",
                 "how": "Retinol 0.25-0.5% (or adapalene 0.1% for acne) 2-3 nights a week, building up to nightly.",
                 "pregnancy_safe": False, "gentle": False},
    "salicylic_acid": {"targets": {"acne", "oiliness", "texture"}, "when": "PM",
                       "how": "Salicylic acid (BHA) 0.5-2% 2-3 nights a week, not on retinoid nights.",
                       "pregnancy_safe": True, "gentle": False},
    "azelaic_acid": {"targets": {"redness", "pigmentation", "acne"}, "when": "PM",
                     "how": "Azelaic acid 10% nightly; well tolerated and pregnancy-compatible.",
                     "pregnancy_safe": True, "gentle": True},
    "aha": {"targets": {"texture", "dullness", "pigmentation"}, "when": "PM",
            "how": "Lactic or glycolic acid (AHA) 1-2 nights a week, not on retinoid nights.",
            "pregnancy_safe": True, "gentle": False},
    "ceramides": {"targets": {"dryness", "redness"}, "when": "AM/PM",
                  "how": "Ceramide + glycerin moisturiser to repair the skin barrier.", "pregnancy_safe": True, "gentle": True},
}

PRODUCTS = [
    # cleansers
    {"id": "cl-gel", "step": "cleanser", "name": "Gentle foaming gel cleanser", "ingredients": ["glycerin", "mild surfactants"],
     "skin": {"oily", "combination", "normal"}, "targets": {"oiliness", "acne"}, "price": "low", "fragrance_free": True, "pregnancy_safe": True},
    {"id": "cl-cream", "step": "cleanser", "name": "Hydrating cream cleanser", "ingredients": ["ceramides", "glycerin"],
     "skin": {"dry", "sensitive", "normal"}, "targets": {"dryness", "redness"}, "price": "low", "fragrance_free": True, "pregnancy_safe": True},
    {"id": "cl-bha", "step": "cleanser", "name": "Salicylic acid 2% cleanser", "ingredients": ["salicylic acid"],
     "skin": {"oily", "combination"}, "targets": {"acne", "oiliness", "texture"}, "price": "mid", "fragrance_free": True, "pregnancy_safe": True},
    {"id": "cl-micellar", "step": "cleanser", "name": "Micellar water (make-up / sunscreen removal)", "ingredients": ["micelles", "glycerin"],
     "skin": set(SKIN_TYPES), "targets": set(), "price": "low", "fragrance_free": True, "pregnancy_safe": True},
    # serums / treatments
    {"id": "se-nia", "step": "am_treatment", "name": "Niacinamide 5% + zinc serum", "ingredients": ["niacinamide", "zinc PCA"],
     "skin": {"oily", "combination", "normal", "sensitive"}, "targets": {"oiliness", "acne", "redness", "pigmentation", "texture"},
     "price": "low", "fragrance_free": True, "pregnancy_safe": True, "active": "niacinamide"},
    {"id": "se-vitc", "step": "am_treatment", "name": "Vitamin C 15% + vitamin E + ferulic serum", "ingredients": ["L-ascorbic acid", "tocopherol", "ferulic acid"],
     "skin": {"normal", "combination", "oily", "dry"}, "targets": {"pigmentation", "dullness", "aging"},
     "price": "high", "fragrance_free": True, "pregnancy_safe": True, "active": "vitamin_c"},
    {"id": "se-vitc-gentle", "step": "am_treatment", "name": "Gentle vitamin C derivative serum", "ingredients": ["ascorbyl glucoside", "niacinamide"],
     "skin": set(SKIN_TYPES), "targets": {"pigmentation", "dullness"}, "price": "mid", "fragrance_free": True, "pregnancy_safe": True, "active": "vitamin_c"},
    {"id": "tr-retinol", "step": "pm_treatment", "name": "Encapsulated retinol 0.3% serum", "ingredients": ["retinol", "squalane"],
     "skin": {"normal", "combination", "oily", "dry"}, "targets": {"aging", "texture", "pigmentation", "acne"},
     "price": "mid", "fragrance_free": True, "pregnancy_safe": False, "active": "retinoid"},
    {"id": "tr-adapalene", "step": "pm_treatment", "name": "Adapalene 0.1% gel", "ingredients": ["adapalene"],
     "skin": {"oily", "combination", "normal"}, "targets": {"acne", "texture"},
     "price": "low", "fragrance_free": True, "pregnancy_safe": False, "active": "retinoid"},
    {"id": "tr-azelaic", "step": "pm_treatment", "name": "Azelaic acid 10% cream", "ingredients": ["azelaic acid"],
     "skin": set(SKIN_TYPES), "targets": {"redness", "pigmentation", "acne"},
     "price": "mid", "fragrance_free": True, "pregnancy_safe": True, "active": "azelaic_acid"},
    {"id": "tr-bha", "step": "pm_treatment", "name": "Salicylic acid 2% leave-on liquid", "ingredients": ["salicylic acid"],
     "skin": {"oily", "combination"}, "targets": {"acne", "oiliness", "texture"},
     "price": "mid", "fragrance_free": True, "pregnancy_safe": True, "active": "salicylic_acid"},
    {"id": "tr-lactic", "step": "pm_treatment", "name": "Lactic acid 5% exfoliating serum", "ingredients": ["lactic acid", "hyaluronic acid"],
     "skin": {"normal", "dry", "combination"}, "targets": {"texture", "dullness", "pigmentation"},
     "price": "low", "fragrance_free": True, "pregnancy_safe": True, "active": "aha"},
    # moisturisers
    {"id": "mo-gel", "step": "moisturizer", "name": "Oil-free gel moisturiser", "ingredients": ["hyaluronic acid", "glycerin"],
     "skin": {"oily", "combination"}, "targets": {"oiliness", "acne"}, "price": "low", "fragrance_free": True, "pregnancy_safe": True},
    {"id": "mo-ceramide", "step": "moisturizer", "name": "Ceramide barrier-repair cream", "ingredients": ["ceramides", "cholesterol", "niacinamide"],
     "skin": {"dry", "sensitive", "normal", "combination"}, "targets": {"dryness", "redness"}, "price": "low", "fragrance_free": True, "pregnancy_safe": True, "active": "ceramides"},
    {"id": "mo-rich", "step": "moisturizer", "name": "Rich peptide night cream", "ingredients": ["peptides", "shea butter", "squalane"],
     "skin": {"dry", "normal"}, "targets": {"aging", "dryness"}, "price": "high", "fragrance_free": False, "pregnancy_safe": True},
    {"id": "mo-cica", "step": "moisturizer", "name": "Soothing centella (cica) balm", "ingredients": ["centella asiatica", "panthenol"],
     "skin": {"sensitive", "dry", "normal"}, "targets": {"redness", "dryness"}, "price": "mid", "fragrance_free": True, "pregnancy_safe": True},
    # sunscreens
    {"id": "spf-fluid", "step": "sunscreen", "name": "Broad-spectrum SPF 50 fluid (matte)", "ingredients": ["UVA/UVB filters"],
     "skin": {"oily", "combination", "normal"}, "targets": {"pigmentation", "aging", "oiliness"}, "price": "mid", "fragrance_free": True, "pregnancy_safe": True},
    {"id": "spf-mineral", "step": "sunscreen", "name": "Mineral SPF 50 (zinc oxide), tinted", "ingredients": ["zinc oxide", "iron oxides"],
     "skin": {"sensitive", "dry", "normal"}, "targets": {"pigmentation", "redness"}, "price": "mid", "fragrance_free": True, "pregnancy_safe": True},
    {"id": "spf-cream", "step": "sunscreen", "name": "Moisturising SPF 30+ cream", "ingredients": ["UVA/UVB filters", "glycerin"],
     "skin": {"dry", "normal"}, "targets": {"dryness", "aging"}, "price": "low", "fragrance_free": True, "pregnancy_safe": True},
]


@dataclass
class Profile:
    skin_type: str = "normal"
    concerns: list[str] = field(default_factory=list)
    age: int | None = None
    sensitive: bool = False
    pregnant: bool = False
    sun_exposure: str = "moderate"  # low | moderate | high
    budget: str = "mid"


def merge_photo_concerns(profile: Profile, photo_scores: dict | None, threshold: int = 40) -> dict[str, str]:
    """Combine self-reported concerns with those measured on the photo. Returns concern -> source."""
    sources = {c: "you" for c in profile.concerns if c in CONCERN_LIST}
    for code, score in (photo_scores or {}).items():
        c = PHOTO_TO_CONCERN.get(code)
        if c and score >= threshold:
            sources[c] = "you + photo" if c in sources else "photo"
    if profile.age and profile.age >= 30:
        sources.setdefault("aging", "age")
    return sources


def _pick_actives(concerns: set[str], p: Profile) -> tuple[str | None, str | None]:
    """Choose at most one morning and one evening active, ranked by concern coverage."""
    sensitive = p.sensitive or p.skin_type == "sensitive"

    def rank(when: str) -> str | None:
        best, best_score = None, 0.0
        for name, a in ACTIVES.items():
            if name == "ceramides" or when not in a["when"]:
                continue
            if p.pregnant and not a["pregnancy_safe"]:
                continue
            score = len(a["targets"] & concerns) + (0.5 if a["gentle"] else 0)
            if sensitive and not a["gentle"]:
                score -= 1.5
            if score > best_score:
                best, best_score = name, score
        return best if best_score >= 1 else None

    return rank("AM"), rank("PM")


def _products_for(step: str, concerns: set[str], p: Profile, active: str | None = None, k: int = 2) -> list[dict]:
    sensitive = p.sensitive or p.skin_type == "sensitive"
    scored = []
    for prod in PRODUCTS:
        if prod["step"] != step:
            continue
        if active and prod.get("active") != active:
            continue
        if p.pregnant and not prod["pregnancy_safe"]:
            continue
        reasons, score = [], 0.0
        if p.skin_type in prod["skin"]:
            score += 2
            reasons.append(f"suits {p.skin_type} skin")
        matched = sorted(prod["targets"] & concerns)
        if matched:
            score += 1.5 * len(matched)
            reasons.append("targets " + ", ".join(matched))
        if prod["price"] == p.budget:
            score += 1
            reasons.append("fits your budget")
        elif BUDGETS.index(prod["price"]) > BUDGETS.index(p.budget):
            score -= 1.5
        if sensitive:
            score += 1 if prod["fragrance_free"] else -3
            if prod["fragrance_free"]:
                reasons.append("fragrance-free")
        scored.append((score, prod, reasons))
    scored.sort(key=lambda t: -t[0])
    return [{"id": pr["id"], "name": pr["name"], "key_ingredients": pr["ingredients"], "price_tier": pr["price"],
             "fragrance_free": pr["fragrance_free"], "match": round(max(0.0, min(1.0, sc / 8)), 2), "why": rs}
            for sc, pr, rs in scored[:k]]


def build_routine(p: Profile, photo_scores: dict | None = None) -> dict:
    if p.skin_type not in SKIN_TYPES:
        p.skin_type = "normal"
    if p.budget not in BUDGETS:
        p.budget = "mid"
    sources = merge_photo_concerns(p, photo_scores)
    if p.skin_type == "oily":
        sources.setdefault("oiliness", "skin type")
    if p.skin_type == "dry":
        sources.setdefault("dryness", "skin type")
    concerns = set(sources)
    am_active, pm_active = _pick_actives(concerns, p)
    spf = "SPF 50" if p.sun_exposure == "high" or "pigmentation" in concerns else "SPF 30+"

    def step(slot: str, title: str, how: str, active: str | None = None) -> dict:
        return {"step": slot, "title": title, "how": how, "products": _products_for(slot, concerns, p, active)}

    am = [step("cleanser", "Cleanse", "Lukewarm water, 30-60 seconds, pat dry." if p.skin_type != "dry"
               else "Rinse with water only, or a cream cleanser if needed.")]
    if am_active:
        am.append(step("am_treatment", "Treat", ACTIVES[am_active]["how"], am_active))
    am.append(step("moisturizer", "Moisturise", "A pea-to-almond-sized amount on face and neck."))
    am.append(step("sunscreen", "Protect", f"Broad-spectrum {spf}, two finger-lengths for face and neck. "
                   "Reapply every 2 hours outdoors. This is the most effective anti-ageing and anti-pigmentation step."))

    pm = [step("cleanser", "Cleanse", "Double-cleanse on days you wore sunscreen or make-up.")]
    if pm_active:
        pm.append(step("pm_treatment", "Treat", ACTIVES[pm_active]["how"], pm_active))
    pm.append(step("moisturizer", "Moisturise", "Apply on slightly damp skin to lock in hydration."))

    tips = ["Introduce one new product at a time, 2 weeks apart. Patch-test on the jaw for 3 days first.",
            "Give actives 8-12 weeks before judging results. Track progress with photos in the same light."]
    if pm_active in {"retinoid", "aha", "salicylic_acid"}:
        tips.append("Do not combine retinoids, AHAs and BHAs on the same night. Pause actives if skin stings or peels.")
    if p.pregnant:
        tips.append("Pregnancy: retinoids are excluded. Check any new product with your doctor or midwife.")
    if "acne" in concerns:
        tips.append("If acne is painful, cystic or scarring, see a dermatologist. Prescription treatment works far better.")
    if "redness" in concerns:
        tips.append("Persistent facial redness with flushing may be rosacea. A dermatologist can confirm and treat it.")

    return {
        "profile": {"skin_type": p.skin_type, "sensitive": p.sensitive, "pregnant": p.pregnant,
                    "sun_exposure": p.sun_exposure, "budget": p.budget},
        "concerns": [{"code": c, "source": s} for c, s in sources.items()],
        "actives": {"am": am_active, "pm": pm_active},
        "am": am,
        "pm": pm,
        "weekly": [{"title": "Skin check", "how": "Once a month, check moles with the ABCDE rule and re-photograph any you are watching."}],
        "tips": tips,
        "disclaimer": "General skincare guidance, not medical advice. Products are generic types; "
                      "choose any brand with the listed key ingredients.",
    }
