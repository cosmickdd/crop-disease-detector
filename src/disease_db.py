"""
Crop Disease Knowledge Database
================================
Maps every PlantVillage class name to structured agronomic information:
  - crop name, disease name (human-readable)
  - pathogen type
  - visible symptoms
  - recommended treatment
  - prevention strategies
  - severity rating

Used by the inference pipeline to enrich raw model predictions with
actionable, farmer-facing advice.
"""

from __future__ import annotations
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Master database
# ---------------------------------------------------------------------------
DISEASE_DATABASE: Dict[str, Dict[str, Any]] = {

    # ═══════════════════════════════════════════════════════════════════════
    # APPLE
    # ═══════════════════════════════════════════════════════════════════════
    "Apple___Apple_scab": {
        "crop": "Apple",
        "disease": "Apple Scab",
        "is_healthy": False,
        "pathogen": "Fungus — Venturia inaequalis",
        "symptoms": [
            "Olive-brown to black velvety spots on leaf surfaces",
            "Distorted or crinkled leaves with scab lesions",
            "Dark corky scab lesions on fruit surface",
            "Premature leaf drop in severe infections",
            "Fruit cracking and deformation",
        ],
        "treatment": [
            "Apply fungicides: Captan, Mancozeb, or Myclobutanil at bud break",
            "Use copper-based fungicide sprays (preventive)",
            "Remove and destroy infected leaves and fallen fruit",
            "Re-apply every 7–10 days during wet spring weather",
        ],
        "prevention": [
            "Plant scab-resistant apple varieties",
            "Prune for good air circulation",
            "Rake and remove fallen leaves in autumn",
            "Avoid overhead irrigation",
        ],
        "severity": "Moderate to High",
    },

    "Apple___Black_rot": {
        "crop": "Apple",
        "disease": "Apple Black Rot",
        "is_healthy": False,
        "pathogen": "Fungus — Botryosphaeria obtusa",
        "symptoms": [
            "Purple flecks on upper leaf surface enlarging to brown frogeye lesions",
            "Black mummified fruit remaining on the tree",
            "Cankers on branches with dead bark and gummosis",
            "Fruit rot starting at the calyx end",
        ],
        "treatment": [
            "Apply Captan or Thiophanate-methyl fungicide",
            "Remove all mummified fruit and prune cankered wood",
            "Cut at least 15 cm below the visible canker margin",
            "Apply protective fungicide spray during bloom period",
        ],
        "prevention": [
            "Maintain tree health through balanced fertilisation",
            "Remove dead wood and fire blight cankers promptly",
            "Practice sanitation: collect and destroy fallen fruit",
            "Avoid bark injuries during pruning",
        ],
        "severity": "Moderate",
    },

    "Apple___Cedar_apple_rust": {
        "crop": "Apple",
        "disease": "Cedar Apple Rust",
        "is_healthy": False,
        "pathogen": "Fungus — Gymnosporangium juniperi-virginianae",
        "symptoms": [
            "Bright orange-yellow spots on upper leaf surface",
            "Pale orange tube-like aecia (spore structures) on leaf undersides",
            "Premature defoliation in severe infections",
            "Yellow-green spots on fruit turning orange and distorted",
        ],
        "treatment": [
            "Apply Myclobutanil or Trifloxystrobin fungicide",
            "Begin spray at pink-bud stage before bloom",
            "Repeat every 7–10 days during wet spring weather",
        ],
        "prevention": [
            "Plant rust-resistant apple varieties",
            "Remove nearby juniper and eastern red cedar (alternate hosts) within 300 m",
            "Apply lime sulfur spray during the dormant season",
        ],
        "severity": "Low to Moderate",
    },

    "Apple___healthy": {
        "crop": "Apple",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": None,
        "symptoms": [],
        "treatment": ["No treatment needed. Continue regular care and monitoring."],
        "prevention": [
            "Annual pruning for structure and air circulation",
            "Monitor regularly for early signs of scab, rust, or fire blight",
            "Ensure adequate, consistent irrigation and balanced NPK fertilisation",
        ],
        "severity": "None",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CORN / MAIZE
    # ═══════════════════════════════════════════════════════════════════════
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "crop": "Corn (Maize)",
        "disease": "Gray Leaf Spot — Cercospora Leaf Spot",
        "is_healthy": False,
        "pathogen": "Fungus — Cercospora zeae-maydis",
        "symptoms": [
            "Rectangular, tan-to-gray lesions parallel to leaf veins (1–6 cm long)",
            "Lesions initially yellow-green with gray centre",
            "Coalescing lesions create large areas of dead leaf tissue",
            "Premature death of leaves progressing from bottom upward",
        ],
        "treatment": [
            "Apply Azoxystrobin, Propiconazole, or Pyraclostrobin fungicide",
            "Time application at VT/silking stage for maximum protection",
            "Use strobilurin fungicides in high-risk / high-humidity areas",
        ],
        "prevention": [
            "Plant hybrid varieties with GLS resistance ratings",
            "Rotate crops — avoid continuous corn",
            "Reduce surface crop residue through tillage",
            "Maintain proper plant spacing for air flow",
        ],
        "severity": "High",
    },

    "Corn_(maize)___Common_rust_": {
        "crop": "Corn (Maize)",
        "disease": "Common Rust",
        "is_healthy": False,
        "pathogen": "Fungus — Puccinia sorghi",
        "symptoms": [
            "Small, oval to elongate brick-red to brown pustules on both leaf surfaces",
            "Pustules turn black (telia) as the season progresses",
            "Chlorotic halos surrounding rust pustules",
            "Severe infection causes premature leaf death",
        ],
        "treatment": [
            "Apply Propiconazole or Mancozeb early in disease development",
            "Most effective at early tassel stage (VT/R1)",
            "Use triazole or strobilurin fungicides",
        ],
        "prevention": [
            "Plant rust-resistant hybrids",
            "Early planting to avoid peak rust season",
            "Monitor fields regularly in cool, humid conditions",
        ],
        "severity": "Moderate",
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Corn (Maize)",
        "disease": "Northern Leaf Blight",
        "is_healthy": False,
        "pathogen": "Fungus — Exserohilum turcicum",
        "symptoms": [
            "Long cigar-shaped gray-green to tan lesions (2.5–15 cm)",
            "Lesions start on lower leaves and progress upward",
            "Dark sooty spore masses visible in lesion centres",
            "Premature leaf death and significant yield loss before silking",
        ],
        "treatment": [
            "Apply Azoxystrobin, Propiconazole, or Trifloxystrobin fungicide",
            "Apply at V8–VT growth stage for best protection",
            "Repeat if disease pressure persists through silking",
        ],
        "prevention": [
            "Plant resistant or tolerant hybrid varieties (Ht gene)",
            "Crop rotation to reduce soilborne inoculum",
            "Tillage to bury infected crop debris",
            "Avoid dense planting populations",
        ],
        "severity": "High",
    },

    "Corn_(maize)___healthy": {
        "crop": "Corn (Maize)",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": None,
        "symptoms": [],
        "treatment": ["No treatment needed."],
        "prevention": [
            "Balanced NPK + micronutrient fertilisation",
            "Regular scouting for pests (earworm, fall armyworm)",
            "Maintain adequate soil moisture during tasselling",
        ],
        "severity": "None",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # BELL PEPPER
    # ═══════════════════════════════════════════════════════════════════════
    "Pepper__bell___Bacterial_spot": {
        "crop": "Bell Pepper",
        "disease": "Bacterial Spot",
        "is_healthy": False,
        "pathogen": "Bacterium — Xanthomonas euvesicatoria",
        "symptoms": [
            "Small, water-soaked spots on leaves with yellow halo",
            "Spots turn dark brown with irregular margins and may coalesce",
            "Severe defoliation, especially in warm, wet conditions",
            "Raised scabby lesions on fruit with water-soaked borders",
            "Fruit drop under high disease pressure",
        ],
        "treatment": [
            "Apply copper hydroxide + Mancozeb combination spray",
            "Spray every 5–7 days during warm, wet weather",
            "Apply in early morning to avoid phytotoxicity from midday heat",
        ],
        "prevention": [
            "Use certified disease-free or resistant seed",
            "Treat seeds with hot water (52 °C for 30 min)",
            "Avoid overhead irrigation — use drip systems",
            "Crop rotation: minimum 2 years without solanaceous crops",
        ],
        "severity": "High",
    },

    "Pepper__bell___healthy": {
        "crop": "Bell Pepper",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": None,
        "symptoms": [],
        "treatment": ["No treatment needed."],
        "prevention": [
            "Crop rotation",
            "Proper plant spacing",
            "Balanced calcium and potassium nutrition",
        ],
        "severity": "None",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # POTATO
    # ═══════════════════════════════════════════════════════════════════════
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "is_healthy": False,
        "pathogen": "Fungus — Alternaria solani",
        "symptoms": [
            "Dark brown circular spots with concentric rings (target / bull-eye pattern)",
            "Yellow halo surrounding spots",
            "Lesions appear first on older, lower leaves",
            "Severe infection causes progressive defoliation from base to top",
            "Dark, leathery, sunken lesions on tuber surface",
        ],
        "treatment": [
            "Apply Mancozeb, Chlorothalonil, or Copper fungicide",
            "Use systemic fungicides: Azoxystrobin or Propiconazole for better control",
            "Begin spraying at first symptom appearance",
            "Repeat every 7–14 days depending on weather",
        ],
        "prevention": [
            "Use certified disease-free seed potatoes",
            "Maintain adequate plant nutrition (especially K and N)",
            "Use drip irrigation — avoid wetting foliage",
            "Rotate crops for minimum 2–3 years",
        ],
        "severity": "Moderate",
    },

    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "is_healthy": False,
        "pathogen": "Oomycete — Phytophthora infestans (caused the 1840s Irish Potato Famine)",
        "symptoms": [
            "Pale green to brown, water-soaked lesions starting at leaf edges or tips",
            "White cottony mould on leaf undersides in humid conditions",
            "Lesions expand very rapidly under cool, moist weather",
            "Brown discoloration of infected stems",
            "Reddish-brown discoloration from skin inward on tubers",
        ],
        "treatment": [
            "Apply Metalaxyl-M + Mancozeb or Cymoxanil IMMEDIATELY at first sign",
            "Use Fosetyl-Al or Dimethomorph for systemic protection",
            "Spray every 5–7 days during cool, wet periods",
            "Destroy infected haulms before harvest to protect tubers",
        ],
        "prevention": [
            "Plant blight-resistant varieties (Sarpo Mira, Defender)",
            "Use certified disease-free seed",
            "Hill soil around plants to protect tubers from spore wash-down",
            "Destroy cull piles and volunteer potato plants",
            "Monitor forecasting alerts (BlightCast) for spray timing",
        ],
        "severity": "Critical — can cause total crop loss",
    },

    "Potato___healthy": {
        "crop": "Potato",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": None,
        "symptoms": [],
        "treatment": ["No treatment needed."],
        "prevention": [
            "Regular hilling to protect tubers",
            "Scout for late blight signs in wet weather",
            "Balanced potassium and nitrogen fertilisation",
        ],
        "severity": "None",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # TOMATO
    # ═══════════════════════════════════════════════════════════════════════
    "Tomato_Bacterial_spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot",
        "is_healthy": False,
        "pathogen": "Bacterium — Xanthomonas vesicatoria / euvesicatoria / gardneri / perforans",
        "symptoms": [
            "Small, water-soaked dark spots on leaves with yellow halo",
            "Spots turn dark brown with irregular edges; may defoliate plant",
            "Raised, scabby brown lesions on green fruit",
            "Water-soaked lesions on seedling leaves",
        ],
        "treatment": [
            "Apply copper hydroxide + Mancozeb combination bactericide",
            "Spray every 5–7 days during warm, wet weather",
            "Avoid working in the field when foliage is wet",
        ],
        "prevention": [
            "Use certified disease-free or resistant seed varieties",
            "Hot water seed treatment: 50 °C for 25 min",
            "Crop rotation: 2–3 years without solanaceous crops",
            "Avoid overhead irrigation",
        ],
        "severity": "High",
    },

    "Tomato_Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "is_healthy": False,
        "pathogen": "Fungus — Alternaria solani",
        "symptoms": [
            "Brown circular spots with concentric rings forming a target-board pattern",
            "Yellow halo surrounding each lesion",
            "Lesions first appear on older, lower leaves",
            "Severe defoliation exposes fruit to sunscald",
            "Dark leathery lesions at the stem end of fruit",
        ],
        "treatment": [
            "Apply Mancozeb, Chlorothalonil, or Azoxystrobin fungicide",
            "Begin application at first symptom appearance",
            "Repeat every 7–10 days",
            "Remove infected lower leaves promptly",
        ],
        "prevention": [
            "Use drip irrigation to keep foliage dry",
            "Stake or cage plants to improve air circulation",
            "Mulch soil surface to reduce splash dispersal",
            "Rotate crops for 2–3 years",
        ],
        "severity": "Moderate",
    },

    "Tomato_Late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight",
        "is_healthy": False,
        "pathogen": "Oomycete — Phytophthora infestans",
        "symptoms": [
            "Pale green to dark brown, greasy, water-soaked lesions on leaves",
            "White downy sporulation on leaf undersides in humid conditions",
            "Large, irregular dark brown lesions that spread rapidly",
            "Brown, greasy cankers on stems",
            "Dark brown, firm, irregular lesions on fruit",
        ],
        "treatment": [
            "Apply Metalaxyl-M + Mancozeb, Cymoxanil, or Fosetyl-Al IMMEDIATELY",
            "Spray every 5–7 days during cool, wet periods",
            "Remove and destroy infected plant material — do not compost",
        ],
        "prevention": [
            "Plant resistant varieties (Legend, Mountain Magic, Plum Regal)",
            "Avoid overhead irrigation",
            "Ensure well-drained soil and raised beds",
            "Destroy cull piles and volunteer plants",
        ],
        "severity": "Critical",
    },

    "Tomato_Leaf_Mold": {
        "crop": "Tomato",
        "disease": "Leaf Mold",
        "is_healthy": False,
        "pathogen": "Fungus — Passalora fulva (formerly Fulvia fulva)",
        "symptoms": [
            "Pale greenish-yellow spots on upper leaf surface",
            "Olive-green to grayish-brown velvety mould on leaf underside",
            "Infected leaves turn yellow then brown and wither",
            "Most common in greenhouses and poly-houses with high humidity",
        ],
        "treatment": [
            "Apply Chlorothalonil, Copper oxychloride, or Mancozeb fungicide",
            "Remove and destroy infected leaves",
            "Ventilate greenhouses to reduce relative humidity",
        ],
        "prevention": [
            "Grow resistant tomato varieties (leaf mould immune lines)",
            "Keep relative humidity below 85% in protected structures",
            "Ensure adequate plant spacing for airflow",
            "Avoid wetting leaves during irrigation",
        ],
        "severity": "Moderate",
    },

    "Tomato_Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot",
        "is_healthy": False,
        "pathogen": "Fungus — Septoria lycopersici",
        "symptoms": [
            "Small circular spots (3–6 mm) with dark brown border and white/tan centre",
            "Dark specks (pycnidia / fruiting bodies) visible in lesion centre",
            "Numerous spots first on lower, older leaves",
            "Progressive yellowing and defoliation from base upward",
        ],
        "treatment": [
            "Apply Mancozeb, Chlorothalonil, or Copper fungicide",
            "Start spray programme when first spots appear",
            "Repeat every 7–10 days in wet conditions",
        ],
        "prevention": [
            "Rotate crops: avoid tomatoes in same site for 1–2 years",
            "Remove plant debris at end of season",
            "Mulch soil surface to prevent splash spore dispersal",
            "Stake or cage plants to improve air circulation",
        ],
        "severity": "Moderate to High",
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "crop": "Tomato",
        "disease": "Spider Mites — Two-Spotted Spider Mite Infestation",
        "is_healthy": False,
        "pathogen": "Arachnid pest — Tetranychus urticae (not a fungus or bacterium)",
        "symptoms": [
            "Fine stippling (tiny pale yellow/white dots) on upper leaf surface",
            "Bronzing or silvering of severely infested leaves",
            "Fine silken webbing on leaf undersides and between stems",
            "Leaves dry out, curl, and may drop prematurely",
            "Tiny moving specks (mites) visible on leaf undersides with hand lens",
        ],
        "treatment": [
            "Apply miticide: Abamectin, Bifenazate, or Hexythiazox — rotate modes of action",
            "Spray leaf undersides thoroughly for contact activity",
            "Apply insecticidal soap or neem oil for organic / IPM control",
            "Release predatory mites (Phytoseiulus persimilis) for biological control",
        ],
        "prevention": [
            "Maintain adequate plant irrigation (drought stress worsens mite outbreaks)",
            "Avoid broad-spectrum insecticides that eliminate natural mite predators",
            "Monitor leaf undersides regularly, especially during hot, dry periods",
            "Use reflective silver mulches to deter mite colonisation",
        ],
        "severity": "Moderate to High",
    },

    "Tomato__Target_Spot": {
        "crop": "Tomato",
        "disease": "Target Spot",
        "is_healthy": False,
        "pathogen": "Fungus — Corynespora cassiicola",
        "symptoms": [
            "Brown, circular to irregular lesions with visible concentric ring pattern",
            "Lesions coalesce to create large areas of dead tissue",
            "Infected leaves yellow and drop prematurely",
            "Dark lesions with concentric rings on fruit surface",
            "Cankers possible on stems",
        ],
        "treatment": [
            "Apply Azoxystrobin, Chlorothalonil, or Mancozeb fungicide",
            "Spray every 7–14 days when wet, warm conditions prevail",
        ],
        "prevention": [
            "Prune lower leaves to improve air circulation",
            "Avoid overhead irrigation",
            "Remove and destroy plant debris promptly after harvest",
        ],
        "severity": "Moderate",
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "crop": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "is_healthy": False,
        "pathogen": "Gemini virus — TYLCV (vector: silverleaf whitefly, Bemisia tabaci)",
        "symptoms": [
            "Upward curling and cupping of young leaves",
            "Yellowing of leaf margins and interveinal areas on new growth",
            "Stunted, bushy plant appearance",
            "Severely reduced fruit set: small, pale, misshapen fruit",
            "Infected plants remain stunted and unproductive",
        ],
        "treatment": [
            "No cure — infected plants should be removed and destroyed immediately",
            "Control whitefly vector with imidacloprid or thiamethoxam systemic insecticides",
            "Use reflective silver mulch to deter whitefly landing",
            "Apply neem oil or insecticidal soap as repellent",
        ],
        "prevention": [
            "Plant TYLCV-resistant or tolerant varieties (Ty-1 gene)",
            "Deploy yellow sticky traps to monitor whitefly populations",
            "Install 50-mesh screens in greenhouses",
            "Remove weeds (especially solanaceous weeds) that harbour whiteflies",
        ],
        "severity": "Critical — no cure available",
    },

    "Tomato__Tomato_mosaic_virus": {
        "crop": "Tomato",
        "disease": "Tomato Mosaic Virus (ToMV)",
        "is_healthy": False,
        "pathogen": "RNA virus — Tomato Mosaic Virus (highly stable, contact-transmitted)",
        "symptoms": [
            "Mosaic pattern: alternating light and dark green / yellow patches on leaves",
            "Leaf malformation: curling, bubbling, and abnormal elongation",
            "Stunted, unproductive plant growth",
            "Reduced fruit set; fruit may show internal browning",
            "Brown streaking on stems in severe cases",
        ],
        "treatment": [
            "No chemical cure available",
            "Remove and destroy infected plants immediately to prevent spread",
            "Disinfect all tools with 10% bleach or trisodium phosphate (TSP) solution",
            "Wash hands thoroughly after handling any infected plants",
        ],
        "prevention": [
            "Use certified virus-free / indexed seed",
            "Plant resistant varieties carrying the Tm-2a resistance gene",
            "Control aphid and thrips vectors",
            "Avoid handling tobacco products near plants (TMV can infect tomato)",
            "Disinfect tools and wash hands between rows",
        ],
        "severity": "High — no chemical treatment",
    },

    "Tomato_healthy": {
        "crop": "Tomato",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": None,
        "symptoms": [],
        "treatment": ["No treatment needed. Continue regular monitoring."],
        "prevention": [
            "Regular field scouting for early disease and pest detection",
            "Balanced NPK fertilisation with calcium supplementation",
            "Consistent drip irrigation management",
        ],
        "severity": "None",
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_disease_info(class_name: str) -> Dict[str, Any]:
    """
    Retrieve structured disease information for a given class name.

    Args:
        class_name: Exact class name as used in config.CLASS_NAMES.

    Returns:
        Dictionary with crop, disease, symptoms, treatment, and prevention keys.
        Falls back to a generic record if the class_name is not found.
    """
    info = DISEASE_DATABASE.get(class_name)
    if info is not None:
        return info

    # Graceful fallback: parse crop/disease from the class name format
    parts = class_name.split("___")
    crop    = parts[0].replace("_", " ").replace(",", "").strip() if parts else "Unknown"
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"

    return {
        "crop": crop,
        "disease": disease,
        "is_healthy": "healthy" in class_name.lower(),
        "pathogen": "Unknown — consult local agricultural extension office",
        "symptoms": ["Consult a local agricultural expert for accurate diagnosis."],
        "treatment": ["Consult a local agricultural expert."],
        "prevention": ["Consult a local agricultural expert."],
        "severity": "Unknown",
    }


def list_all_diseases() -> list[str]:
    """Return a list of all disease / class names in the database."""
    return list(DISEASE_DATABASE.keys())


def get_all_crops() -> list[str]:
    """Return a deduplicated list of crop names in the database."""
    return sorted({v["crop"] for v in DISEASE_DATABASE.values()})
