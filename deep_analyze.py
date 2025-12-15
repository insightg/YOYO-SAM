#!/usr/bin/env python3
"""
Deep Object Analysis using LLM (OpenRouter/Gemini)
Analyzes cropped detection images to provide detailed classification.
Context-aware: passes the detected category to focus the analysis.
"""

import os
import io
import base64
import re
import httpx
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.0-flash-001"


def build_contextual_prompt(category: str) -> str:
    """
    Build a context-aware prompt that focuses on the specific category.
    The LLM will ignore other objects and only classify within the given category.
    """

    # Category-specific guidance
    category_hints = {
        "road sign": "tipi: limite_velocita_XX, divieto_sosta, dare_precedenza, stop, senso_unico, divieto_accesso, pericolo_generico, curva_pericolosa, incrocio, passaggio_pedonale, divieto_sorpasso, strada_prioritaria, fine_divieto",
        "street sign": "tipi: nome_via, numero_civico, indicazione_direzione, località, km_distanza",
        "traffic light": "stati: rosso, giallo, verde, rosso_pedonale, verde_pedonale, lampeggiante_giallo, spento, freccia_verde_destra, freccia_verde_sinistra",
        "speed limit sign": "tipi: limite_20, limite_30, limite_50, limite_60, limite_70, limite_80, limite_90, limite_110, limite_130",
        "road marking": "tipi: attraversamento_pedonale, linea_continua, linea_tratteggiata, freccia_direzione, stop, dare_precedenza, parcheggio, divieto_sosta, zig_zag",
        "road crack": "tipi: fessura_longitudinale, fessura_trasversale, fessura_ramificata, buca, cedimento, crepa_alligatore, giunzione_deteriorata",
        "pothole": "tipi: buca_piccola, buca_media, buca_grande, avvallamento, cedimento_bordo",
        "manhole": "tipi: tombino_fognatura, tombino_acqua, tombino_gas, tombino_elettrico, tombino_telecomunicazioni, chiusino_quadrato, chiusino_rotondo",
        "guardrail": "tipi: guardrail_metallico, guardrail_cemento, barriera_new_jersey, guard_rail_doppio, terminale_guardrail",
        "pole": "tipi: palo_illuminazione, palo_segnaletica, palo_semaforo, palo_elettrico, palo_telecomunicazioni",
        "curb": "tipi: cordolo_standard, cordolo_ribassato, cordolo_rampa_disabili, cordolo_danneggiato, bordo_marciapiede",
        "sidewalk": "tipi: marciapiede_integro, marciapiede_danneggiato, pavimentazione_dissestata, mattonelle_rotte, radici_affioranti",
        "crosswalk": "tipi: attraversamento_zebrato, attraversamento_rialzato, attraversamento_semaforico, attraversamento_ciclabile",
        "fire hydrant": "tipi: idrante_soprasuolo, idrante_sottosuolo, idrante_a_colonna, idrante_murale",
        "vehicle": "tipi: automobile, motocicletta, bicicletta, camion, autobus, furgone, scooter",
        "pedestrian": "tipi: pedone_adulto, pedone_bambino, pedone_anziano, persona_carrozzina, persona_passeggino",
    }

    # Get category-specific hints
    category_lower = category.lower()
    hints = ""
    for key, value in category_hints.items():
        if key in category_lower:
            hints = f"\nSottotipi comuni per {category}: {value}"
            break

    prompt = f"""CONTESTO: Questa immagine contiene un oggetto rilevato come "{category}".
Il tuo compito è identificare il SOTTOTIPO SPECIFICO di questo {category}.

IMPORTANTE:
- Concentrati SOLO sull'oggetto "{category}" nell'immagine
- IGNORA completamente qualsiasi altro oggetto visibile (veicoli, persone, altri elementi)
- Classifica SOLO il {category}, non altri elementi presenti
- Se l'oggetto {category} non è chiaramente identificabile, rispondi "non_identificabile"
{hints}

FORMATO RISPOSTA:
Rispondi con UNA SOLA PAROLA o frase breve (max 3 parole) in italiano, usando underscore al posto degli spazi.
NON aggiungere spiegazioni, solo l'identificatore del sottotipo.

Qual è il sottotipo specifico di questo {category}?"""

    return prompt


def crop_detection(image: Image.Image, bbox: list, padding_percent: float = 0.1) -> Image.Image:
    """Crop the detection area from the image with optional padding."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Add padding
    pad_x = width * padding_percent
    pad_y = height * padding_percent

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image.width, x2 + pad_x)
    y2 = min(image.height, y2 + pad_y)

    return image.crop((int(x1), int(y1), int(x2), int(y2)))


def image_to_base64(image: Image.Image, max_size: int = 1024) -> str:
    """Convert PIL Image to base64 string, resizing if needed."""
    # Resize if too large
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to JPEG bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def analyze_with_llm(image: Image.Image, category: str) -> str:
    """Send image to LLM for deep analysis with category context."""
    if not OPENROUTER_API_KEY:
        return "errore_api_key_mancante"

    # Build contextual prompt with the category
    prompt = build_contextual_prompt(category)
    image_b64 = image_to_base64(image)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://sam3-tool.local",
        "X-Title": "SAM3 Deep Analysis"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 50,
        "temperature": 0.1
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(OPENROUTER_URL, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()

            # Clean up the response - extract only the classification
            # Remove any explanation, keep only the identifier
            content = content.split("\n")[0].strip()
            # Remove common prefixes the model might add
            content = re.sub(r'^(il |la |lo |l\'|un |una |tipo:|sottotipo:)', '', content, flags=re.IGNORECASE)
            content = re.sub(r'[^a-zA-Z0-9_àèéìòù]', '_', content.lower())
            content = re.sub(r'_+', '_', content).strip('_')

            if len(content) > 50:
                content = content[:50]

            return content if content else "non_identificabile"

    except httpx.TimeoutException:
        return "timeout"
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code} - {e.response.text}")
        return "errore_api"
    except Exception as e:
        print(f"Error in LLM analysis: {e}")
        return "errore_analisi"


def deep_analyze_detection(
    image_path: Path,
    detection: dict,
    base_class: str
) -> str:
    """
    Perform deep analysis on a detection.

    Args:
        image_path: Path to the full image
        detection: Detection dict with bbox
        base_class: The base class name (without "- deep" suffix)

    Returns:
        New class name in format: base_class.subclass
    """
    try:
        # Load and crop image
        image = Image.open(image_path).convert("RGB")
        cropped = crop_detection(image, detection["bbox"])

        # Analyze with LLM, passing the category context
        subclass = analyze_with_llm(cropped, base_class)

        # Format result
        base_clean = base_class.replace(" ", "_").lower()

        # If not identifiable, return base_class.generic
        if subclass in ("non_identificabile", "errore_api", "errore_analisi", "timeout"):
            return f"{base_clean}.generic"

        return f"{base_clean}.{subclass}"

    except Exception as e:
        print(f"Error in deep analysis: {e}")
        base_clean = base_class.replace(" ", "_").lower()
        return f"{base_clean}.generic"


def parse_threshold_prefix(class_name: str) -> tuple[str, float | None]:
    """
    Parse optional threshold prefix from class name.

    Format: "30 light poles" -> ("light poles", 0.30)
            "light poles" -> ("light poles", None)

    Returns:
        (class_name_without_prefix, threshold_or_none)
    """
    import re
    name = class_name.strip()
    # Match 2-digit number at start followed by space
    match = re.match(r'^(\d{2})\s+(.+)$', name)
    if match:
        threshold = int(match.group(1)) / 100.0
        return match.group(2), threshold
    return name, None


def is_deep_class(class_name: str) -> tuple[bool, str, float | None]:
    """
    Check if a class requires deep analysis.

    Returns:
        (is_deep, base_class_name, threshold_or_none)
    """
    # First extract threshold prefix if present
    name, threshold = parse_threshold_prefix(class_name)

    if name.lower().endswith("- deep"):
        base_class = name.rsplit("-", 1)[0].strip()
        return True, base_class, threshold
    return False, name, threshold


if __name__ == "__main__":
    # Test
    print(f"API Key configured: {bool(OPENROUTER_API_KEY)}")
    print(f"Model: {MODEL}")

    # Test class detection with thresholds
    test_classes = [
        "road sign - deep",
        "road crack - deep",
        "traffic light",
        "manhole - deep",
        "stop sign",
        "30 light pole",
        "40 road sign - deep",
        "50 traffic sign"
    ]

    print("\nTest classi (con soglie):")
    for cls in test_classes:
        is_deep, base, threshold = is_deep_class(cls)
        print(f"  {cls} -> deep={is_deep}, base='{base}', threshold={threshold}")

    # Show example prompt
    print("\n--- Esempio prompt per 'road crack' ---")
    print(build_contextual_prompt("road crack"))
