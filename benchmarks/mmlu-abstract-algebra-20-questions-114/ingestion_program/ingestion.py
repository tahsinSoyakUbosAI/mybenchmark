#!/usr/bin/env python3
"""
Coktan Secmeli Kiyaslama - Alma Programi (LLM-as-Judge Destekli)
Multiple Choice Benchmark - Ingestion Program (with LLM-as-Judge support)

Bu program, kullanicinin model.py dosyasindan LLM yapilandirmasini alir
ve sorulari LLM'e gondererek cevaplari toplar.

This program takes the LLM configuration from user's model.py
and sends questions to LLM to collect answers.
"""

import json
import os
import sys
import time
import traceback
import requests
import base64
import re
from typing import Dict, List, Any, Optional, Tuple

# ============================================================================
# REDIRECT STDERR TO STDOUT - All logs will appear in stdout
# ============================================================================
# This ensures all output (warnings, errors, library logs) go to stdout
# so they appear in the "Ingestion stdout" tab instead of stderr
sys.stderr = sys.stdout

# CodaBench icin standart dizin yollari
INPUT_DIR = "/app/input_data"
OUTPUT_DIR = "/app/output"
PROGRAM_DIR = "/app/program"
SUBMISSION_DIR = "/app/ingested_program"

sys.path.append(SUBMISSION_DIR)
sys.path.append(PROGRAM_DIR)

# ============================================================================
# BENCHMARK LANGUAGE (tr | en)
# Selects the prompt/instruction language. Default "tr" preserves the original
# Turkish behaviour; set BENCHMARK_LANGUAGE=en for English datasets such as MMLU.
# ============================================================================
_DETECTED_LANGUAGE = None

# Letters that exist in Turkish but not in English (and not in Cyrillic text).
_TURKISH_ONLY_CHARS = set("ğĞıİşŞ")


def detect_benchmark_language(questions) -> str:
    """Pick the instruction language from the questions themselves.

    "tr" only when the benchmark is actually Turkish; everything else (English,
    Ukrainian, any other language) gets the English instruction set, whose
    REASONING/ANSWER keywords every model knows. Before this, the language came
    only from BENCHMARK_LANGUAGE, which the platform never sets, so EVERY
    benchmark was instructed in Turkish: a Ukrainian quiz asked a model to
    answer "GEREKCE: ... CEVAP: ...", which small models cannot follow.
    """
    global _DETECTED_LANGUAGE
    sample = " ".join(str(q.get("question", "")) for q in (questions or [])[:50])
    hits = sum(1 for ch in sample if ch in _TURKISH_ONLY_CHARS)
    _DETECTED_LANGUAGE = "tr" if hits >= 3 else "en"
    return _DETECTED_LANGUAGE


def get_benchmark_language() -> str:
    """Return the benchmark prompt language (tr|en).

    An explicit BENCHMARK_LANGUAGE wins (read LAZILY, not at import, because it
    is injected via the imported model.py, which loads after this module).
    Otherwise the language detected from the questions; "tr" only if nothing
    was detected, which preserves the original behaviour for direct callers.
    """
    explicit = (os.environ.get("BENCHMARK_LANGUAGE") or "").strip().lower()
    if explicit in ("tr", "en"):
        return explicit
    return _DETECTED_LANGUAGE or "tr"


# ============================================================================
# KATEGORI ISTEMLERI (KATILIMCILAR GORMEZ)
# CATEGORY PROMPTS (HIDDEN FROM PARTICIPANTS)
# ============================================================================

CATEGORY_PROMPTS = {
    'turkish_grammar': """Türkçe dilbilgisi sorusu. Doğru cevabı seç.

Soru: {question}

Seçenekler:
{choices}

{reasoning_instruction}""",

    'general_knowledge': """Genel kültür sorusu. Doğru cevabı seç.

Soru: {question}

Seçenekler:
{choices}

{reasoning_instruction}""",

    'text_summarization': """Metin özetleme sorusu. En iyi özeti seç.

Soru: {question}

Seçenekler:
{choices}

{reasoning_instruction}""",

    'sentence_completion': """Cümle tamamlama sorusu. Boşluğa uygun seçeneği bul.

Soru: {question}

Seçenekler:
{choices}

{reasoning_instruction}""",

    'spelling_errors': """Yazım hatası sorusu. Doğru veya yanlış yazımı bul.

Soru: {question}

Seçenekler:
{choices}

{reasoning_instruction}""",

    'mathematics': """Matematik sorusu. Doğru cevabı hesapla.

Soru: {question}

Seçenekler:
{choices}

{reasoning_instruction}""",

    'translation': """Çeviri sorusu. En iyi çeviriyi seç.

Soru: {question}

Seçenekler:
{choices}

{reasoning_instruction}""",

    'coding': """Kodlama sorusu. Doğru kod veya çıktıyı seç.

Soru: {question}

Seçenekler:
{choices}

{reasoning_instruction}""",

    'image_interpretation': """Görsel yorumlama sorusu. Resmi incele ve doğru cevabı seç.

Soru: {question}

Seçenekler:
{choices}

{reasoning_instruction}"""
}

DEFAULT_PROMPT = """Aşağıdaki soruyu dikkatlice oku ve en uygun cevabı seç.

Soru: {question}

Seçenekler:
{choices}

{reasoning_instruction}"""

REASONING_INSTRUCTION = """Lütfen şu formatta cevap ver:
GEREKCE: [Neden bu cevabı seçtiğinizi kısa açıklayınız]
CEVAP: [A, B, C veya D]"""

NO_REASONING_INSTRUCTION = """Sadece doğru cevabın harfini yaz (A, B, C veya D).
Cevap:"""


# ============================================================================
# ENGLISH PROMPTS (BENCHMARK_LANGUAGE=en) — used for datasets such as MMLU.
# MMLU subjects don't map to the Turkish category keys, so they fall through to
# DEFAULT_PROMPT_EN, a generic multiple-choice prompt.
# ============================================================================
DEFAULT_PROMPT_EN = """Read the following multiple-choice question carefully and select the best answer.

Question: {question}

Options:
{choices}

{reasoning_instruction}"""

CATEGORY_PROMPTS_EN = {
    'mathematics': """Solve the following mathematics question and select the correct answer.

Question: {question}

Options:
{choices}

{reasoning_instruction}""",
    'coding': """Answer the following programming question. Select the correct code or output.

Question: {question}

Options:
{choices}

{reasoning_instruction}""",
}

REASONING_INSTRUCTION_EN = """Please answer in this format (keep the words REASONING and ANSWER exactly as written; the explanation may be in the language of the question):
REASONING: [Briefly explain why you chose this answer]
ANSWER: [A, B, C or D]"""

NO_REASONING_INSTRUCTION_EN = """Only write the letter of the correct answer (A, B, C or D).
Answer:"""


def load_json(path: str) -> Any:
    """JSON dosyasini yukle"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_file(base_dir: str, filename: str) -> Optional[str]:
    """Dizin agacinda dosya ara"""
    direct_path = os.path.join(base_dir, filename)
    if os.path.exists(direct_path):
        return direct_path
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def load_image(image_path: str) -> str:
    """Resmi base64'e donustur"""
    possible_paths = [
        os.path.join(INPUT_DIR, image_path),
        os.path.join(INPUT_DIR, 'images', os.path.basename(image_path)),
        os.path.join('/app/input/', image_path),
        os.path.join('/app/', image_path),
        image_path
    ]
    
    for full_path in possible_paths:
        try:
            with open(full_path, 'rb') as f:
                print(f"  Resim yuklendi: {full_path}")
                return base64.b64encode(f.read()).decode('utf-8')
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"  Resim hatasi ({full_path}): {e}")
            continue
    
    print(f"  Resim bulunamadi: {image_path}")
    return ""


_ANSWER_MARKERS = r"(?:DOGRU CEVAP|DOĞRU CEVAP|CEVAP|FINAL ANSWER|ANSWER|ВІДПОВІДЬ|ОТВЕТ)"


def extract_answer(response: str, valid_choices: List[str] = ['A', 'B', 'C', 'D'],
                   choice_texts: Optional[List[str]] = None) -> str:
    """Pull the chosen letter out of a model response, or "" when there is none.

    Never guesses. The previous version fell back to "the first A-D character
    anywhere in the text", which is not a guess but a constant: the keyword
    GEREKCE contains a C and REASONING contains an A, so every response that
    lacked a final CEVAP/ANSWER line was scored as that letter, even when the
    model had named the right option. An unparseable response now counts as
    unanswered, which is what it is.
    """
    if not response:
        return ""
    letters = "".join(valid_choices)
    text = response.strip()
    upper = text.upper()

    # 1. Explicit final-answer marker. LAST occurrence wins (the reasoning may
    #    say "the answer" earlier), and the letter must stand alone, so
    #    "ANSWER BECAUSE ..." is not read as B.
    marked = re.findall(_ANSWER_MARKERS + r"\s*[:\-]?\s*[\[\(\*\"']*\s*([" + letters + r"])(?![A-Z])", upper)
    if marked:
        return marked[-1]
    # 2. A single letter in brackets or parentheses: "[A] Andromeda", "(B)".
    bracketed = re.findall(r"[\[\(]\s*([" + letters + r"])\s*[\]\)]", upper)
    if bracketed:
        return bracketed[-1]
    # 3. An option-style line start: "B) Venus", "B. Venus", "B: Venus".
    line_start = re.findall(r"(?m)^\s*([" + letters + r"])\s*[\)\.:]", upper)
    if line_start:
        return line_start[-1]
    # 4. The whole response is just the letter ("B", "b.").
    bare = re.fullmatch(r"\W*([" + letters + r"])\W*", upper)
    if bare:
        return bare.group(1)
    # 5. The response names exactly ONE option by its text ("Venus").
    if choice_texts:
        lowered = text.lower()
        named = [valid_choices[i] for i, t in enumerate(choice_texts[:len(valid_choices)])
                 if str(t).strip() and str(t).strip().lower() in lowered]
        if len(named) == 1:
            return named[0]
    return ""


def parse_reasoning_response(response: str, valid_choices: List[str] = ['A', 'B', 'C', 'D'],
                             choice_texts: Optional[List[str]] = None) -> Tuple[str, str]:
    """Gerekce ve cevabi ayristir"""
    if not response:
        return "", ""

    reasoning = ""
    # Reasoned format (tr|en): GEREKCE/REASONING: ... CEVAP/ANSWER: X
    reasoning_match = re.search(
        r'(?:GEREKCE|REASONING):\s*(.+?)(?=CEVAP:|ANSWER:|$)',
        response, re.DOTALL | re.IGNORECASE,
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    return extract_answer(response, valid_choices, choice_texts), reasoning


def build_prompt(question: Dict[str, Any], with_reasoning: bool = True) -> str:
    """Build the prompt for a question, honoring BENCHMARK_LANGUAGE (tr|en)."""
    category = question.get('category', 'general_knowledge')
    question_text = question.get('question', '')
    choices = question.get('choices', {})

    # Format choices
    labels = choices.get('label', ['A', 'B', 'C', 'D'])
    texts = choices.get('text', [])
    choices_str = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))

    # Select the language-appropriate prompt + reasoning instruction set.
    if get_benchmark_language() == "en":
        template = CATEGORY_PROMPTS_EN.get(category, DEFAULT_PROMPT_EN)
        reasoning_instruction = REASONING_INSTRUCTION_EN if with_reasoning else NO_REASONING_INSTRUCTION_EN
    else:
        template = CATEGORY_PROMPTS.get(category, DEFAULT_PROMPT)
        reasoning_instruction = REASONING_INSTRUCTION if with_reasoning else NO_REASONING_INSTRUCTION

    return template.format(
        question=question_text,
        choices=choices_str,
        reasoning_instruction=reasoning_instruction
    )


def call_llm_text(model, prompt: str) -> Optional[str]:
    """LLM API cagirisi (metin) - Model nesnesini kullanarak"""
    try:
        # Yeni model interface'i kontrol et
        if hasattr(model, 'call_llm'):
            return model.call_llm(prompt)
        
        # Eski config-based interface
        config = getattr(model, 'config', None)
        if config:
            headers = {
                "Authorization": f"Bearer {config.get('api_token', config.get('api_key', ''))}",
                "Content-Type": "application/json"
            }
            # Platform-injected spend attribution for LiteLLM (benchmark/run tags).
            _litellm_tags = os.environ.get("LITELLM_TAGS", "").strip()
            if _litellm_tags:
                headers["x-litellm-tags"] = _litellm_tags
            payload = {
                "model": config.get("model_name", ""),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": getattr(model, 'max_tokens', 300),
                "temperature": getattr(model, 'temperature', 0.1)
            }
            
            response = requests.post(
                config.get("api_url", ""),
                headers=headers,
                json=payload,
                timeout=getattr(model, 'timeout', 60)
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0].get('message', {}).get('content', '').strip()
        
        return None
    except Exception as e:
        print(f"LLM API hatasi: {e}")
        return None


def call_llm_vision(model, prompt: str, image_base64: str) -> Optional[str]:
    """LLM API cagirisi (gorsel) - Model nesnesini kullanarak"""
    try:
        # Yeni model interface'i kontrol et
        if hasattr(model, 'call_llm_vision'):
            return model.call_llm_vision(prompt, image_base64)
        
        # Vision destegi yoksa None dondur
        return None
    except Exception as e:
        print(f"Vision LLM API hatasi: {e}")
        return None


def _find_test_data() -> str:
    """Test verisini bul ve yolunu dondur."""
    data_path = find_file(INPUT_DIR, "testing_data.json")
    if not data_path:
        data_path = find_file(os.path.join(INPUT_DIR, "public_data"), "testing_data.json")
    if not data_path:
        data_path = find_file(INPUT_DIR, "benchmark.json")
    if not data_path:
        print("Hata: testing_data.json bulunamadi")
        sys.exit(1)
    return data_path


def _load_model():
    """Modeli yukle ve dondur."""
    try:
        from model import TurkishLanguageModel
        model = TurkishLanguageModel()
        print(f"Model yuklendi: {getattr(model, 'model_name', 'Bilinmiyor')}")
        vision_status = "Acik" if getattr(model, 'vision_enabled', False) else "Kapali"
        reasoning_status = "Acik" if getattr(model, 'reasoning_enabled', True) else "Kapali"
        print(f"  Vision: {vision_status}")
        print(f"  Reasoning: {reasoning_status}")
        print(f"  Temperature: {getattr(model, 'temperature', 0.1)}")
        print(f"  Max Tokens: {getattr(model, 'max_tokens', 300)}")
        return model
    except ImportError:
        from model import Model
        model = Model()
        print(f"Model yuklendi (eski format): {model.config.get('model_name', 'Bilinmiyor')}")
        return model


def _process_image_question(model, prompt, image_path, vision_enabled):
    """Gorsel soruyu isle."""
    if not vision_enabled:
        return None, True  # response=None, skipped=True
    image_base64 = load_image(image_path)
    if image_base64:
        return call_llm_vision(model, prompt, image_base64), False
    return None, False


def _create_prediction(question_id, category, response, with_reasoning, question=None):
    """Tahmin sozlugu olustur."""
    if not response:
        # No response is no answer. It used to be recorded as "A", which handed
        # a failed API call a one-in-four chance of scoring.
        return {
            "id": question_id,
            "prediction": "",
            "predicted_answer": "",
            "reasoning": '',
            "category": category,
            "confidence": 0.0
        }

    choices = (question or {}).get('choices', {}) or {}
    labels = list(choices.get('label') or ['A', 'B', 'C', 'D'])
    texts = list(choices.get('text') or [])

    if with_reasoning:
        answer, reasoning = parse_reasoning_response(response, labels, texts)
    else:
        answer = extract_answer(response, labels, texts)
        reasoning = ""

    return {
        "id": question_id,
        "prediction": answer,
        "predicted_answer": answer,
        "reasoning": reasoning,
        "category": category,
        "confidence": 1.0 if answer else 0.0
    }


def _save_results(predictions, categories, execution_time, vision_enabled, with_reasoning, vision_skipped):
    """Sonuclari kaydet."""
    reasoning_count = sum(1 for p in predictions if p.get('reasoning'))
    answered_count = sum(1 for p in predictions if p.get('predicted_answer'))
    
    print(f"\n{'='*60}")
    print("Sonuclar:")
    print(f"  Toplam soru: {len(predictions)}")
    print(f"  Cevaplanan: {answered_count}")
    print(f"  Gerekceli: {reasoning_count}")
    if vision_skipped > 0:
        print(f"  Gorsel atlanan: {vision_skipped} (vision kapali)")
    print(f"  Sure: {execution_time:.2f} saniye")
    print("=" * 60)
    
    output_data = {
        'predictions': predictions,
        'metadata': {
            'execution_time': execution_time,
            'total_questions': len(predictions),
            'answered_questions': answered_count,
            'categories': list(categories),
            'status': 'success',
            'model_settings': {
                'vision_enabled': vision_enabled,
                'reasoning_enabled': with_reasoning,
                'vision_skipped': vision_skipped
            },
            'reasoning_stats': {
                'total_with_reasoning': reasoning_count,
                'percentage': reasoning_count / len(predictions) * 100 if predictions else 0
            }
        }
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "predictions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Tahminler kaydedildi: {output_path}")


def main():
    print("=" * 60)
    print("Coktan Secmeli Kiyaslama - Alma Programi")
    print("(LLM-as-Judge Destekli)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test verisini bul ve yukle
    data_path = _find_test_data()
    questions = load_json(data_path)
    print(f"{len(questions)} soru yuklendi")
    print(f"Prompt language: {detect_benchmark_language(questions)} (BENCHMARK_LANGUAGE overrides)")
    
    # Kategorileri cikar
    categories = {q.get('category', 'genel_bilgi') for q in questions}
    print(f"Kategoriler: {list(categories)}")
    
    # Modeli yukle
    try:
        model = _load_model()
    except Exception as e:
        print(f"Model yuklenemedi: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    with_reasoning = getattr(model, 'reasoning_enabled', True)
    vision_enabled = getattr(model, 'vision_enabled', False)
    
    # Sorulari isle
    predictions = []
    vision_skipped = 0
    
    for i, question in enumerate(questions):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Soru isleniyor: {i+1}/{len(questions)}")
        
        question_id = question.get('id', i)
        category = question.get('category', 'genel_bilgi')
        image_path = question.get('image')
        prompt = build_prompt(question, with_reasoning=with_reasoning)
        
        if image_path:
            response, skipped = _process_image_question(model, prompt, image_path, vision_enabled)
            if skipped:
                vision_skipped += 1
                predictions.append({
                    "id": question_id, "prediction": "", "predicted_answer": "",
                    "reasoning": "", "category": category, "confidence": 0.0
                })
                continue
        else:
            response = call_llm_text(model, prompt)
        
        prediction = _create_prediction(question_id, category, response, with_reasoning, question)
        # Trajectory capture for distillation: the exact prompt the model saw and
        # its full raw answer (reasoning + final answer). The scorer copies them
        # into finetune_artifact.zip, so a strong model's run becomes a training
        # set in the very format this program parses at evaluation time.
        prediction["prompt"] = prompt
        prediction["raw_response"] = response or ""
        # A service error (401/429/5xx, timeout) is not an empty model answer:
        # the universal model.py leaves the reason in last_error, keep it.
        error = getattr(model, "last_error", None)
        if error and not response:
            prediction["error"] = error
        predictions.append(prediction)
        time.sleep(0.3)
    
    execution_time = time.time() - start_time
    _save_results(predictions, categories, execution_time, vision_enabled, with_reasoning, vision_skipped)


if __name__ == "__main__":
    main()
