#!/usr/bin/env python3
"""
Çoktan Seçmeli Kıyaslama için Gelişmiş Skorlama Programı (LLM-as-Judge Destekli)

Bu skorlama programı, dinamik kategorileri işler ve hem geleneksel doğruluk
metriklerini hem de LLM-as-Judge ile değerlendirilen gerekçe kalitesi metriklerini
hesaplar.

Desteklenen Metrikler:
1. Geleneksel Metrikler:
   - Accuracy (Doğruluk)
   - Category-wise Accuracy (Kategori Bazlı Doğruluk)

2. LLM-as-Judge Metrikleri (reasoning gerektiren):
   - Reasoning Quality (Gerekçelendirme Kalitesi)
   - Knowledge Depth (Bilgi Derinliği)
   - Logical Coherence (Mantıksal Tutarlılık)
   - Answer-Reasoning Alignment (Cevap-Gerekçe Uyumu)
   - Category-Specific Evaluation (Kategoriye Özel Değerlendirme)
"""
# --- LiteLLM attribution shim (auto-injected by the platform at bundle upload) ---
# Adds the `x-litellm-tags` header to outgoing HTTP requests so gateway spend
# is attributed to this run (benchmark:/run:/owner:, rl_loop:/round: tags).
# No-op when LITELLM_TAGS is absent from the environment.
def _install_litellm_attribution_shim():
    import os as _os
    _tags = (_os.environ.get("LITELLM_TAGS") or "").strip()
    if not _tags:
        return
    try:
        import urllib.request as _ur
        if not getattr(_ur, "_litellm_attribution_shim", False):
            _ur._litellm_attribution_shim = True
            _orig_urlopen = _ur.urlopen
            def _tagged_urlopen(url, *args, **kwargs):
                try:
                    if isinstance(url, str):
                        url = _ur.Request(url)
                    if isinstance(url, _ur.Request):
                        url.add_header("x-litellm-tags", _tags)
                except Exception:
                    pass
                return _orig_urlopen(url, *args, **kwargs)
            _ur.urlopen = _tagged_urlopen
    except Exception:
        pass
    try:
        import requests as _rq
        if not getattr(_rq.Session, "_litellm_attribution_shim", False):
            _rq.Session._litellm_attribution_shim = True
            _orig_send = _rq.Session.send
            def _tagged_send(self, request, **kwargs):
                try:
                    request.headers.setdefault("x-litellm-tags", _tags)
                except Exception:
                    pass
                return _orig_send(self, request, **kwargs)
            _rq.Session.send = _tagged_send
    except Exception:
        pass
    try:
        import httpx as _hx
        for _client_cls in (_hx.Client, _hx.AsyncClient):
            if getattr(_client_cls, "_litellm_attribution_shim", False):
                continue
            _client_cls._litellm_attribution_shim = True
            def _make_send(_orig):
                def _tagged(self, request, *args, **kwargs):
                    try:
                        request.headers.setdefault("x-litellm-tags", _tags)
                    except Exception:
                        pass
                    return _orig(self, request, *args, **kwargs)
                return _tagged
            _client_cls.send = _make_send(_client_cls.send)
    except Exception:
        pass
_install_litellm_attribution_shim()
del _install_litellm_attribution_shim
# --- end LiteLLM attribution shim ---

import json
import os
import sys
from collections import defaultdict
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple

# ============================================================================
# REDIRECT STDERR TO STDOUT - All logs will appear in stdout
# ============================================================================
# This ensures all output (warnings, errors, library logs) go to stdout
# so they appear in the "stdout" tab instead of stderr
sys.stderr = sys.stdout

# CodaBench için standart dizin yolları
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
REF_DIR = os.path.join(INPUT_DIR, "ref")
RES_DIR = os.path.join(INPUT_DIR, "res")
PROGRAM_DIR = "/app/program"

# File name constants
SCORES_FILENAME = "scores.json"
FINETUNE_ARTIFACT_FILENAME = "finetune_artifact.zip"


def _q_text(q):
    """Best-effort question text from a testing_data item."""
    for k in ("question", "text", "stem", "prompt", "input", "query"):
        v = q.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _render_options(q):
    """Render MCQ choices as 'A) ...' lines, or '' when there are none."""
    ch = q.get("choices") or q.get("options")
    if not ch:
        return ""
    # ARC shape: {"label": [...], "text": [...]}
    if isinstance(ch, dict) and ch.get("label") and ch.get("text"):
        return "\n".join(f"{l}) {t}" for l, t in zip(ch["label"], ch["text"]))
    if isinstance(ch, list):
        out = []
        for i, c in enumerate(ch):
            if isinstance(c, dict):
                lab = c.get("label") or chr(65 + i)
                txt = c.get("text") or c.get("value") or ""
                out.append(f"{lab}) {txt}")
            else:
                out.append(f"{chr(65 + i)}) {c}")
        return "\n".join(out)
    return ""


def _correct_text(q, label):
    """Resolve a correct-answer label (e.g. 'C') to its option text; fall back to the label."""
    ch = q.get("choices") or q.get("options")
    lab = str(label).strip()
    if isinstance(ch, dict) and ch.get("label") and ch.get("text"):
        for l, t in zip(ch["label"], ch["text"]):
            if str(l).strip().upper() == lab.upper():
                return t
    if isinstance(ch, list):
        for i, c in enumerate(ch):
            if isinstance(c, dict):
                clab = c.get("label") or chr(65 + i)
                if str(clab).strip().upper() == lab.upper():
                    return c.get("text") or c.get("value") or lab
            else:
                if chr(65 + i).upper() == lab.upper():
                    return str(c)
    return lab


def write_finetune_artifact(detailed_results, questions, predictions=None):
    """Build finetune_artifact.zip from the scored items.

    Two datasets in one artifact:
      failures.jsonl  the items the model got wrong (prompt = question + options,
                      response = the correct option's full text). Default input
                      of from-run training: it teaches what this model missed.
      all.jsonl       every scored item, tagged is_correct and carrying the
                      model's raw answer as model_response. Input for
                      distillation-style training from a strong model's run
                      (dataset_mode=all), where failures alone would be near empty.
    Written whenever at least one item was scored, so a run with zero failures
    still yields an artifact. Best-effort and never fatal: the caller wraps it
    in try/except so a failure here can never affect scoring.

    Trajectories: when the ingestion program recorded them, every all.jsonl row
    also carries teacher_prompt (the exact prompt the model saw) and
    model_response becomes the model's FULL answer (reasoning + final answer)
    instead of the bare letter. dataset_mode=distill trains a student on the
    rows the benchmarked (strong) model got right.
    """
    import zipfile
    by_id = {str(q.get("id")): q for q in (questions or [])}
    pred_by_id = {str(p.get("id")): p for p in (predictions or []) if isinstance(p, dict)}
    all_rows, failures = [], []
    service_errors = 0
    for dr in (detailed_results or []):
        q = by_id.get(str(dr.get("id")), {})
        qtext = _q_text(q)
        if not qtext:
            continue
        opts = _render_options(q)
        prompt = f"{qtext}\n\n{opts}" if opts else qtext
        response = str(_correct_text(q, dr.get("correct", ""))).strip()
        if not response:
            continue
        correct = bool(dr.get("is_correct"))
        pred = pred_by_id.get(str(dr.get("id")), {})
        predicted = str(dr.get("predicted", "") or "")
        trajectory = str(pred.get("raw_response") or "").strip()
        if not trajectory and str(pred.get("reasoning") or "").strip():
            # Older ingestion: only the parsed pieces survive, rebuild the format.
            trajectory = f"REASONING: {str(pred.get('reasoning')).strip()}\nANSWER: {predicted}"
        error = pred.get("error") if isinstance(pred.get("error"), dict) else None
        row = {
            "prompt": prompt,
            "response": response,
            "is_correct": correct,
            "model_response": trajectory or predicted,
            "teacher_prompt": str(pred.get("prompt") or ""),
            "has_trajectory": bool(trajectory),
        }
        if error:
            # The model was never heard (service error, timeout): not a wrong
            # answer, so it neither trains as a failure nor counts as answered.
            row["error"] = error
            service_errors += 1
        all_rows.append(row)
        if not correct and not error:
            failures.append({"prompt": prompt, "response": response})
    if not all_rows:
        print("[FT_ARTIFACT] no scored examples to write; skipping artifact")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    artifact_path = os.path.join(OUTPUT_DIR, FINETUNE_ARTIFACT_FILENAME)
    files = {"all": "all.jsonl"}
    if failures:
        files["failures"] = "failures.jsonl"
    manifest = {
        "num_examples": len(failures),
        "num_all": len(all_rows),
        "num_service_errors": service_errors,
        "num_correct": len(all_rows) - len(failures),
        # rows usable by dataset_mode=distill: answered correctly WITH a trajectory
        "num_distill": sum(1 for r in all_rows if r["is_correct"] and r["has_trajectory"]),
        "format": "prompt_response_failures",
        "files": files,
        "source": "multiple_choice_scorer",
    }
    def _jsonl(rows):
        return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    with zipfile.ZipFile(artifact_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if failures:
            zf.writestr("failures.jsonl", _jsonl(failures))
        zf.writestr("all.jsonl", _jsonl(all_rows))
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[FT_ARTIFACT] wrote {len(failures)} failure / {len(all_rows)} total examples -> {artifact_path}")

# Translation constants
CATEGORY_OKUMA_ANLAMA = 'Okuma Anlama'

# LLM Judge modülünü import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROGRAM_DIR)

try:
    from llm_judge import MCLLMJudge, load_mc_llm_judge_config
    LLM_JUDGE_AVAILABLE = True
except ImportError:
    LLM_JUDGE_AVAILABLE = False
    print("Uyarı: LLM Judge modülü yüklenemedi. Sadece geleneksel metrikler kullanılacak.")


def load_json(path: str) -> Any:
    """JSON dosyasını yükle"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_file(base_dir: str, filenames: List[str]) -> Optional[str]:
    """Olası dosya isimlerinden ilk eşleşeni bul"""
    if isinstance(filenames, str):
        filenames = [filenames]
    
    for filename in filenames:
        direct_path = os.path.join(base_dir, filename)
        if os.path.exists(direct_path):
            return direct_path
        
        # Alt dizinlerde ara
        for root, dirs, files in os.walk(base_dir):
            if filename in files:
                return os.path.join(root, filename)
    return None


def load_runtime_llm_config() -> Optional[Dict[str, Any]]:
    """
    Load LLM Judge configuration from runtime config file written by compute worker.
    """
    def extract_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
        if 'llm_judge' in config:
            return config['llm_judge']
        return config
    
    # Check for runtime config in program directory
    runtime_config_path = os.path.join(PROGRAM_DIR, "llm_judge_config.json")
    
    if os.path.exists(runtime_config_path):
        try:
            with open(runtime_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            llm_config = extract_llm_config(config)
            print(f"[LLM Judge] Loaded runtime config from: {runtime_config_path}")
            print(f"[LLM Judge] Config: enabled={llm_config.get('enabled')}, model={llm_config.get('model_name')}")
            return llm_config
        except Exception as e:
            print(f"[LLM Judge] Failed to load runtime config: {e}")
    
    # Fallback to bundled config
    bundled_config_path = os.path.join(os.path.dirname(__file__), "llm_judge_config.json")
    if os.path.exists(bundled_config_path):
        try:
            with open(bundled_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            llm_config = extract_llm_config(config)
            print(f"[LLM Judge] Loaded bundled config from: {bundled_config_path}")
            return llm_config
        except Exception as e:
            print(f"[LLM Judge] Failed to load bundled config: {e}")
    
    return None


class ExtendedMCScorer:
    """
    Genişletilmiş Çoktan Seçmeli Skorlayıcı
    
    Geleneksel doğruluk metriklerine ek olarak LLM-as-Judge kullanarak
    gerekçe kalitesi metriklerini hesaplar.
    """
    
    def _log_llm_config(self, llm_config: dict) -> None:
        """Log LLM configuration details."""
        api_url = llm_config.get('api_url', '').strip()
        model_name = llm_config.get('model_name', '').strip()
        is_enabled = llm_config.get('enabled', False)
        is_configured = bool(api_url and model_name)
        print("[LLM Judge] Configuration check:")
        print(f"[LLM Judge]   - enabled: {is_enabled}")
        print(f"[LLM Judge]   - api_url: {api_url[:50] if api_url else 'NOT CONFIGURED'}")
        print(f"[LLM Judge]   - model_name: {model_name if model_name else 'NOT CONFIGURED'}")
        print(f"[LLM Judge]   - properly_configured: {is_configured}")

    def _init_llm_judge(self, llm_config: dict) -> Optional['MCLLMJudge']:
        """Initialize and return LLM judge if configuration is valid."""
        api_url = llm_config.get('api_url', '').strip()
        model_name = llm_config.get('model_name', '').strip()
        is_enabled = llm_config.get('enabled', False)
        is_configured = bool(api_url and model_name)
        self._log_llm_config(llm_config)
        if not is_enabled:
            print("[LLM Judge] Disabled in configuration")
            return None
        if not is_configured:
            print("[LLM Judge] ERROR: Not properly configured!")
            return None
        try:
            if not LLM_JUDGE_AVAILABLE:
                print("[LLM Judge] MCLLMJudge not available")
                return None
            judge = MCLLMJudge(llm_config)
            print("[LLM Judge] Successfully initialized")
            return judge
        except Exception as e:
            print(f"[LLM Judge] Initialization failed: {e}")
            return None

    def __init__(self):
        """Skorlayıcıyı başlat"""
        llm_config = load_runtime_llm_config()
        self.llm_judge = None
        if LLM_JUDGE_AVAILABLE and llm_config:
            self.llm_judge = self._init_llm_judge(llm_config)
    
    def load_reference_data(self) -> List[Dict[str, Any]]:
        """Referans verilerini yükle"""
        ref_path = find_file(REF_DIR, ["testing_label.json", "testing_data.json", "ground_truth.json"])
        if not ref_path:
            raise FileNotFoundError("Referans verisi bulunamadı")
        
        print(f"Referans verisi yükleniyor: {ref_path}")
        return load_json(ref_path)
    
    def load_questions_data(self) -> List[Dict[str, Any]]:
        """Soru verilerini yükle (seçenekler için)"""
        questions_path = find_file(REF_DIR, ["testing_data.json", "questions.json"])
        if questions_path:
            return load_json(questions_path)
        return []
    
    def load_predictions(self) -> tuple:
        """Tahminleri yükle"""
        pred_path = find_file(RES_DIR, ["predictions.json", "results.json", "answers.json"])
        if not pred_path:
            raise FileNotFoundError("Tahminler bulunamadı")
        
        print(f"Tahminler yükleniyor: {pred_path}")
        pred_data = load_json(pred_path)
        
        # Format normalize et
        if isinstance(pred_data, dict) and 'predictions' in pred_data:
            predictions = pred_data['predictions']
            metadata = pred_data.get('metadata', {})
        elif isinstance(pred_data, list):
            predictions = pred_data
            metadata = {}
        else:
            raise ValueError("Bilinmeyen tahmin formatı")
        
        return predictions, metadata
    
    def calculate_traditional_scores(
        self, 
        references: List[Dict[str, Any]], 
        predictions: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Geleneksel doğruluk skorlarını hesapla"""
        
        # Referans haritası oluştur
        ref_map = {}
        ref_categories = {}
        for r in references:
            rid = str(r.get("id"))
            answer = r.get("correct_answer") or r.get("answer") or r.get("label", "")
            ref_map[rid] = str(answer).strip().upper()
            ref_categories[rid] = r.get("category", "genel_bilgi")
        
        # Skorları hesapla
        category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        overall_correct = 0
        overall_total = 0
        
        detailed_results = []
        
        for p in predictions:
            pid = str(p.get("id"))
            pval = p.get("predicted_answer") or p.get("prediction") or ""
            pval = str(pval).strip().upper()
            
            if pid in ref_map:
                overall_total += 1
                category = ref_categories.get(pid, "genel_bilgi")
                category_stats[category]['total'] += 1
                
                is_correct = pval == ref_map[pid]
                if is_correct:
                    overall_correct += 1
                    category_stats[category]['correct'] += 1
                
                detailed_results.append({
                    'id': pid,
                    'category': category,
                    'predicted': pval,
                    'correct': ref_map[pid],
                    'is_correct': is_correct
                })
        
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
        
        scores = {
            "accuracy": overall_accuracy * 100,
            "correct": overall_correct,
            "total": overall_total,
            "category_scores": {}
        }
        
        for category, stats in category_stats.items():
            cat_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            scores["category_scores"][category] = {
                "accuracy": cat_accuracy * 100,
                "correct": stats['correct'],
                "total": stats['total']
            }
        
        return scores, detailed_results
    
    def calculate_llm_judge_scores(
        self,
        questions: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """LLM-as-Judge skorlarını hesapla"""
        
        if not self.llm_judge:
            print("[LLM Judge] Not available, skipping LLM evaluation")
            return {
                "enabled": False,
                "reason": "LLM Judge not configured or disabled"
            }
        
        # Gerekçeli tahminleri kontrol et
        reasoning_count = sum(1 for p in predictions if p.get('reasoning'))
        if reasoning_count == 0:
            print("[LLM Judge] No predictions with reasoning found")
            return {
                "enabled": True,
                "reason": "No predictions with reasoning",
                "reasoning_quality": 0.0,
                "knowledge_depth": 0.0,
                "logical_coherence": 0.0,
                "answer_reasoning_alignment": 0.0,
                "category_specific": 0.0,
                "overall_llm_judge_score": 0.0
            }
        
        print(f"[LLM Judge] Evaluating {reasoning_count} predictions with reasoning")
        
        # Verileri hizala (ID bazlı)
        ref_map = {str(r.get('id')): r for r in references}
        q_map = {str(q.get('id')): q for q in questions}
        
        aligned_data = []
        for p in predictions:
            pid = str(p.get('id'))
            if pid in ref_map:
                q = q_map.get(pid, {})
                aligned_data.append({
                    'question': {**q, 'id': pid},
                    'prediction': p,
                    'reference': ref_map[pid]
                })
        
        # LLM Judge ile değerlendir
        results = []
        for item in aligned_data:
            if item['prediction'].get('reasoning'):
                result = self.llm_judge.evaluate_single_prediction(
                    item['question'],
                    item['prediction'],
                    item['reference']
                )
                results.append(result)
        
        # Skorları aggregate et
        if results:
            aggregated = self.llm_judge.aggregate_scores(results)
            aggregated['enabled'] = True
            aggregated['evaluated_count'] = len(results)
            return aggregated
        else:
            return {
                "enabled": True,
                "reason": "No valid predictions to evaluate",
                "overall_llm_judge_score": 0.0
            }
    
    def calculate_combined_score(
        self,
        traditional_scores: Dict[str, Any],
        llm_judge_scores: Dict[str, Any]
    ) -> float:
        """Geleneksel ve LLM Judge skorlarını birleştir"""
        
        accuracy = traditional_scores.get('accuracy', 0)
        
        if llm_judge_scores.get('enabled') and llm_judge_scores.get('overall_llm_judge_score', 0) > 0:
            llm_score = llm_judge_scores.get('overall_llm_judge_score', 0)
            # 50% accuracy, 50% LLM Judge
            combined = (accuracy * 0.5) + (llm_score * 0.5)
        else:
            # LLM Judge yoksa sadece accuracy
            combined = accuracy
        
        return combined
    
    def score_submission(self) -> Dict[str, float]:
        """Ana skorlama işlevi"""
        
        print("=" * 60)
        print("Çoktan Seçmeli Kıyaslama Skorlama Programı")
        print("(LLM-as-Judge Destekli)")
        print("=" * 60)
        
        start_time = time.time()
        
        # Create initial detailed results file immediately
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.create_initial_report()
        
        try:
            # Verileri yükle
            references = self.load_reference_data()
            questions = self.load_questions_data()
            predictions, metadata = self.load_predictions()
            
            print(f"{len(references)} referans cevabı yüklendi")
            print(f"{len(predictions)} tahmin yüklendi")
            
            # Reasoning istatistikleri
            reasoning_stats = metadata.get('reasoning_stats', {})
            if reasoning_stats:
                print(f"Gerekçeli tahmin: {reasoning_stats.get('total_with_reasoning', 0)}")
                print(f"Gerekçe oranı: {reasoning_stats.get('percentage_with_reasoning', 0):.1f}%")
            
            # Geleneksel skorları hesapla
            traditional_scores, detailed_results = self.calculate_traditional_scores(
                references, predictions
            )
            
            print(f"\n=== GELENEKSEL SKORLAR ===")
            print(f"Doğruluk: {traditional_scores['accuracy']:.2f}%")
            print(f"Doğru: {traditional_scores['correct']}/{traditional_scores['total']}")
            
            # LLM Judge skorlarını hesapla
            llm_judge_scores = self.calculate_llm_judge_scores(
                questions if questions else references,
                predictions,
                references
            )
            
            if llm_judge_scores.get('enabled') and llm_judge_scores.get('overall_llm_judge_score', 0) > 0:
                print(f"\n=== LLM-AS-JUDGE SKORLARI ===")
                print(f"Anlamsal Tutarlılık: {llm_judge_scores.get('semantic_coherence', 0):.2f}")
                print(f"Akıcılık: {llm_judge_scores.get('fluency', 0):.2f}")
                print(f"Cevap Gerekçesi: {llm_judge_scores.get('response_justification', 0):.2f}")
                print(f"Düşünce Zinciri: {llm_judge_scores.get('chain_of_thought_coherence', 0):.2f}")
                print(f"Olgusal Tutarlılık: {llm_judge_scores.get('factual_consistency', 0):.2f}")
                print(f"Halüsinasyon: {llm_judge_scores.get('hallucination_index', 0):.2f}")
                print(f"Genel LLM Judge Skoru: {llm_judge_scores.get('overall_llm_judge_score', 0):.2f}")
            
            # Birleşik skor hesapla
            combined_score = self.calculate_combined_score(traditional_scores, llm_judge_scores)
            
            execution_time = time.time() - start_time
            
            print(f"\n=== SONUÇ ===")
            print(f"Genel Skor: {combined_score:.2f}")
            print(f"İşlem Süresi: {execution_time:.2f} saniye")
            
            # Çıktı skorlarını hazırla
            scores = {
                "overall_score": combined_score,
                "accuracy": traditional_scores['accuracy'],
                "correct": traditional_scores['correct'],
                "total": traditional_scores['total'],
                "duration": execution_time
            }
            
            # Kategori skorlarını ekle
            for category, cat_scores in traditional_scores.get('category_scores', {}).items():
                scores[f"{category}_accuracy"] = cat_scores['accuracy']
            
            # LLM Judge skorlarını ekle (leaderboard column key'leri ile eşleşmeli)
            if llm_judge_scores.get('enabled'):
                # Unified metrics - leaderboard column key'leri ile eşleşir
                scores["semantic_coherence"] = llm_judge_scores.get('semantic_coherence', 0)
                scores["fluency"] = llm_judge_scores.get('fluency', 0)
                scores["response_justification"] = llm_judge_scores.get('response_justification', 0)
                scores["chain_of_thought_coherence"] = llm_judge_scores.get('chain_of_thought_coherence', 0)
                scores["factual_consistency"] = llm_judge_scores.get('factual_consistency', 0)
                scores["hallucination_index"] = llm_judge_scores.get('hallucination_index', 0)
                scores["llm_judge_score"] = llm_judge_scores.get('overall_llm_judge_score', 0)
            
            # Skorları kaydet
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            scores_path = os.path.join(OUTPUT_DIR, SCORES_FILENAME)
            with open(scores_path, "w", encoding="utf-8") as f:
                json.dump(scores, f, indent=2)

            # Emit the fine-tune dataset (failures) so from-run training works
            # for this benchmark. Best-effort: never let it break scoring.
            try:
                write_finetune_artifact(detailed_results, questions, predictions)
            except Exception as _ft_e:
                print(f"[FT_ARTIFACT] skipped ({type(_ft_e).__name__}: {_ft_e})")
            
            # Detaylı HTML rapor oluştur
            self.generate_detailed_report(
                traditional_scores, llm_judge_scores, combined_score, 
                detailed_results, execution_time
            )
            
            return scores
            
        except Exception as e:
            print(f"Skorlama hatası: {e}")
            traceback.print_exc()
            
            # Varsayılan skorlar
            scores = {"overall_score": 0, "accuracy": 0, "duration": time.time() - start_time}
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(OUTPUT_DIR, SCORES_FILENAME), "w") as f:
                json.dump(scores, f)
            
            return scores
    
    def create_initial_report(self):
        """Create an initial detailed results file to prevent timeout warnings"""
        html_content = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>Değerlendirme Devam Ediyor</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .loading { text-align: center; padding: 50px; background: white; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        h1 { color: #667eea; }
        p { color: #666; font-size: 18px; }
    </style>
</head>
<body>
    <div class="loading">
        <div class="spinner"></div>
        <h1>Değerlendirme Devam Ediyor...</h1>
        <p>Skorlar hesaplanıyor, lütfen bekleyin.</p>
        <p style="font-size: 14px; margin-top: 30px; color: #999;">Bu sayfa otomatik olarak güncellenecektir.</p>
    </div>
</body>
</html>
"""
        with open(os.path.join(OUTPUT_DIR, "detailed_results.html"), "w", encoding="utf-8") as f:
            f.write(html_content)
        print("[REPORT] Initial detailed results file created")
    
    @staticmethod
    def _get_progress_color(accuracy: float) -> str:
        """Return progress bar color based on accuracy percentage."""
        if accuracy >= 70:
            return "#4caf50"
        if accuracy >= 50:
            return "#ff9800"
        return "#f44336"
    
    def generate_detailed_report(
        self,
        traditional_scores: Dict[str, Any],
        llm_judge_scores: Dict[str, Any],
        combined_score: float,
        _detailed_results: List[Dict[str, Any]],
        execution_time: float
    ):
        """Detaylı HTML rapor oluştur"""
        
        # Kategori çevirileri
        category_translations = {
            'general': 'Genel',
            'general_knowledge': 'Genel Bilgi',
            'genel_bilgi': 'Genel Bilgi',
            'grammar': 'Dilbilgisi',
            'dilbilgisi': 'Dilbilgisi',
            'turkish_grammar': 'Türkçe Dilbilgisi',
            'spelling': 'Yazım',
            'yazim': 'Yazım',
            'spelling_errors': 'Yazım Hataları',
            'yazim_hatalari': 'Yazım Hataları',
            'vocabulary': 'Kelime Bilgisi',
            'kelime_bilgisi': 'Kelime Bilgisi',
            'reading': CATEGORY_OKUMA_ANLAMA,
            'okuma_anlama': CATEGORY_OKUMA_ANLAMA,
            'reading_comprehension': CATEGORY_OKUMA_ANLAMA,
            'math': 'Matematik',
            'matematik': 'Matematik',
            'mathematics': 'Matematik',
            'science': 'Fen Bilgisi',
            'fen_bilgisi': 'Fen Bilgisi',
            'history': 'Tarih',
            'tarih': 'Tarih',
            'geography': 'Coğrafya',
            'cografya': 'Coğrafya',
            'literature': 'Edebiyat',
            'edebiyat': 'Edebiyat',
            'coding': 'Programlama',
            'programlama': 'Programlama',
        }
        
        llm_enabled = llm_judge_scores.get('enabled', False) and llm_judge_scores.get('overall_llm_judge_score', 0) > 0
        
        # LLM Judge metrikleri HTML
        llm_section = ""
        if llm_enabled:
            llm_section = f"""
            <div class="section">
                <h2>LLM-as-Judge Değerlendirmesi</h2>
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value">{llm_judge_scores.get('semantic_coherence', 0):.1f}</div>
                        <div class="stat-label">Anlamsal Tutarlılık</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{llm_judge_scores.get('fluency', 0):.1f}</div>
                        <div class="stat-label">Akıcılık</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{llm_judge_scores.get('response_justification', 0):.1f}</div>
                        <div class="stat-label">Cevap Gerekçesi</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{llm_judge_scores.get('chain_of_thought_coherence', 0):.1f}</div>
                        <div class="stat-label">Düşünce Zinciri</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{llm_judge_scores.get('factual_consistency', 0):.1f}</div>
                        <div class="stat-label">Olgusal Tutarlılık</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{llm_judge_scores.get('hallucination_index', 0):.1f}</div>
                        <div class="stat-label">Halüsinasyon</div>
                    </div>
                </div>
                <div class="highlight">
                    <strong>Genel LLM Judge Skoru:</strong> {llm_judge_scores.get('overall_llm_judge_score', 0):.2f}
                </div>
            </div>
            """
        else:
            llm_section = """
            <div class="section warning">
                <h2>LLM-as-Judge Değerlendirmesi</h2>
                <p>LLM-as-Judge değerlendirmesi yapılamadı. Olası nedenler:</p>
                <ul>
                    <li>Model gerekçe (reasoning) sağlamadı</li>
                    <li>LLM Judge yapılandırılmamış</li>
                    <li>LLM Judge devre dışı</li>
                </ul>
                <p><strong>İpucu:</strong> LLM-as-Judge skorları için modelinizin <code>predict_with_reasoning()</code> 
                metodunu uygulaması ve her tahmin için gerekçe döndürmesi gerekir.</p>
            </div>
            """
        
        # Kategori tablosu
        category_rows = ""
        for category, cat_scores in traditional_scores.get('category_scores', {}).items():
            cat_name = category_translations.get(category.lower(), category.replace('_', ' ').title())
            cat_accuracy = cat_scores['accuracy']
            progress_color = self._get_progress_color(cat_accuracy)
            category_rows += f"""
            <tr>
                <td><strong>{cat_name}</strong></td>
                <td>{cat_accuracy:.2f}%</td>
                <td>{cat_scores['correct']}</td>
                <td>{cat_scores['total']}</td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {cat_accuracy}%; background: {progress_color};">
                            {cat_accuracy:.1f}%
                        </div>
                    </div>
                </td>
            </tr>
            """
        
        html_content = f"""
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>Kıyaslama Sonuçları</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin-bottom: 25px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }}
        .header h1 {{ margin: 0 0 15px 0; font-size: 28px; }}
        .score {{ font-size: 56px; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }}
        .subtitle {{ font-size: 18px; opacity: 0.9; margin-top: 10px; }}
        .section {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .section.warning {{ background: #fff3e0; border-left: 4px solid #ff9800; }}
        h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; margin-top: 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .progress-bar {{ background: #e9ecef; border-radius: 10px; overflow: hidden; height: 24px; }}
        .progress-fill {{ height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px; transition: width 0.5s ease; }}
        .stats {{ display: flex; gap: 20px; margin-top: 20px; flex-wrap: wrap; }}
        .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 10px; flex: 1; min-width: 150px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
        .stat-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .highlight {{ background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%); padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center; }}
        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Çoktan Seçmeli Kıyaslama Sonuçları</h1>
        <div class="score">{combined_score:.2f}%</div>
        <div class="subtitle">Genel Skor {"(Doğruluk + LLM Judge)" if llm_enabled else "(Sadece Doğruluk)"}</div>
    </div>
    
    <div class="section">
        <h2>Genel İstatistikler</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{traditional_scores['accuracy']:.1f}%</div>
                <div class="stat-label">Doğruluk</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{traditional_scores['correct']}</div>
                <div class="stat-label">Doğru Cevap</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{traditional_scores['total']}</div>
                <div class="stat-label">Toplam Soru</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{execution_time:.2f}s</div>
                <div class="stat-label">İşlem Süresi</div>
            </div>
        </div>
    </div>
    
    {llm_section}
    
    <div class="section">
        <h2>Kategori Bazında Sonuçlar</h2>
        <table>
            <tr>
                <th>Kategori</th>
                <th>Doğruluk</th>
                <th>Doğru</th>
                <th>Toplam</th>
                <th>İlerleme</th>
            </tr>
            {category_rows}
        </table>
    </div>
    
    <div style="text-align: center; margin-top: 20px; color: #666; font-size: 14px;">
        Bu sonuçlar otomatik olarak oluşturulmuştur. • LLM-as-Judge Destekli Değerlendirme
    </div>
</body>
</html>
"""
        
        with open(os.path.join(OUTPUT_DIR, "detailed_results.html"), "w", encoding="utf-8") as f:
            f.write(html_content)


def write_default_scores():
    """Hata durumunda varsayılan sıfır skorları yaz"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scores = {"overall_score": 0, "accuracy": 0}
    with open(os.path.join(OUTPUT_DIR, SCORES_FILENAME), "w") as f:
        json.dump(scores, f)


def main():
    """Ana fonksiyon"""
    scorer = ExtendedMCScorer()
    scorer.score_submission()


if __name__ == "__main__":
    main()
