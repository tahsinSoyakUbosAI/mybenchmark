#!/usr/bin/env python3
"""
LLM-as-Judge Module for Multiple Choice Benchmark Scoring
(Uses Unified LLM Judge System)

Bu modül, çoktan seçmeli benchmark skorlamasında ortak LLM-as-Judge
metriklerini kullanır.

Ortak Metrikler (Her İki Benchmark İçin):
=========================================
Quality: semantic_coherence, fluency, response_justification
Reasoning: chain_of_thought_coherence, answer_traceability, depth_of_understanding  
Factuality: factual_consistency, hallucination_index, knowledge_grounding
Coverage: coverage_score
Trust: trust_score
Safety: unbiased_response
Cultural: cultural_context_preservation
"""

import json
import os
import re
import time
import requests
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import traceback

# SSL Error Message Constants
MSG_SSL_OUTDATED = "[LLM Judge] Docker container may have outdated SSL libraries."
MSG_SSL_SKIP_EVAL = "[LLM Judge] Skipping LLM evaluation - traditional metrics will still be calculated."

# Try to import unified LLM judge from common module
try:
    import sys
    # Add common directory to path
    common_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common')
    if common_dir not in sys.path:
        sys.path.insert(0, common_dir)
    from unified_llm_judge import UnifiedLLMJudge, get_common_metrics_config, load_unified_llm_judge_config
    UNIFIED_AVAILABLE = True
except ImportError:
    UNIFIED_AVAILABLE = False


class MCLLMJudge:
    """
    Multiple Choice LLM-as-Judge sınıfı
    
    Ortak birleştirilmiş LLM Judge sistemini kullanarak çoktan seçmeli
    sorularda model değerlendirmesi yapar.
    """
    
    # Common metrics from unified system
    COMMON_METRICS = [
        "semantic_coherence",
        "fluency", 
        "response_justification",
        "chain_of_thought_coherence",
        "answer_traceability",
        "depth_of_understanding",
        "factual_consistency",
        "hallucination_index",
        "knowledge_grounding",
        "coverage_score",
        "trust_score",
        "unbiased_response",
        "cultural_context_preservation"
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        LLM Judge'u yapılandırma ile başlat
        
        Args:
            config: LLM API yapılandırması. None ise varsayılan kullanılır.
        """
        self.default_config = {
            "api_url": "",
            "api_key": "",
            "model_name": "",
            "timeout": 60,
            "max_retries": 3,
            "batch_size": 10,
            "temperature": 0.1,
            "max_tokens": 500,
            "enabled": False
        }
        
        self.config = {**self.default_config, **(config or {})}
        self.metrics_cache = {}
        
        # Initialize unified LLM judge if available
        self.unified_judge = None
        if UNIFIED_AVAILABLE and self.is_configured():
            try:
                self.unified_judge = UnifiedLLMJudge(self.config)
                print("[MC LLM Judge] Using unified LLM judge system")
            except Exception as e:
                print(f"[MC LLM Judge] Failed to initialize unified judge: {e}")
        
    def is_configured(self) -> bool:
        """Check if LLM Judge is properly configured"""
        api_url = self.config.get('api_url', '').strip()
        model_name = self.config.get('model_name', '').strip()
        enabled = self.config.get('enabled', False)
        return bool(enabled and api_url and model_name)

    def _is_ssl_error(self, error: Exception) -> bool:
        """Check if an exception is SSL-related."""
        error_str = str(error).lower()
        ssl_keywords = ('ssl', 'openssl', 'certificate', 'wantreaderror')
        return any(keyword in error_str for keyword in ssl_keywords)

    def _log_ssl_error(self, error: Exception) -> None:
        """Log SSL error and provide guidance."""
        print(f"[LLM Judge] SSL-related error: {error}")
        print(MSG_SSL_OUTDATED)
        print(MSG_SSL_SKIP_EVAL)

    def _extract_response_content(self, result: dict) -> Optional[str]:
        """Extract content from LLM API response."""
        if "choices" not in result or not result["choices"]:
            return None
        content = result["choices"][0].get("message", {}).get("content")
        return content.strip() if content else None

    def _handle_request_error(self, error: Exception, attempt: int) -> Tuple[bool, Optional[str]]:
        """Handle request errors. Returns (should_retry, result)."""
        if isinstance(error, requests.exceptions.SSLError):
            self._log_ssl_error(error)
            return False, None
        if isinstance(error, requests.exceptions.Timeout):
            print(f"[LLM Judge] API timeout (attempt {attempt + 1})")
            time.sleep(2 ** attempt)
            return True, None
        if isinstance(error, requests.exceptions.RequestException):
            if self._is_ssl_error(error):
                self._log_ssl_error(error)
                return False, None
            print(f"[LLM Judge] API error (attempt {attempt + 1}): {error}")
            time.sleep(2 ** attempt)
            return True, None
        # Generic exception
        if self._is_ssl_error(error):
            self._log_ssl_error(error)
            return False, None
        print(f"[LLM Judge] Unexpected error: {error}")
        return False, None
        
    def _call_llm(self, prompt: str, max_tokens: Optional[int] = None) -> Optional[str]:
        """LLM API'sini çağır"""
        if not self.is_configured():
            return None
            
        headers = {
            "Authorization": f"Bearer {self.config.get('api_key', '')}",
            "Content-Type": "application/json"
        }
        # Platform-injected spend attribution for LiteLLM (benchmark/run tags).
        _litellm_tags = os.environ.get("LITELLM_TAGS", "").strip()
        if _litellm_tags:
            headers["x-litellm-tags"] = _litellm_tags
        payload = {
            "model": self.config["model_name"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.get("temperature", 0.1),
            "max_tokens": max_tokens or self.config.get("max_tokens", 500)
        }
        
        for attempt in range(self.config.get("max_retries", 3)):
            try:
                response = requests.post(
                    self.config["api_url"],
                    headers=headers,
                    json=payload,
                    timeout=self.config.get("timeout", 60),
                    verify=True
                )
                response.raise_for_status()
                return self._extract_response_content(response.json())
            except Exception as e:
                should_retry, result = self._handle_request_error(e, attempt)
                if not should_retry:
                    return result
        return None
    
    def _parse_score(self, response: str, default: float = 0.0) -> float:
        """LLM yanıtından skor çıkar"""
        if not response:
            return default
            
        patterns = [
            r'(?:skor|score|puan)[:\s]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:/\s*100|%)',
            r'(?:^|\s)(\d+(?:\.\d+)?)(?:\s|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                score = float(match.group(1))
                if score <= 1.0 and score > 0:
                    score *= 100
                return min(100, max(0, score))
        
        return default
    
    def evaluate_single_prediction(
        self,
        question: Dict[str, Any],
        prediction: Dict[str, Any],
        reference: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Tek bir tahmin için tüm ortak metrikleri değerlendir
        
        Args:
            question: Soru bilgileri
            prediction: Model tahmini (answer + reasoning)
            reference: Referans cevap
            
        Returns:
            Tüm metriklerin skorları
        """
        question_text = question.get('question', '')
        choices = question.get('choices', {'label': ['A', 'B', 'C', 'D'], 'text': []})
        category = question.get('category', 'general_knowledge')
        
        predicted_answer = prediction.get('predicted_answer', prediction.get('prediction', 'A'))
        reasoning = prediction.get('reasoning', prediction.get('explanation', ''))
        
        correct_answer = reference.get('correct_answer', reference.get('answer', 'A'))
        
        results = {
            'question_id': question.get('id'),
            'category': category,
            'predicted_answer': predicted_answer,
            'correct_answer': correct_answer,
            'is_correct': str(predicted_answer).upper() == str(correct_answer).upper(),
            'has_reasoning': bool(reasoning and reasoning.strip()),
            'reasoning_length': len(reasoning) if reasoning else 0,
            'metrics': {}
        }
        
        # Gerekçe yoksa tüm skorları 0 yap
        if not reasoning or not reasoning.strip():
            for metric in self.COMMON_METRICS:
                results['metrics'][metric] = 0.0
            results['llm_judge_note'] = "Gerekçe sağlanmadığı için LLM-as-Judge skorları 0 olarak belirlendi."
            return results
        
        # Use unified judge if available
        if self.unified_judge:
            try:
                unified_results = self.unified_judge.evaluate_for_multiple_choice(
                    question=question_text,
                    choices=choices,
                    predicted_answer=predicted_answer,
                    reasoning=reasoning,
                    correct_answer=correct_answer,
                    category=category
                )
                results['metrics'] = unified_results
            except Exception as e:
                print(f"[MC LLM Judge] Error using unified judge: {e}")
                traceback.print_exc()
                # Fall back to individual evaluations
                results['metrics'] = self._evaluate_fallback(
                    question_text, choices, predicted_answer, reasoning, correct_answer, category
                )
        else:
            # Use fallback individual evaluations
            results['metrics'] = self._evaluate_fallback(
                question_text, choices, predicted_answer, reasoning, correct_answer, category
            )
        
        return results
    
    def _evaluate_fallback(
        self,
        question_text: str,
        choices: Dict[str, List[str]],
        predicted_answer: str,
        reasoning: str,
        correct_answer: str,
        category: str
    ) -> Dict[str, float]:
        """
        Fallback değerlendirme (unified judge yoksa)
        """
        metrics = {}
        
        # Format choices
        choices_text = ""
        labels = choices.get('label', ['A', 'B', 'C', 'D'])
        texts = choices.get('text', [])
        for label, text in zip(labels, texts):
            choices_text += f"{label}) {text}\n"
        
        try:
            # Semantic Coherence
            prompt = f"""Sen bir dil uzmanısın. Aşağıdaki gerekçenin anlamsal tutarlılığını değerlendir.

SORU: {question_text}
GEREKÇELENDİRME: {reasoning}

0-100 arasında skor ver. Format: SKOR: [skor]"""
            response = self._call_llm(prompt)
            metrics['semantic_coherence'] = self._parse_score(response, 50.0)
            
            # Fluency
            prompt = f"""Sen bir dil uzmanısın. Metnin akıcılığını değerlendir.

METİN: {reasoning}

0-100 arasında skor ver. Format: SKOR: [skor]"""
            response = self._call_llm(prompt)
            metrics['fluency'] = self._parse_score(response, 60.0)
            
            # Response Justification
            prompt = f"""Cevabın gerekçelendirilme kalitesini değerlendir.

SORU: {question_text}
SEÇENEKLER: {choices_text}
CEVAP: {predicted_answer}
GEREKÇELENDİRME: {reasoning}

0-100 arasında skor ver. Format: SKOR: [skor]"""
            response = self._call_llm(prompt)
            metrics['response_justification'] = self._parse_score(response, 50.0)
            
            # Chain of Thought Coherence
            prompt = f"""Düşünce zincirinin tutarlılığını değerlendir.

SORU: {question_text}
DÜŞÜNCE ADIMLARI: {reasoning}
CEVAP: {predicted_answer}
DOĞRU CEVAP: {correct_answer}

0-100 arasında skor ver. Format: SKOR: [skor]"""
            response = self._call_llm(prompt)
            metrics['chain_of_thought_coherence'] = self._parse_score(response, 50.0)
            
            # Answer Traceability
            prompt = f"""Cevabın gerekçeden izlenebilirliğini değerlendir.

SORU: {question_text}
GEREKÇELENDİRME: {reasoning}
CEVAP: {predicted_answer}

0-100 arasında skor ver. Format: SKOR: [skor]"""
            response = self._call_llm(prompt)
            metrics['answer_traceability'] = self._parse_score(response, 50.0)
            
            # Depth of Understanding
            prompt = f"""Anlama derinliğini değerlendir.

SORU: {question_text}
ALAN: {category}
CEVAP VE GEREKÇELENDİRME: {reasoning}

0-100 arasında skor ver. Format: SKOR: [skor]"""
            response = self._call_llm(prompt)
            metrics['depth_of_understanding'] = self._parse_score(response, 50.0)
            
            # Trust Score
            prompt = f"""Yanıtın genel güvenilirliğini değerlendir.

SORU: {question_text}
YANIT: {reasoning}

0-100 arasında skor ver. Format: SKOR: [skor]"""
            response = self._call_llm(prompt)
            metrics['trust_score'] = self._parse_score(response, 50.0)
            
            # Factual Consistency
            full_context = f"Soru: {question_text}\nSeçenekler: {choices_text}"
            prompt = f"""Olgusal tutarlılığı değerlendir.

BAĞLAM: {full_context}
YANIT: {reasoning}

0-100 arasında skor ver. Format: SKOR: [skor]"""
            response = self._call_llm(prompt)
            metrics['factual_consistency'] = self._parse_score(response, 50.0)
            
            # Knowledge Grounding
            prompt = f"""Bilgi temellendirmesini değerlendir.

SORU: {question_text}
YANIT: {reasoning}

0-100 arasında skor ver. Format: SKOR: [skor]"""
            response = self._call_llm(prompt)
            metrics['knowledge_grounding'] = self._parse_score(response, 50.0)
            
            # Hallucination Index (lower is better)
            prompt = f"""Halüsinasyon indeksini değerlendir (yüksek = çok halüsinasyon).

SORU: {question_text}
DOĞRU CEVAP: {correct_answer}
YANIT: {reasoning}

0-100 arasında skor ver. Format: SKOR: [skor]"""
            response = self._call_llm(prompt)
            metrics['hallucination_index'] = self._parse_score(response, 20.0)
            
            # Coverage Score
            metrics['coverage_score'] = 50.0  # Default for MC
            
            # Unbiased Response
            prompt = f"""Yanıtın tarafsızlığını değerlendir.

YANIT: {reasoning}

0-100 arasında skor ver (100 = tamamen tarafsız). Format: SKOR: [skor]"""
            response = self._call_llm(prompt)
            metrics['unbiased_response'] = self._parse_score(response, 80.0)
            
            # Cultural Context Preservation
            metrics['cultural_context_preservation'] = 70.0  # Default
            
        except Exception as e:
            print(f"[MC LLM Judge] Fallback evaluation error: {e}")
            for metric in self.COMMON_METRICS:
                if metric not in metrics:
                    metrics[metric] = 0.0
        
        return metrics
    
    def evaluate_batch(
        self,
        questions: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Toplu değerlendirme yap
        """
        results = []
        total = len(questions)
        
        print(f"[MC LLM Judge] Evaluating {total} predictions...")
        
        for i, (q, p, r) in enumerate(zip(questions, predictions, references)):
            print(f"[MC LLM Judge] Processing {i+1}/{total}...")
            result = self.evaluate_single_prediction(q, p, r)
            results.append(result)
            time.sleep(0.3)
        
        return results
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Return the default metric weights for scoring."""
        return {
            # Quality (25%)
            'semantic_coherence': 0.10,
            'fluency': 0.08,
            'response_justification': 0.07,
            # Reasoning (25%)
            'chain_of_thought_coherence': 0.10,
            'answer_traceability': 0.08,
            'depth_of_understanding': 0.07,
            # Factuality (25%)
            'factual_consistency': 0.10,
            'hallucination_index': 0.08,  # Reversed
            'knowledge_grounding': 0.07,
            # Coverage (10%)
            'coverage_score': 0.10,
            # Trust (10%)
            'trust_score': 0.10,
            # Safety (5%)
            'unbiased_response': 0.05
        }

    def _collect_scores_by_metric(self, results: List[Dict[str, Any]]) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
        """Collect scores grouped by metric and category."""
        metrics_aggregated = defaultdict(list)
        category_metrics = defaultdict(lambda: defaultdict(list))
        for result in results:
            metrics = result.get('metrics', {})
            category = result.get('category', 'unknown')
            for metric_name, score in metrics.items():
                if score is not None and isinstance(score, (int, float)):
                    metrics_aggregated[metric_name].append(score)
                    category_metrics[category][metric_name].append(score)
        return metrics_aggregated, category_metrics

    def _calculate_metric_averages(self, metrics_aggregated: Dict[str, List[float]], category_metrics: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Calculate average scores for each metric."""
        aggregated = {}
        for metric, scores in metrics_aggregated.items():
            aggregated[metric] = sum(scores) / len(scores) if scores else 0.0
        aggregated['category_scores'] = {}
        for category, metrics in category_metrics.items():
            aggregated['category_scores'][category] = {
                metric: sum(scores) / len(scores)
                for metric, scores in metrics.items() if scores
            }
        return aggregated

    def _calculate_weighted_score(self, aggregated: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate the overall weighted score."""
        weighted_sum = 0.0
        total_weight = 0.0
        for metric, weight in weights.items():
            if metric not in aggregated:
                continue
            score = aggregated[metric]
            if metric == 'hallucination_index':
                score = 100 - score
            weighted_sum += score * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def aggregate_scores(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Sonuçları topla ve ağırlıklı ortalama hesapla"""
        metrics_aggregated, category_metrics = self._collect_scores_by_metric(results)
        aggregated = self._calculate_metric_averages(metrics_aggregated, category_metrics)
        weights = self._get_default_weights()
        aggregated['overall_llm_judge_score'] = self._calculate_weighted_score(aggregated, weights)
        return aggregated


def load_mc_llm_judge_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    LLM Judge yapılandırmasını yükle
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            if 'llm_judge' in config:
                return config['llm_judge']
            return config
    
    return {
        "enabled": False,
        "api_url": "",
        "api_key": "",
        "model_name": "",
        "timeout": 60,
        "max_retries": 3,
        "batch_size": 10,
        "temperature": 0.1,
        "max_tokens": 500
    }


if __name__ == "__main__":
    print("Testing MC LLM Judge with Unified Metrics...")
    print("=" * 60)
    print(f"Unified system available: {UNIFIED_AVAILABLE}")
    print(f"\nCommon metrics: {MCLLMJudge.COMMON_METRICS}")
    print("\nMC LLM Judge module loaded successfully!")
