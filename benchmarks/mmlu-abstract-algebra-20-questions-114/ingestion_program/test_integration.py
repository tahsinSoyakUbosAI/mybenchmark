#!/usr/bin/env python3
"""
Türkçe Dil Kıyaslaması için entegrasyon testi
Veri yüklemesinden tahminen kadar tam pipeline'ı test eder
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
import sys
import os
from pathlib import Path

# Add solution directory to path
sys.path.append(str(Path(__file__).parent / "solution"))
sys.path.append(str(Path(__file__).parent / "ingestion_program"))

def test_complete_pipeline():
    """Tam pipeline'ı test et"""
    print("=" * 60)
    print("TÜRKÇE DİL KIYASLAMASI - ENTEGRASYON TESTİ")
    print("=" * 60)
    
    # Test 1: Import and initialize model
    print("\n1. Model import ve başlatması test ediliyor...")
    try:
        from model import TurkishLanguageModel, TurkishGrammarModel, Model
        model = TurkishLanguageModel()
        print("✓ Model başarıyla import edildi ve başlatıldı")
    except Exception as e:
        print(f"✗ Model import başarısız: {e}")
        return False
    
    # Test 2: Load test data
    print("\n2. Veri yükleme test ediliyor...")
    try:
        test_data_path = Path(__file__).parent / "public_data" / "testing_data.json"
        if test_data_path.exists():
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            print(f"✓ {len(test_data)} test sorusu yüklendi")
        else:
            # Create sample data
            test_data = [
                {
                    "id": 1,
                    "question": "Aşağıdaki cümlelerden hangisinde yazım hatası vardır?",
                    "choices": {
                        "label": ["A", "B", "C", "D"],
                        "text": ["Kitap okuyorum", "Okule gidiyorum", "Eve dönüyorum", "Ders çalışıyorum"]
                    },
                    "category": "spelling_errors",
                    "correct_answer": "B"
                }
            ]
            print("✓ Örnek test verisi oluşturuldu")
    except Exception as e:
        print(f"✗ Veri yükleme başarısız: {e}")
        return False
    
    # Test 3: Model training
    print("\n3. Model eğitimi test ediliyor...")
    try:
        model.fit(test_data)
        print("✓ Model eğitimi tamamlandı")
    except Exception as e:
        print(f"✗ Model eğitimi başarısız: {e}")
        return False
    
    # Test 4: Single prediction
    print("\n4. Tekli tahmin test ediliyor...")
    try:
        prediction = model.predict(test_data[0])
        print(f"✓ Tekli tahmin: {prediction}")
        
        # Validate prediction format
        required_fields = ['id', 'predicted_answer', 'category', 'confidence']
        for field in required_fields:
            if field not in prediction:
                print(f"✗ Tahminde eksik alan: {field}")
                return False
        print("✓ Tahmin formatı doğrulandı")
        
    except Exception as e:
        print(f"✗ Tekli tahmin başarısız: {e}")
        return False
    
    # Test 5: Batch prediction
    print("\n5. Toplu tahmin test ediliyor...")
    try:
        predictions = model.predict_batch(test_data)
        print(f"✓ Toplu tahmin: {len(predictions)} tahmin")
        
        # Validate all predictions
        for i, pred in enumerate(predictions):
            for field in required_fields:
                if field not in pred:
                    print(f"✗ {i} numaralı tahminde eksik alan: {field}")
                    return False
        print("✓ Tüm tahminler doğrulandı")
        
    except Exception as e:
        print(f"✗ Toplu tahmin başarısız: {e}")
        return False
    
    # Test 6: Multi-category interface
    print("\n6. Çok-kategorili arayüz test ediliyor...")
    try:
        data_dict = {'testing': test_data, 'training': test_data}
        predictions = model.predict_multi_category(data_dict)
        print(f"✓ Çok-kategorili tahmin: {len(predictions)} tahmin")
    except Exception as e:
        print(f"✗ Çok-kategorili tahmin başarısız: {e}")
        return False
    
    # Test 7: Backward compatibility
    print("\n7. Geriye dönük uyumluluk test ediliyor...")
    try:
        grammar_model = TurkishGrammarModel()
        grammar_model.fit(test_data)
        
        # Test legacy string input
        legacy_result = grammar_model.predict("Bu cümle doğru mu?")
        print(f"✓ Eski format tahmini: {legacy_result}")
        
        # Test new format
        new_result = grammar_model.predict(test_data[0])
        print(f"✓ Yeni format tahmini: {new_result}")
        
    except Exception as e:
        print(f"✗ Geriye dönük uyumluluk testi başarısız: {e}")
        return False
    
    # Test 8: Test ingestion compatibility
    print("\n8. Alma sistemi uyumluluğu test ediliyor...")
    try:
        from ingestion import TurkishLanguageIngestion
        ingestion = TurkishLanguageIngestion()
        print("✓ Alma programı başarıyla import edildi")
        
        # Test if ingestion can load the model
        test_model = ingestion._load_submission_model("solution")
        print("✓ Model alma sistemi aracılığıyla yüklendi")
        
    except Exception as e:
        print(f"⚠ Alma testi atlandı (mevcut olmayabilir): {e}")
    
    print("\n" + "=" * 60)
    print("🎉 TÜM TESTLER BAŞARILI! Türkçe Dil Kıyaslaması hazır!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_complete_pipeline()
    sys.exit(0 if success else 1)
