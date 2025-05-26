# DSPy ile Atıf Niyeti Sınıflandırma ve Optimizasyon Projesi

Bu proje, akademik metinlerdeki atıfların retorik niyetlerini (örneğin, 'background', 'basis', 'support' vb.) 
sınıflandırmak için DSPy kütüphanesini ve Büyük Dil Modellerini (LLM'ler - OpenAI GPT ve Google Gemini modelleri) 
kullanır. 

Proje, `MIPROv2` optimizer'ı aracılığıyla en iyi prompt'ları (talimatlar ve few-shot demolar) bulmak için 
bir optimizasyon süreci içerir.

## Temel Özellikler
* **Atıf Niyeti Sınıflandırması:** Akademik atıfları önceden tanımlanmış kategorilere ayırır.
* **DSPy Entegrasyonu:**
    * `dspy.Signature`: Görev tanımı ve LLM'e verilecek talimatlar.
    * `dspy.Module`: Sınıflandırma mantığını içeren modül (`ClassifyCitation`).
    * `dspy.Predict` (veya `dspy.ChainOfThought`): LLM ile etkileşim katmanı.
* **Prompt Optimizasyonu:** `dspy.teleprompt.MIPROv2` optimizer'ı kullanılarak en iyi performansı veren prompt (talimat ve demo) kombinasyonları aranır.
* **Model Desteği:** OpenAI GPT modelleri (`gpt-4o`, `gpt-4o-mini`) ve Google Gemini modelleri (`gemini-2.5-flash-preview-05-20`) ile çalışacak şekilde yapılandırılmıştır.
* **Veri Yönetimi:** `trainset.csv` ve `devset.csv` dosyalarından eğitim ve geliştirme verilerini yükler.
* **Kalıcılık:** Optimize edilmiş program kaydedilebilir (`.json`) ve daha sonra yeniden kullanılabilir.
* **Loglama:** `MIPROv2`'nin optimizasyon adımlarını detaylı loglama (`log_dir` aracılığıyla) ve genel script çıktılarını takip etme.

## Gereksinimler ve Kurulum

### Gereksinimler
* Python 3.9+
* Temel kütüphaneler:
    * `dspy-ai`
    * `pandas`
    * `openai`
    * `google-generativeai` (veya Gemini için DSPy'ın kullandığı ilgili Google kütüphanesi)

### Kurulum Adımları
1.  **Proje Klonlama (Eğer bir Git reposu ise):**
    ```bash
    git clone https://github.com/TOBB-University-Master/CitationClassifier_ICL_DSPy.git
    ```
2.  **Sanal Ortam Oluşturma (Conda):**
    ```bash
    conda create --name tobb_tez_dspy
    conda activate tobb_tez_dspy
    ```
3.  **Gerekli Kütüphaneleri Yükleme:**
    Projenizin kök dizininde bir `requirements.txt` dosyası oluşturulmuştur. 
    Bu dosyaAşağıdaki komut ile bu dosyayı kullanarak gerekli kütüphaneleri yükleyebilirsiniz:
    
    ```bash
    pip install -r requirements.txt
    ```
4.  **API Anahtarlarını Ayarlama (ÇOK ÖNEMLİ):**
    * OpenAI ve/veya Google Gemini API anahtarlarınızı doğrudan koda yazmak yerine **ortam değişkenleri (environment variables)** olarak ayarlamanız şiddetle tavsiye edilir.
    * Örneğin, terminalinizde (veya `.bashrc`, `.zshrc`, `.env` dosyalarınızda):
        ```bash
        export OPENAI_API_KEY="sk-..."
        export GOOGLE_API_KEY="AIza..."
        ```
    * Ardından Python kodunuzda `os.getenv("OPENAI_API_KEY")` ve `os.getenv("GOOGLE_API_KEY")` ile bu anahtarlara erişebilirsiniz. Mevcut kodunuzda bu değişikliği yapmanız önerilir.

## Veri Hazırlığı
Projenin çalışması için `trainset.csv` ve `devset.csv` dosyalarına ihtiyaç vardır. 
Bu dosyalar genellikle projenin içinde bir `data/` klasöründe tutulur.

* **Dosya Formatı:** CSV dosyaları aşağıdaki sütunları içermelidir:
    * `citation_id`: Her atıf için benzersiz bir tanımlayıcı (isteğe bağlı ama takip için faydalı).
    * `citation_context`: Sınıflandırılacak atıf metni.
    * `section`: Atıfın içinde bulunduğu makale bölümünün başlığı.
    * `citation_intent`: Atıfın doğru niyet etiketi (örneğin, `background`, `basis` vb.).
    * `reasoning` (İsteğe bağlı): Eğer `Signature`'ınızda `reasoning` alanı varsa ve eğitim verinizde bu bilgi mevcutsa, bu sütunu da ekleyebilirsiniz. (Mevcut durumda `reasoning` kaldırıldığı için bu sütuna gerek yoktur).

* **Örnek Yerleşim:**
    ```
    ProjeKlasoru/
    ├── data/
    │   ├── trainset.csv
    │   └── devset.csv
    ├── dspy_citation_classifier.py  (Ana betiğiniz)
    └── README.md
    ```

## Kullanım

Ana betiğiniz (örneğin, `dspy_citation_classifier.py`) üzerinden hem optimizasyon sürecini başlatabilir hem de optimize 
edilmiş bir modeli kullanarak çıkarım yapabilirsiniz.

1.  **Optimizasyon Sürecini Çalıştırma:**
    * Betiği çalıştırdığınızda, eğer `optimized_citation_classifier.json` dosyası bulunamazsa veya mevcut programı daha da optimize etmek üzere ayarlandıysa, `MIPROv2` optimizasyon sürecini başlatır.
    * Bu süreç, `trainset.csv` verisini demo seçimi için, `devset.csv` verisini ise aday prompt'ların performansını değerlendirmek için kullanır.
    * Optimizasyon tamamlandığında, en iyi bulunan program `optimized_citation_classifier.json` dosyasına kaydedilir.
    * Eğer `optimizer.compile()` içinde `log_dir="mipro_logs"` gibi bir parametre belirlerseniz, optimizasyonun detaylı logları bu klasöre kaydedilir.
    ```bash
    python dspy_citation_classifier.py
    ```

2.  **Optimize Edilmiş Model ile Çıkarım Yapma:**
    * Betiğiniz, başlangıçta `optimized_citation_classifier.json` dosyasını yüklemeye çalışır. Eğer dosya mevcutsa ve optimizasyon adımı atlanacak şekilde bir mantık kurulduysa (veya optimizasyon sonrası), bu yüklenmiş program yeni atıfları sınıflandırmak için kullanılabilir.
    * Kodunuzdaki `main_inference()` benzeri bir fonksiyon veya test bölümü, yüklenmiş `optimized_program`'ı kullanarak örnek bir atıfın nasıl sınıflandırıldığını gösterir.

## Kod Yapısı (Özet)

* **`dspy_citation_classifier.py`**: Ana betik. Veri yükleme, DSPy bileşenlerinin (Signature, Module) tanımı, LLM yapılandırması, `MIPROv2` ile optimizasyon ve optimize edilmiş programla çıkarım yapma mantığını içerir.
* **`load_and_prepare_trainset(...)`**: CSV dosyalarından `dspy.Example` listeleri oluşturan fonksiyon.
* **`CitationIntentSignature`**: LLM'e görevin nasıl yapılacağını tanımlayan DSPy Signature sınıfı.
* **`ClassifyCitation`**: Sınıflandırma görevini yürüten DSPy Module sınıfı.
* **`exact_match_metric(...)`**: Optimizasyon sırasında performansı ölçmek için kullanılan metrik.
* **`data/` klasörü**: `trainset.csv` ve `devset.csv` dosyalarını içerir.
* **`optimized_citation_classifier.json`**: Başarılı bir optimizasyon sonrası kaydedilen, optimize edilmiş programın durumunu içeren dosya.
* **`mipro_optimization_logs/`** (Eğer `log_dir` kullanılırsa): `MIPROv2`'nin detaylı optimizasyon loglarını içerir.

## Kullanılan Temel DSPy Kavramları

* **`dspy.Signature`**: LLM'den ne beklendiğini (girdiler, çıktılar, talimatlar) tanımlar.
* **`dspy.Module`**: Bir veya daha fazla `Signature` kullanarak belirli bir görevi yerine getiren programatik bileşenler.
* **`dspy.Predict` / `dspy.ChainOfThought`**: Basit veya adımlı düşünme gerektiren LLM çağrılarını yöneten temel DSPy modülleri.
* **`dspy.teleprompt.MIPROv2`**: Talimatları ve few-shot demo'ları optimize etmek için kullanılan gelişmiş bir teleprompter (optimizer).
* **`dspy.Example`**: DSPy'ın eğitim, geliştirme ve test verilerini temsil etmek için kullandığı standart format.
* **`dspy.LM`**: Farklı LLM API'leriyle (OpenAI, Gemini vb.) etkileşim kurmak için kullanılan soyutlama katmanı.

## Gelecekteki İyileştirmeler (İsteğe Bağlı)

* Daha kapsamlı bir metrik kullanmak (örn: F1 skoru, her sınıf için ayrı metrikler).
* Weights & Biases veya MLflow gibi deney takip araçlarıyla entegrasyon.
* Farklı LLM'ler için optimize edilmiş ayrı programlar kaydetme ve yönetme.
* Kullanıcı arayüzü (örn: Streamlit veya Flask ile basit bir web arayüzü) eklemek.

---

Bu README taslağını projenizin özel durumuna göre (örneğin, kullanılan kesin LLM modelleri, spesifik optimizasyon parametreleri, projenizin Git repo adresi vb.) detaylandırabilirsiniz. Umarım işinize yarar!