import dspy
import os
import pandas as pd
import logging
import sys
from dspy.teleprompt import MIPROv2
from dspy.teleprompt import BootstrapFewShot
import traceback

# Metric tanımı (örnek eşleşme kontrolü)
def exact_match_metric(example, prediction, trace=None):
    try:
        ground_truth = str(example.citation_intent).strip().lower()
        predicted_intent = str(prediction.intent).strip().lower()
        is_match = ground_truth == predicted_intent
        return int(is_match)
    except AttributeError:
        return 0
    except Exception as e:
        print(f"Error in metric: {e}")
        # print(f"Example: {example} - Prediction: {prediction}")
        return 0

def load_and_prepare_trainset(csv_path, citation_classes, get_all_samples=False, samples_per_class=2, random_state_val=42):
    try:
        train_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"HATA: '{csv_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
        return []
    except Exception as e:
        print(f"CSV dosyası ('{csv_path}') okunurken bir hata oluştu: {e}")
        return []

    if train_df.empty:
        print(f"Uyarı: '{csv_path}' dosyası boş veya okunamadı.")
        return []

    if "citation_intent" not in train_df.columns or \
       "citation_context" not in train_df.columns or \
       "section" not in train_df.columns:
        print("HATA: CSV dosyasında beklenen sütunlar ('citation_intent', 'citation_context', 'section') bulunamadı.")
        return []

    subset_df = train_df
    if not get_all_samples:
        balanced_subset_list = []
        for cls in citation_classes:
            if cls in train_df["citation_intent"].unique():
                class_subset_df = train_df[train_df["citation_intent"] == cls]
                # İstenen örnek sayısını veya sınıftaki mevcut örnek sayısını (hangisi daha küçükse) al
                sample_n = min(samples_per_class, len(class_subset_df))
                if sample_n > 0:
                    class_subset = class_subset_df.sample(n=sample_n, random_state=random_state_val, replace=False)
                    balanced_subset_list.append(class_subset)

        if not balanced_subset_list:
            print("Uyarı: Trainset için dengeli alt küme oluşturulamadı (sınıflar bulunamadı veya boş).")
            return []

        subset_df = pd.concat(balanced_subset_list)
        if subset_df.empty:
            print("Uyarı: Birleştirilmiş subset DataFrame boş.")
            return []

    trainset_examples = []
    try:
        trainset_examples = [
            dspy.Example(
                citation=row["citation_context"],
                section=row["section"],
                citation_intent=row["citation_intent"]
            ).with_inputs("citation", "section") # Modelin Signature'daki InputField'ları ile eşleşmeli
            for _, row in subset_df.iterrows()
        ]
    except KeyError as e:
        print(f"HATA: dspy.Example oluşturulurken CSV sütunlarından biri bulunamadı: {e}")
        return [] # Hata durumunda boş liste

    if not trainset_examples:
        print("Uyarı: DSPy Example nesnelerinden oluşan trainset boş.")

    return trainset_examples

class CitationIntentSignature(dspy.Signature):
    """
    You are an expert academic editor specializing in computer science and artificial intelligence. Your task is to meticulously analyze and classify academic citations from Turkish research papers based on their rhetorical intent. The citations and their corresponding section titles will be provided in Turkish.
    Your goal is to classify each citation into one of the following six categories. These categories are inspired by the Web of Science (WoS) citation classification schema (Clarivate) and have been refined with details from their guidelines for enhanced clarity:
    1.  **background**:
        * **Description**: The cited work is referred to for general context, historical information, or to acknowledge foundational studies that are **not directly built upon** by the current research. These citations help set the stage, place the current study within a broader scholarly conversation, or might acknowledge a method/software that is not central to the current paper's core work.
        * **WoS Insight**: previously published research that orients the current study within a scholarly area.
        * **Typical Turkish Sections**: 'Giriş' (Introduction), 'Literatür Taraması' (Literature Review), 'İlgili Çalışmalar' (Related Work), 'Genel Bilgiler' (General Information).
        * **Key Idea**: Provides broader context or acknowledges foundational knowledge that is not a direct methodological pillar for the current study.

    2.  **basis**:
        * **Description**: The cited work provides a fundamental pillar for the current study. The current research **directly reports using or adapting the specific methods, algorithms, data sets, software, or equipment** described in the cited work for its own execution. These citations are central to how the research was designed and conducted. Studies usually rely on a relatively small number of such foundational works.
        * **WoS Insight**: references that report the data sets, methods, concepts and ideas that the author is using for her work directly or on which the author bases her work
        * **Typical Turkish Sections**: 'Yöntem' (Methodology), 'Materyal ve Metot' (Material and Method), 'Model Tasarımı' (Model Design), 'Veri Seti' (Dataset), 'Uygulama' (Implementation).
        * **Key Idea**: The current study's methodology or execution directly and essentially depends on the content of the cited work.

    3.  **discuss**:
        * **Description**: The cited work is actively and substantively discussed, analyzed, or critically evaluated within the current study. This can involve a detailed examination of its specific arguments, findings, theories, contributions, strengths, or weaknesses. The discussion often relates the cited work's importance or relevance to the current research, or compares/contrasts its approach beyond a simple statement of similar/dissimilar results.
        * **WoS Insight**: references mentioned because the current study is going into a more detailed discussion.
        * **Typical Turkish Sections**: 'Literatür Taraması' (Literature Review), 'Tartışma' (Discussion), 'Bulgular ve Tartışma' (Results and Discussion), 'İlgili Çalışmalar' (Related Work).
        * **Key Idea**: The cited work is a subject of significant intellectual engagement, analysis, or critique, not just a simple reference for support or difference of findings.

    4.  **support**:
        * **Description**: The findings, arguments, or **results reported in the cited work are directly compared with those of the current study and are presented as being consistent with, and thereby reinforcing or validating, the results, claims, or conclusions of the current (citing) study.** The current study uses the cited work to show that its own findings are corroborated. These citations are generally few in number per study.
        * **WoS Insight**: references which the current study reports to have similar results to. This may also refer to similarities in methodology or in some cases replication of results.
        * **Typical Turkish Sections**: 'Bulgular' (Results), 'Sonuçlar' (Results/Conclusion), 'Tartışma' (Discussion), 'Doğrulama' (Validation).
        * **Key Idea**: The cited work's outcomes lend credibility and support to the current study's findings by demonstrating consistency.

    5.  **differ**:
        * **Description**: The findings, arguments, or **results reported in the cited work are directly compared with those of the current study and are presented as contrasting with, contradicting, or highlighting different perspectives or outcomes compared to those of the current (citing) study.** The current study uses the cited work to highlight how its own findings differ or offer an alternative view. These citations are also generally few in number per study.
        * **WoS Insight**: references which the current study reports to have differing results to. This may also refer to differences in methodology or differences in sample sizes, affecting results.
        * **Typical Turkish Sections**: 'Bulgular' (Results), 'Sonuçlar' (Results/Conclusion), 'Tartışma' (Discussion).
        * **Key Idea**: The cited work's *results* are shown to diverge from, contradict, or present significantly different outcomes when compared to the current study's findings.

    6.  **other**:
        * **Description**: The citation's rhetorical intent cannot be confidently determined from the provided excerpt and context. This category **explicitly includes very short citation phrases (e.g., 3-4 Turkish words) that lack sufficient semantic content to convey a clear purpose**, incomplete citation references, or mentions that don't fit any other specific rhetorical function (e.g., a passing mention without clear intent).
        * **WoS Insight**: Citations that are not classifiable into other specific categories.
        * **Key Idea**: Insufficient information for classification, the citation is semantically too weak for intent analysis, or it serves a purely bibliographic purpose without clear rhetorical intent in the given context.

    **Important Considerations for Classification**:
    * **Language**: The `citation` text and `section` titles in the input JSON will be in **Turkish**. Your classification should be based on understanding this Turkish content.
    * **Context is Key**: While the 'section' (Turkish section title) where the citation appears provides a strong contextual clue (typical Turkish section names are provided as hints for each category), the primary basis for classification should be the semantic content and rhetorical function of the 'citation' text itself. A citation's intent might occasionally differ from its section's typical use.
    * **Zero-Shot Task**: This is a zero-shot classification task. Do not expect or use any pre-defined examples for learning within this prompt.

    **Input Format**:
    Each citation will be provided as a JSON object with the following fields:
    * `id`: A unique identifier for the citation (String).
    * `citation`: The citation sentence or excerpt **in Turkish** (String).
    * `section`: The title of the section in which the citation appears, **in Turkish** (String).

    All citations will be presented as a JSON array.
    Example of Input Data Structure:
    [
      {
        "id": "unique_id_1",
        "citation": "Alanyazında bu konuda farklı yaklaşımlar mevcuttur (Yılmaz, 2020; Kaya, 2019).",
        "section": "Giriş"
      },
      {
        "id": "unique_id_2",
        "citation": "Bu çalışmada, Demir ve ark. (2021) tarafından önerilen sinir ağı mimarisi temel alınmıştır.",
        "section": "Yöntem"
      }
    ]

    **Expected Output Format**:
    The output must be a JSON array of objects. Each object should contain the `id` of the citation and its classified `intent`.
    Example of Output Data Structure:
    [
      {
        "id": "unique_id_1",
        "intent": "background"
      },
      {
        "id": "unique_id_2",
        "intent": "basis"
      }
    ]

    Please return your response strictly as a valid JSON array. Do not include any additional commentary, explanation, text, or formatting outside of the JSON array itself.
   """
    citation = dspy.InputField(desc="Citation Context")
    section = dspy.InputField(desc="Citation Section Title")
    intent = dspy.OutputField(desc="Please enter one of the following citation intent: 'background', 'basis', 'discuss', 'support', 'differ', 'other'")

class ClassifyCitation(dspy.Module):
    def __init__(self):
        super().__init__()
        self.citation_intent_signature = CitationIntentSignature
        self.classifier = dspy.ChainOfThought(signature=CitationIntentSignature)
    def forward(self, citation, section):
        prediction = self.classifier(citation=citation, section=section)
        return prediction


# Initialization
CSV_TRAIN_PATH = "data/trainset.csv"
CSV_DEV_PATH = "data/devset.csv"
CITATION_CLASSES = [ "background", "basis", "discuss", "support", "differ", "other"]
MIN_TRAINSET_SIZE_FOR_MIPRO = 3
loaded_program = None
save_path = "optimized_citation_classifier.json"  # Kaydettiğiniz dosyanın yolu

# Initialization
# model = 'openai/gpt-4o-mini'
# model = 'openai/gpt-4o'
# lm = dspy.LM(model, api_key='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')
model = 'gemini/gemini-2.5-flash-preview-05-20'
lm = dspy.LM(model, api_key='AIzaSyD70GnTv82tXv0boDCfJivSC4PZ5B2q9Oo')

dspy.configure(lm=lm)

# Initialization for MIPRO
optimized_program = ClassifyCitation() # Varsayılan, optimize edilmemiş program
try:
    optimized_program.load(save_path)
    print(f"Optimize edilmiş program '{save_path}' dosyasından başarıyla yüklendi.")
    program_initially_loaded=True
except Exception as e:
    print(f"Program yüklenirken bir hata oluştu: {e}")
    program_initially_loaded = False
    loaded_program = ClassifyCitation() # Optimize edilmemiş haliyle yükle

# Initialization for training set
trainset = load_and_prepare_trainset(
    csv_path=CSV_TRAIN_PATH,
    citation_classes=CITATION_CLASSES,
    get_all_samples=True
)

devset = load_and_prepare_trainset(
    csv_path=CSV_DEV_PATH,
    citation_classes=CITATION_CLASSES,
    get_all_samples=True
)

small_balanced_set = load_and_prepare_trainset(
    csv_path=CSV_TRAIN_PATH, # Veya başka bir CSV
    get_all_samples=False, # <<<--- DENGELİ ÖRNEKLEM YAP
    citation_classes=CITATION_CLASSES, # ['background', 'basis', ...] listeniz
    samples_per_class=2,
    random_state_val=42
)


# MIPRO ile optimize et
program_was_optimized = False
if len(trainset) < MIN_TRAINSET_SIZE_FOR_MIPRO:
    print(f"UYARI: Trainset boyutu ({len(trainset)}) Optimizasyonu için çok küçük. Optimizasyon atlanıyor, varsayılan program kullanılacak.")
else:
    print(f"Optimizasyonu {len(trainset)} eğitim örneği ile başlatılıyor...")
    optimizer = MIPROv2(
        metric=exact_match_metric,
        auto='heavy',
        verbose=True
    )

    #optimizer = BootstrapFewShot(
    #    metric=exact_match_metric,
    #    max_bootstrapped_demos=4  # Oluşturulacak demo sayısı (her Predictor için)
    #)

    try:
        compiled_program = optimizer.compile(
            student=optimized_program,
            trainset=trainset,
            valset=devset,
            max_bootstrapped_demos=6,           # trainset'ten seçilecek demo sayısı
            max_labeled_demos=0,                # Eğer manuel demo vermiyorsanız 0
            # API kullanılmadan önce onay isteme için
            requires_permission_to_run=False,
        )
        optimized_program = compiled_program
        program_was_optimized = True
        optimized_program.save(save_path)
        print("Optimizer çalıştırıldı, optimize edildi ve kaydedildi...")

    except Exception as e:
        print(f"Optimizasyonu sırasında bir hata oluştu: {e}")
        traceback.print_exc()
        print("Optimizasyon başarısız oldu, varsayılan program kullanılacak.")


# Test verisi
example_data = {
    "citation": "Yöntemimiz, literatürdeki yaklaşımlarla benzer sonuçlar üretmektedir (Çelik ve Aydın, 2022).",
    "section": "Bulgular"
}

# Sonuç
print("\n\n------------------------------------------------------------")
print("------------------------------------------------------------")
print("--- Program Çıktısı ---")
try:
    result = optimized_program.forward(citation=example_data["citation"], section=example_data["section"])
    print("Tahmin edilen intent:", result.intent if hasattr(result, 'intent') else "N/A")
    if hasattr(result, 'reasoning') and result.reasoning:
        print("Gerekçe:", result.reasoning)
    elif hasattr(result, 'rationale') and result.rationale: # ChainOfThought için fallback
        print("Gerekçe (rationale):", result.rationale)
    else:
        print("Gerekçe üretilmedi veya bulunamadı.")

except Exception as e:
    print(f"Program çalıştırılırken bir hata oluştu: {e}")


# --- Optimize Edilmiş Prompt (Classifier) ---
print("------------------------------------------------------------")
print("------------------------------------------------------------")
print("--- Optimize Edilmiş Prompt (Classifier) ---")

if program_was_optimized and hasattr(optimized_program, 'classifier'):
    print(f"Optimize edilmiş programın türü: {type(optimized_program)}")

    chain_of_thought_module = optimized_program.classifier
    print(f"Optimize edilmiş programın Classifier (ChainOfThought) modülü bulundu. Türü: {type(chain_of_thought_module)}")

    # ChainOfThought'un içindeki ana Predict modülüne erişelim
    # Debugger görüntüsüne göre bu özellik 'predict' (küçük harf) olarak adlandırılmış.
    internal_predict_module = None
    if hasattr(chain_of_thought_module, 'predict') and isinstance(chain_of_thought_module.predict, dspy.Predict): # 'predictor' yerine 'predict'
        internal_predict_module = chain_of_thought_module.predict
        print("ChainOfThought modülü içinde 'predict' (dspy.Predict modülü) bulundu.")
    else:
        print(f"ChainOfThought modülü içinde 'predict' (dspy.Predict modülü) bulunamadı. 'predict' özelliği var mı? {hasattr(chain_of_thought_module, 'predict')}")
        if hasattr(chain_of_thought_module, 'predict'):
            print(f"'predict' özelliğinin türü: {type(chain_of_thought_module.predict)}")


    if internal_predict_module:
        # 1. Signature'daki talimatları yazdır
        if hasattr(internal_predict_module, 'signature') and internal_predict_module.signature:
            print("\nInternal Predict Modülünün Signature'ı (Talimatlar):")
            print(str(internal_predict_module.signature))
            if internal_predict_module.signature.instructions:
                 print("\nSadece Talimatlar (Instructions):")
                 print(internal_predict_module.signature.instructions)
        else:
            print("Internal Predict Modülünde signature bulunamadı.")

        # 2. Kullanılan demoları (eğer varsa) yazdır
        if hasattr(internal_predict_module, 'demos') and internal_predict_module.demos:
            print(f"\nKullanılan Demo Sayısı (Internal Predict Modülü): {len(internal_predict_module.demos)}")
            print("Demolar:")
            for i, demo in enumerate(internal_predict_module.demos):
                print(f"--- Demo {i+1} ---")
                # Demo içeriğini daha okunaklı yazdırmak için
                # dspy.Example nesnelerinin içini görmek gerekebilir.
                # print(demo)
                # Demo'nun input ve output alanlarını gösterelim (Signature'daki alan adlarına göre)
                demo_inputs_str = []
                for k_sig, _ in internal_predict_module.signature.input_fields.items():
                    if k_sig in demo:
                        demo_inputs_str.append(f"  {k_sig}: {demo[k_sig]}")
                print(" Inputs:\n" + "\n".join(demo_inputs_str))

                demo_outputs_str = []
                for k_sig, _ in internal_predict_module.signature.output_fields.items():
                     if k_sig in demo:
                        demo_outputs_str.append(f"  {k_sig}: {demo[k_sig]}")
                print(" Outputs:\n" + "\n".join(demo_outputs_str))
        else:
            print("\nInternal Predict Modülünde demo bulunamadı.")

        # ... (Tahmini Tam Prompt Yapısı kısmı aynı kalabilir, internal_predict_module kullanacak)

    else:
        print("\nChainOfThought içindeki ana Predict modülü (internal_predict_module) bulunamadı.")


elif not program_was_optimized:
    print("Program optimize edilmedi (trainset boyutu yetersiz veya optimizasyon atlandı).")
else:
    print("Optimize edilmiş programda 'classifier' (ChainOfThought modülü) özniteliği bulunamadı.")

# LM'in son çağrılarını incelemek için (debugging)
print("------------------------------------------------------------")
print("------------------------------------------------------------")
print("--- LM Son Çağrılar (Son 1) ---")
try:
    lm.inspect_history(n=1)
except Exception as e:
    print(f"LM geçmişi incelenirken hata: {e}")

print("------------------------------------------------------------")
print("------------------------------------------------------------")
print("Program tamamlandı. dspy.__version__ " + str(dspy.__version__))
