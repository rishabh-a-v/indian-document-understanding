"""Model wrappers. Heavy imports are lazy so the rest of the package (metrics,
retrieval, tests) works without torch/transformers installed."""
import os

PATRAM_ID = "bharatgenai/patram-7b-instruct"
IT2_INDIC_EN = "ai4bharat/indictrans2-indic-en-1B"
IT2_EN_INDIC = "ai4bharat/indictrans2-en-indic-1B"


def hf_login() -> bool:
    """Log in to Hugging Face from $HF_TOKEN or a Colab secret named HF_TOKEN.
    Patram is a gated repo: you must also have been granted access on its model page."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            token = None
    if token:
        from huggingface_hub import login
        login(token=token)
        return True
    return False


class PatramVLM:
    """Callable: vlm(PIL.Image, prompt, max_new_tokens=512) -> str"""

    def __init__(self, model_id: str = PATRAM_ID, load_4bit=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("Patram-7B needs a GPU (Colab: Runtime > Change runtime type > GPU).")
        if load_4bit is None:  # < 20 GB VRAM (e.g. T4) -> 4-bit
            load_4bit = torch.cuda.get_device_properties(0).total_memory < 20e9
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        kw = dict(trust_remote_code=True, device_map={"": 0})
        if load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        else:
            kw["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()

    def __call__(self, image, prompt: str, max_new_tokens: int = 512) -> str:
        import torch
        from transformers import GenerationConfig

        inputs = self.processor.process(images=[image.convert("RGB")], text=prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            out = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=max_new_tokens,
                                 stop_strings=["<|endoftext|>"], do_sample=False),
                tokenizer=self.processor.tokenizer,
            )
        gen = out[0, inputs["input_ids"].size(1):]
        return self.processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


class _IndicDirection:
    def __init__(self, model_id):
        import torch
        from IndicTransToolkit.processor import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if not hasattr(self.tok, "_special_tokens_map"):  # newer transformers compat
            self.tok._special_tokens_map = {}
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype).to(self.device).eval()
        self.ip = IndicProcessor(inference=True)

    def translate(self, sentences, src, tgt, batch_size=8):
        import torch
        results = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = self.ip.preprocess_batch(sentences[i:i + batch_size], src_lang=src, tgt_lang=tgt)
                enc = self.tok(batch, truncation=True, padding="longest", return_tensors="pt",
                               max_length=256).to(self.device)
                out = self.model.generate(**enc, num_beams=5, max_length=256,
                                          num_return_sequences=1, use_cache=False)
                dec = self.tok.batch_decode(out, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True)
                results += self.ip.postprocess_batch(dec, lang=tgt)
        return results


class Translator:
    """IndicTrans2 both ways. Interface: to_english(text, src) / from_english(text, tgt)."""

    def __init__(self):
        self._indic_en = _IndicDirection(IT2_INDIC_EN)
        self._en_indic = _IndicDirection(IT2_EN_INDIC)

    def to_english(self, text: str, src: str = None) -> str:
        from .langs import detect_lang, split_sentences
        src = src or detect_lang(text)
        if src == "eng_Latn":
            return text
        return " ".join(self._indic_en.translate(split_sentences(text), src, "eng_Latn"))

    def from_english(self, text: str, tgt: str) -> str:
        from .langs import split_sentences
        if tgt == "eng_Latn":
            return text
        return " ".join(self._en_indic.translate(split_sentences(text), "eng_Latn", tgt))
