from transformers import pipeline
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import sentencepiece
import torch

def rezumat_text(text):
    nume_model = "google/pegasus-cnn_dailymail"
    lungime_text = len(text)
    pegasus_tokenizer = PegasusTokenizer.from_pretrained(nume_model)
    pegasus_model = PegasusForConditionalGeneration.from_pretrained(nume_model)
    tokens = pegasus_tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    rezumat_encoded = pegasus_model.generate(
        **tokens,
        min_length=75,
        max_length=100,
        num_beams=4,
        early_stopping=True
    )

    rezumat = pegasus_tokenizer.decode(rezumat_encoded[0], skip_special_tokens=True).replace("<n>", "\n").replace(" .", ".")

    return(rezumat)
