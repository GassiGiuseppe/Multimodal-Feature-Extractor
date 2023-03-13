from transformers import pipeline

from src.internal.father_classes.CnnFeatureExtractorFather import CnnFeatureExtractorFather


class TextualCnnFeatureExtractor(CnnFeatureExtractorFather):
    def __init__(self, gpu='-1'):
        self._tokenizer = None
        super().__init__(gpu)

    def set_model(self, model_name):
        sentiment_pipeline = pipeline(model=model_name)

        # this is as the professor has written
        model = list(sentiment_pipeline.model.children())[-3]  # HERE LAYER???
        # this is as hugme suggest:
        # model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
        model.eval()
        model.to(self._device)
        self._model = model
        self._tokenizer = sentiment_pipeline.tokenizer
        # or
        # AutoTokenizer.from_pretrained("bert-base-cased")

    def extract_feature(self, sample_input):
        output = self._tokenizer.encode_plus(sample_input, return_tensors="pt").to(self._device)
        return self._model(**output.to(self._device)).pooler_output.detach().cpu().numpy()
