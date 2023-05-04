from transformers import pipeline

from src.internal.father_classes.CnnFeatureExtractorFather import CnnFeatureExtractorFather


class TextualCnnFeatureExtractor(CnnFeatureExtractorFather):
    def __init__(self, gpu='-1'):
        self._tokenizer = None
        super().__init__(gpu)

    def set_model(self, model_name):
        """
        Args:
            model_name: is the name of the model to use.
                        NOTE: in this case we are using transformers so the model name have to be in its list
        Returns: nothing but it initializes the protected model and tokenizer attributes, later used for extraction
        """
        sentiment_pipeline = pipeline(model=model_name)
        model = list(sentiment_pipeline.model.children())[-3]
        model.eval()
        model.to(self._device)
        self._model = model
        self._tokenizer = sentiment_pipeline.tokenizer

    def extract_feature(self, sample_input):
        '''
        if isinstance(sample_input, list):
            output_list = []
            for input_el in sample_input:
                output = self._tokenizer.encode_plus(input_el, return_tensors="pt").to(self._device)
                output_list.append(self._model(**output.to(self._device)).pooler_output.detach().cpu().numpy())
            return output_list
        else:
            output = self._tokenizer.encode_plus(sample_input, return_tensors="pt").to(self._device)
            return self._model(**output.to(self._device)).pooler_output.detach().cpu().numpy()
        '''
        output = self._tokenizer.encode_plus(sample_input, return_tensors="pt").to(self._device)
        return self._model(**output.to(self._device)).pooler_output.detach().cpu().numpy()
