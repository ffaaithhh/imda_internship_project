from keras.models import load_model
from keras.models import model_from_json
from allennlp.models.model import Model
from allennlp.fairness.fairness_metrics import Independence, Separation, Sufficiency

# load json and create model
json_file = open('model_num.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_num.h5")
print("Loaded model from disk")

class YourModel(Model):

    def __init__(self, *args, **kwargs):
        ...
        # Initialize fairness metric objects
        self._independence = Independence(2, 1)
        self._separation = Separation(2, 1)
        self._sufficiency = Sufficiency(2, 1)
        ...

    def forward(self, *args, **kwargs):
        ...
        # Accumulate metrics over batches
        self._independence(predicted_labels, protected_variable_labels)
        self._separation(predicted_labels, gold_labels, protected_variable_labels)
        self._sufficiency(predicted_labels, gold_labels, protected_variable_labels)
        ...

model = YourModel(...)
...
# Get final values of metrics after all batches have been processed
print(model._independence.get_metric(), model._separation.get_metric(), model._sufficiency.get_metric())