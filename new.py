import pickle
from toxicity_model_utils import classify_text

# Example usage
input_text = "This is a toxic comment."
prediction = classify_text(input_text)
print("Predicted class:", prediction)
