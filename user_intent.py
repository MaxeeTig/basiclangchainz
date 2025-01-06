import os
from pydantic import BaseModel
from transformers import pipeline


# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", multi_label=False, model="facebook/bart-large-mnli")

# ===== function to classify user intent in its input 
def classify_text(text: str, candidate_labels: list) -> dict:
    """
    Classify the given text using the facebook/bart-large-mnli model.

    :param text: The text to classify.
    :param candidate_labels: A list of candidate labels.
    :return: A dictionary containing the top label and its confidence level.
    """
    hypothesis_template = "This is a request for {}."

    # Perform classification
    result = classifier(
        text,
        candidate_labels=candidate_labels,
        hypothesis_template=hypothesis_template,
        multi_label=False
    )

    # Extract the top label and its confidence score
    top_label = result['labels'][0]
    confidence = result['scores'][0]

    # Return the classification result
    return {
        "top_label": top_label,
        "confidence": confidence
    }

