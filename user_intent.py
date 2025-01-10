import os
from pydantic import BaseModel
from transformers import pipeline
from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model="mistral-large-latest")
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict, Annotated


class LLMOutput(TypedDict):
    """Generated LLM structured output."""
    query: Annotated[str, ..., "Structured output dictionary."]

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", multi_label=False, model="facebook/bart-large-mnli")

# ===== fuction to clarify user intent with LLM
def clarify_intent(user_input):
    system_prompt = '''
As helpful assistant on the first line of customer support you:
1. Carefully review user's query.
2. Define what user wants to know. 
3. Create and print out short description summarizing user intent.

#Output:#
Should be string summarizing user intent.
#Examples#:
User: 'What is the distance between Paris and Rome?'
Assistant: 'distance between cities'
User: 'How can I get from Paris to Rome?'
Assistant: 'transportation options'

Do not answer user's question. Just return summary intent.
If user enters random symbols return Reponse: unknown 
#Example#:
User: 'dfwtugeslgrsdgf'
Assistant: 'unknown'
'''

    messages = [
    SystemMessage(system_prompt),
    HumanMessage(user_input),
    ]

    structured_llm = model.with_structured_output(LLMOutput)

    print(f"Debug: Clarifing user intent...")
    response = structured_llm.invoke(messages)

    #    response = model.invoke(messages)
    print(f"Debug. LLM clarifies that: {response}")

    return response


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

