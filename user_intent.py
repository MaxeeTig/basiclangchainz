import os
from pydantic import BaseModel
from transformers import pipeline
from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model="mistral-large-latest")
from langchain_core.messages import HumanMessage, SystemMessage

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", multi_label=False, model="facebook/bart-large-mnli")

# ===== function to clarify user intent with LLM
def clarify_intent(user_input):
    system_prompt = '''
As helpful assistant on the first line of customer support you:
1. Carefully review user's query.
2. Define what user wants to know.
3. Re-phrase user's query.
4. Create and print out three different versions of user's query.
5. Create and print out short description summarizing user intent.

#Output:#
Should be structured in the following format:
{'user':' ', 'message': ' ', 'intent': ' '}, where
'user' - echo of user's query
'message' - concatenated re-phrased questions
'intent' - summary of user's intent

#Examples#:
{'user':'What is the distance between Paris and Rome?', 'message': 'How long is the path between Paris and Rome?', 'intent': 'distance between cities'}
{'user':'How can I get from Paris to Rome?', 'message': 'What transport can I use to go between Paris and Rome', 'intent': 'transportation options'}

Do not answer user's question. Just return questions and summary intent.
If user enters random symbols please return excuse statement and Intent: unknown
#Example#:
{'user':'dfwtugeslgrsdgf', 'message': 'Sorry, I don't understand the question.', 'intent': 'unknown'}
'''

    messages = [
    SystemMessage(system_prompt),
    HumanMessage(user_input),
    ]

    print(f"Debug: Clarifying user intent...")
    response = model.invoke(messages)
    print(f"Debug. LLM clarifies that: {response.content}")

    # Parse the response content as a dictionary
    import ast
    response_dict = ast.literal_eval(response.content)
    return response_dict

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
        hypothesis_template=hypothes_template,
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
