from user_intent import classify_text
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from query_writer import validate_query, write_query, execute_query, generate_answer, State

model = ChatMistralAI(model="mistral-large-latest")

def call_llm(prompt, model):
    return model.invoke(prompt)

def main_chat_loop(welcome_message="Welcome to Chatbot!"):
    """Main function to run the chatbot."""
    print(welcome_message)
    while True:
        user_input = input("Please enter your query (type 'exit' to quit): ")

        if user_input.lower() == 'exit':
            print("Exiting the chatbot. Goodbye!")
            break
        else:
            print(f"Getting extra information from local data for query {user_input}...")
            intent_labels = ["draw graph", "customer portrait", "cluster", "select data from database"]
            user_intent = classify_text(user_input, intent_labels)
            print(f"User needs to: {user_intent['top_label']}")
            print(user_intent)

            if user_intent['top_label'] == 'draw graph':
                print("Draw graph")
            elif user_intent['top_label'] == 'select data from database':
                print("Select data from database")
            else:
                print("Intent not recognized")

if __name__ == "__main__":
    main_chat_loop()
