import torch
from transformers.pipelines import pipeline, Conversation

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():  # Macbook Metal
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ===========================================================================
# Question answering
# ===========================================================================
qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=device)

qa(dict(
    question="What is human life expectancy in the United States?",
    context="In 2018, the life expectancy in the United States was 78.6 years."
))

qa(dict(
    question="How to change font size on Twitter?",
    context="To manage your font size, text color and background mode via twitter.com, "
            "click on More  then select Display from the menu. Choose your preferred font "
            "size and color. Use the radio buttons to choose between the default white background "
            "or the two dark mode options Dim and Lights Out. Dark mode options are also available "
            "on Twitter for iOS and Twitter for Android."
))

# ===========================================================================
# Conversational
# ===========================================================================
# "BlackSamorez/rudialogpt3_medium_based_on_gpt2_2ch"
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")
conversation = Conversation("Going to the movies tonight - any suggestions?")
conversation = chatbot(conversation, max_length=1000, pad_token_id=50256)
conversation.add_user_input("Is it an action movie?")
conversation = chatbot(conversation, max_length=1000, pad_token_id=50256)
conversation.add_user_input("What is your name?")
conversation = chatbot(conversation, max_length=1000, pad_token_id=50256)
print(conversation.past_user_inputs)

# ===========================================================================
# Document question answering
# ===========================================================================
nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)

nlp(
    "https://d1csarkz8obe9u.cloudfront.net/posterpreviews/simple-resume-design-template-fc913cd1d0931c61fceefee01e312218_screen.jpg",
    "What is the name of the person?",
)
nlp(
    "https://d1csarkz8obe9u.cloudfront.net/posterpreviews/simple-resume-design-template-fc913cd1d0931c61fceefee01e312218_screen.jpg",
    "What does Martin do?",
)
