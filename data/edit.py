import json
import stanza
import stanza.pipeline
from object_dataset import JSONObject, LocalityObject, PortabilityObject, PromptObject

nlp = stanza.Pipeline('en', processors="tokenize, mwt, pos, lemma, depparse")

def extract_subject(sentence: str) -> str:
    doc = nlp(sentence)
    result = []

    for sentence in doc.sentences:
        for word in sentence.words:
            if word.deprel == "nsubj":
                print(f"Subject {word.text}")
                result.append(word.text)

    return result


def create_edit_pair(question, answer):
    subject = extract_subject(
        question
    )
    pass




if __name__ == "__main__":
    extract_subject("Which family does Epaspidoceras belong to?")