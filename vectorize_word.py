import spacy
import numpy as np


def vectorize_word(similarity, failure_final_text, requirement_final_text):
    if type(failure_final_text) and type(requirement_final_text) != str:
        failure_final_text = ' '.join(map(str, failure_final_text))
        requirement_final_text = ' '.join(map(str, requirement_final_text))

    nlp = spacy.load("en_core_web_lg")
    doc1 = nlp(failure_final_text)
    doc2 = nlp(requirement_final_text)
    if not similarity:
        # Concatenate the texts of failure and requirement to get a word vector
        vector = np.concatenate([doc1.vector, doc2.vector])
    else:
        # Concatenate the texts of similarities, failure, and requirement to get a word vector
        vector = np.concatenate([similarity, doc1.vector, doc2.vector])
    return vector
