import spacy
import numpy as np


def vectorize_word(simimlarity, failure_final_text, requirement_final_text):
    if type(failure_final_text) and type(requirement_final_text) != str:
        failure_final_text = ' '.join(map(str, failure_final_text))
        requirement_final_text = ' '.join(map(str, requirement_final_text))

    nlp = spacy.load("en_core_web_lg")
    doc1 = nlp(failure_final_text)
    doc2 = nlp(requirement_final_text)
    if not simimlarity:
        vector = np.concatenate([doc1.vector, doc2.vector])
    else:
        vector = np.concatenate([simimlarity, doc1.vector, doc2.vector])
    return vector
