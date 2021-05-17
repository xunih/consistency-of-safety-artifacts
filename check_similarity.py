import textdistance
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import numpy as np
import tensorflow_hub as hub


def check_similarity_levenshtein(failure_final_text, requirement_final_text):
    # Calculate levenshtein similarity
    similarity_by_levenshtein = textdistance.levenshtein.normalized_similarity(failure_final_text,
                                                                               requirement_final_text)
    return similarity_by_levenshtein


def check_similarity_jaccard(failure_final_text, requirement_final_text):
    # Calculate jaccard similarity
    similarity_by_jaccard = textdistance.jaccard.normalized_similarity(failure_final_text,
                                                                       requirement_final_text)
    return similarity_by_jaccard


def check_similarity_pairwise(failure_final_text, requirement_final_text):
    # Convert the inputs for checking similarity to string if they are not strings
    if type(failure_final_text) and type(requirement_final_text) != str:
        text_one = ' '.join(map(str, failure_final_text))
        text_two = ' '.join(map(str, requirement_final_text))
    else:
        text_one = failure_final_text
        text_two = requirement_final_text

    test_text = [text_one, text_two]
    vector = TfidfVectorizer()
    tfidf = vector.fit_transform(test_text)
    pairwise_similarity_matrix = tfidf * tfidf.T.toarray()
    # Extract needed similarity numbers frm the similarity matrix
    pairwise_similarity = pairwise_similarity_matrix[0][1]

    return pairwise_similarity


def check_similarity_cosine_smallmodel(failure_final_text, requirement_final_text):
    # Calculate cosine similarity by using small data model of spaCy
    nlp = spacy.load('en_core_web_sm')
    if type(failure_final_text) and type(requirement_final_text) != str:
        text_one_pre = ' '.join(map(str, failure_final_text))
        text_two_pre = ' '.join(map(str, requirement_final_text))
    else:
        text_one_pre = failure_final_text
        text_two_pre = requirement_final_text

    text_one = nlp(text_one_pre)
    text_two = nlp(text_two_pre)

    return text_one.similarity(text_two)


def check_similarity_cosine_largemodel(failure_final_text, requirement_final_text):
    # Calculate cosine similarity by using large data model of spaCy
    nlp = spacy.load('en_core_web_lg')
    if type(failure_final_text) and type(requirement_final_text) != str:
        text_one_pre = ' '.join(map(str, failure_final_text))
        text_two_pre = ' '.join(map(str, requirement_final_text))
    else:
        text_one_pre = failure_final_text
        text_two_pre = requirement_final_text

    text_one = nlp(text_one_pre)
    text_two = nlp(text_two_pre)

    return text_one.similarity(text_two)


def check_similarity_universal(failure_final_text, requirement_final_text):
    # tf_hub_cache_dir = "universal_encoder_cached/"
    # np.os.environ["TFHUB_CACHE_DIR"] = tf_hub_cache_dir
    # module_url = tf_hub_cache_dir+"/063d866c06683311b44b4992fd46003be952409c/"
    # embed = hub.load(module_url)
    # Calculate similarity by universal sentence encoder word embeddings
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    text_one_pre = ' '.join(map(str, failure_final_text))
    text_two_pre = ' '.join(map(str, requirement_final_text))
    text = [text_one_pre, text_two_pre]
    embeddings = embed(text)
    similarity = np.inner(embeddings[0], embeddings[1])
    return similarity
