import nltk
import extract_data
import check_similarity
import classifier
import vectorize_word

sim_only_non_file = 'sim_only_non.csv'
sim_only_nor_file = 'sim_only_nor.csv'
vec_only_non_file = 'vect_only_non.csv'
vec_only_nor_file = 'vect_only_nor.csv'
both_fea_non_file = 'both_non.csv'
both_fea_nor_file = 'both_nor.csv'


def main():
    failure_list = extract_data.add_failure()
    sim_only_1, sim_only_2, vect_only_1, vect_only_2, vect_1, vect_2 = get_input_for_classifier(failure_list)
    classifier.classifier(sim_only_1, sim_only_non_file)
    classifier.classifier(sim_only_2, sim_only_nor_file)
    classifier.classifier(vect_only_1, vec_only_non_file)
    classifier.classifier(vect_only_2, vec_only_nor_file)
    classifier.classifier(vect_1, both_fea_non_file)
    classifier.classifier(vect_2, both_fea_nor_file)


def get_input_for_classifier(failure_list):
    sim_only_1 = []
    sim_only_2 = []
    vect_1 = []
    vect_2 = []
    vect_only_1 = []
    vect_only_2 = []
    for failure in failure_list:
        for hazard in failure.target:
            for safety_goal in hazard.target:
                for requirement in safety_goal.target:
                    failure_token = nltk.RegexpTokenizer(r'\w+').tokenize(failure.text)
                    requirement_token = nltk.RegexpTokenizer(r'\w+').tokenize(requirement.text)
                    failure_stop = set(nltk.corpus.stopwords.words('english'))
                    requirement_stop = set(nltk.corpus.stopwords.words('english'))
                    failure_filtered_text = [word for word in failure_token if not word in failure_stop]
                    requirement_filtered_text = [word for word in requirement_token if not word in requirement_stop]
                    failure_lowercase = [failure_filtered_text.lower() for failure_filtered_text in
                                         failure_filtered_text]
                    requirement_lowercase = [requirement_filtered_text.lower() for requirement_filtered_text in
                                             requirement_filtered_text]

                    similarity_levenshtein_nonnormalized = check_similarity.check_similarity_levenshtein(failure.text,
                                                                                                         requirement.text)
                    similarity_levenshtein_normalized = check_similarity.check_similarity_levenshtein(failure_lowercase,
                                                                                                      requirement_lowercase)

                    similarity_jaccard_nonnormalized = check_similarity.check_similarity_jaccard(failure.text,
                                                                                                 requirement.text)
                    similarity_jaccard_normalized = check_similarity.check_similarity_jaccard(failure_lowercase,
                                                                                              requirement_lowercase)

                    similarity_pairwise_nonnormalized = check_similarity.check_similarity_pairwise(failure.text,
                                                                                                   requirement.text)
                    similarity_pairwise_normalized = check_similarity.check_similarity_pairwise(failure_lowercase,
                                                                                                requirement_lowercase)

                    similarity_cosine_s_nonnormalized = check_similarity.check_similarity_cosine_smallmodel(
                        failure.text,
                        requirement.text)
                    similarity_cosine_s_normalized = check_similarity.check_similarity_cosine_smallmodel(
                        failure_lowercase,
                        requirement_lowercase)
                    similarity_cosine_l_nonnormalized = check_similarity.check_similarity_cosine_largemodel(
                        failure.text,
                        requirement.text)
                    similarity_cosine_l_normalized = check_similarity.check_similarity_cosine_largemodel(
                        failure_lowercase,
                        requirement_lowercase)
                    similarity_universial_nonnormalized = check_similarity.check_similarity_universal(failure.text,
                                                                                                      requirement.text)
                    similarity_universial_normalized = check_similarity.check_similarity_universal(failure_lowercase,
                                                                                                   requirement_lowercase)

                    sim_one = []
                    sim_one.append(similarity_levenshtein_nonnormalized)
                    sim_one.append(similarity_jaccard_nonnormalized)
                    sim_one.append(similarity_pairwise_nonnormalized)
                    sim_one.append(similarity_cosine_s_nonnormalized)
                    sim_one.append(similarity_cosine_l_nonnormalized)
                    sim_one.append(similarity_universial_nonnormalized)
                    sim_only_1.append(sim_one)

                    sim_two = []
                    sim_two.append(similarity_levenshtein_normalized)
                    sim_two.append(similarity_jaccard_normalized)
                    sim_two.append(similarity_pairwise_normalized)
                    sim_two.append(similarity_cosine_s_normalized)
                    sim_two.append(similarity_cosine_l_normalized)
                    sim_two.append(similarity_universial_normalized)
                    sim_only_2.append(sim_two)

                    vector_one = vectorize_word.vectorize_word(sim_one, failure.text, requirement.text)
                    vector_two = vectorize_word.vectorize_word(sim_two, failure_lowercase, requirement_lowercase)
                    vect_1.append(vector_one)
                    vect_2.append(vector_two)

                    vector_only_one = vectorize_word.vectorize_word([], failure.text, requirement.text)
                    vector_only_two = vectorize_word.vectorize_word([], failure_lowercase, requirement_lowercase)
                    vect_only_1.append(vector_only_one)
                    vect_only_2.append(vector_only_two)

    return sim_only_1, sim_only_2, vect_only_1, vect_only_2, vect_1, vect_2


if __name__ == '__main__':
    main()
