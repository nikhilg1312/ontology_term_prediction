import spacy
import pandas as pd
nlp = spacy.load('en_core_sci_lg')

ip_ontology_df = pd.read_csv(
    "/home/nikhil/Documents/Projects/glove_try1/try1/spacy_word_embeding/true_labels.csv").iloc[:, 1]
ip_labels_df = pd.read_csv("/home/nikhil/Documents/Projects/glove_try1/try1/spacy_word_embeding/ip_labels.csv").iloc[:,
               1]


def get_similar_ontology_term(term, onto_df):
    op_df = pd.DataFrame(columns=["term", "match%"])
    n_term = nlp(str.lower(term))

    for onto_term in onto_df:
        n_onto_term = nlp(str.lower(onto_term))
        op_df = op_df.append([{"term": onto_term, "match%": n_term.similarity(n_onto_term)}])
    return op_df


output_matches = get_similar_ontology_term('DOB', ip_ontology_df)
print(output_matches.nlargest(5, "match%"))


output_matches = get_similar_ontology_term('DOB is date of birth', ip_ontology_df)
print(output_matches.nlargest(5, "match%"))
