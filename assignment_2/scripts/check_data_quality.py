import utils
from collections import Counter

def count_word_frequencies(input_path, top_n=50):
    review_texts, _ = utils.read_json_data(input_path) 
    
    review_texts = list(map(str, review_texts))
    
    word_counter = Counter()
    
    for text in review_texts:
        words = text.split() 
        word_counter.update(words)

    most_common_words = word_counter.most_common(top_n)
    
    print("Most Frequent Words in the Dataset:")
    for word, count in most_common_words:
        print(f"{word}: {count}")

input_path = "/deac/csc/classes/csc373/data/fantasy/fantasy_10000.json"
count_word_frequencies(input_path)