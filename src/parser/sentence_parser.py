import string
import statistics
import pandas as pd
from fpd.feature_gen import remove_invalid_keystrokes, split_into_four


def is_capital_letter(letter: str) -> bool:
    if letter in list(string.ascii_uppercase):
        return True
    return False


def helping_verbs():
    return [
        "am",
        "did",
        "having",
        "should",
        "are",
        "do",
        "is",
        "was",
        "be",
        "does",
        "may",
        "were",
        "been",
        "going to",
        "might",
        "will",
        "being",
        "had",
        "must",
        "will be",
        "can",
        "has",
        "ought to",
        "will have",
        "could",
        "have",
        "shall",
        "would",
    ]


def articles():
    return ["the", "an", "a"]


class SentenceParser:
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path

    def path(self):
        return self.csv_file_path

    def as_df(self):
        return pd.read_csv(self.path(), header=None)

    def letters(self):
        return remove_invalid_keystrokes(split_into_four(self.as_df()))

    def make_sentences(self):
        filtered = []
        ignorable = ["Key.cmd", "Key.tab", "Key.shift", "Key.shift_r"]
        keys = self.letters().iloc[:, 1]
        action = self.letters().iloc[:, 0]
        assert action.size == keys.size
        for i in range(keys.size):
            if action[i] == "P":
                filtered.append(keys[i])
        sentence = ""
        next = 0
        for i in range(len(filtered)):
            if i + 1 < len(filtered) and i - 1 >= 0:
                key = filtered[i]
                next = (next + 1) % len(filtered)
                if not key in ignorable:
                    if key == "Key.enter":
                        sentence += "\n"
                        continue
                    elif key == "Key.space":
                        sentence += " "
                        continue
                    elif key == "Key.backspace" or key == "Key.ctrl":
                        continue
                    elif filtered[i - 1] == "Key.ctrl":
                        # Technically when the participant hits ctrl c, they are ending the current session and
                        # starting a new one so that should probably be thought of a as a distinct sentence
                        sentence += "\n"
                        continue
                    elif filtered[next] == "Key.backspace":
                        continue
                    sentence += key.strip()
        return sentence

    def number_of_words_feature(self):
        return len(self.get_words())

    def get_words(self):
        sentences = self.make_sentences()
        word = ""
        words = []
        for letter in sentences:
            if not letter == " ":
                word += letter
            if letter == " " or letter == "\n":
                words.append(word)
                word = ""
        return words

    def punctuation_count_feature(self):
        count = 0
        sentences = self.make_sentences()
        for letter in sentences:
            if letter in string.punctuation:
                count += 1
        return count

    def mean_word_length_feature(self):
        words = self.get_words()
        word_count = 0
        size = 0
        for word in words:
            size += len(word)
            word_count += 1
        return size / word_count

    def standard_deviation_word_length_feature(self):
        words = self.get_words()
        lengths = []
        for word in words:
            lengths.append(len(word))
        return statistics.stdev(lengths)

    def helping_verbs_count_feature(self):
        count = 0
        verbs = helping_verbs()
        words = self.get_words()
        for word in words:
            if word in verbs:
                count += 1
        return count

    def articles_count_feature(self):
        count = 0
        article = articles()
        words = self.get_words()
        for word in words:
            print(word)
            if word in article:
                count += 1
        return count

    def capital_letters_count_feature(self):
        sentence_str = self.make_sentences()
        count = 0
        for letter in sentence_str:
            if is_capital_letter(letter):
                count += 1
        return count
