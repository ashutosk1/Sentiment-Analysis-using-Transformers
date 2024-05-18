import string
from constants import LIST_OF_EMOJIS, LIST_OF_REPLACED_LETTERS

class TextClassifier:
    """This class provides methods to clean and classify the text elements (words) as URLs, emojis, usernames
    or filter out punctuations to construct the text in best possible way to extract the potential sentiment
    out of it.
    """

    def __init__(self, replace_letters: dict = None):
        self.replace_letters = replace_letters or {}

    
    def is_url(self, word):
        """Checks if the given word is a URL (starting with http/https/www).
        """
        return word[:7] == "http://" or word[:8] == "https://" or word[:4] == "www."
    

    def is_emoji_not_punctuation(self, word):
        """if the given word is an emoji that is not considered punctuation.
        """
        return word in LIST_OF_EMOJIS.keys()
    
    
    def is_username(self, word):
        """ Checks if the given word is a username starting with an "@" symbol and containing
        alphanumeric characters and underscores.
        """
        ascii_letters = string.ascii_letters
        digits = string.digits
        chars = ascii_letters + digits + "_"
        if word[0] == "@":
            return all(x in chars for x in word[1:])


    def filter_punctuation(self, word):
        """ Filters out punctuation characters from the given word. Optionally, replaces specific
        punctuation characters with provided replacements.
        """
        punctuations = string.punctuation
        chars = string.ascii_letters + string.digits
        filtered_word = word
        for idx, curr_char in enumerate(word):
            if curr_char in punctuations:
                if curr_char in LIST_OF_REPLACED_LETTERS:
                    filtered_word = filtered_word.replace(curr_char, (" " + LIST_OF_REPLACED_LETTERS[curr_char]), 1)
                filtered_word = filtered_word.replace(curr_char, "", 1)
        return filtered_word


