import re
import string
import warnings
import config


class TextCleaner:
    def __init__(self):
        config.set_seed()

    def fraction_ascii(self, text: str) -> dict:
        """how much percent of the string is ascii"""
        text_length = len(text)
        ascii_count = len(text.encode("ascii", "ignore"))
        ascii_percent = ascii_count / text_length
        return {
            "ascii_percent": ascii_percent,
            "non_ascii_percent": 1 - ascii_percent,
        }

    # def convert_emoji_to_text(self, text: str) -> str:
    #     """Convert emoji to text"""
    #     warnings.warn("Not Working")
    #     return emoji.demojize(text)

    def remove_punctuations(
        self, text: str, punct_str_to_remove: str = string.punctuation
    ) -> str:
        """Convert Punctuations to Null"""
        translator = str.maketrans("", "", punct_str_to_remove)
        return text.translate(translator)

    def url_to_null(self, text: str) -> str:
        """Replace URL with Empty String"""
        return re.sub(r"http\S+", "", text)

    def lemmatize_english(self, text: str) -> str:
        """Lemmatize English"""
        raise NotImplementedError("Not Implemented Yet")

    def tokenize_english(self, text):
        """Tokenize English Text"""
        raise NotImplementedError("Not Implemented Yet")

    def remove_handles(self, text: str) -> str:
        clean_text = re.sub("@[A-Za-z0-9_]+.\s?", "", text)
        clean_text = re.sub("#[A-Za-z0-9_]+.\s?", "", clean_text)
        return clean_text

    def single_text_cleaner(self, text: str) -> str:
        """Cleans a Text, and returns it"""
        text = text.lower()  # Lowercase
        text = self.url_to_null(text)
        text = self.remove_handles(text)
        text = self.remove_punctuations(text)
        # text = self.convert_emoji_to_text(text)
        return text
