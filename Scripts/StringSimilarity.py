# Given two column names, return a similarity score (or Boolean for identity).
# This helps decide if two columns should be compared for content too.
import fuzzy
import logging
from itertools import groupby
import Levenshtein as levenshtein

class StringSimilarity:
    def __init__(self, min_similarity: float):
        """Compare a list of strings for similarity.

        :param min_similarity: The minimum similarity value below which two strings are still considered "equal enough"
        """
        # TODO: Methods that should be used? (e.g. only written methods, not Soundex)
        self.min_similarity = min_similarity

    def identity(self, strings: list):
        # https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
        g = groupby(strings)
        return next(g, True) and not next(g, False)

    def levenshtein_distance(self, strings: list):
        # Edit distance (number of inserts, deletions, changes)
        if len(strings) == 2:
            return levenshtein.distance(strings[0], strings[1])
        else:
            raise NotImplemented("Levenshtein distance only implemented for 2 strings to compare")

    def levenshtein_metric(self, strings: list):
        # 0 if equal, 1 if maximally unequal.
        if len(strings) == 2:
            return levenshtein.distance(strings[0], strings[1]) / max(len(s) for s in strings)
        else:
            raise NotImplemented("Levenshtein metric only implemented for 2 strings to compare")

    def soundex_encoding(self, strings: list):
        """Given a list of strings, return their Soundex (4) encoding.

        :param strings: Input strings to be encoded
        :return: Soundex (4) encoding of strings
        """
        soundex = fuzzy.Soundex(4)
        soundex_encodings = [soundex(s) for s in strings]
        return soundex_encodings

    def soundex_metric(self, strings: list):
        """Use Levenshtein metric on Soundex encoded strings.

        :param strings: Strings for which Soundex-Levenshtein metric should be returned.
        :return: Levenshtein distance of soundex encoded strings.
        """
        # https://stackoverflow.com/questions/66715423/distance-between-strings-by-similarity-of-sound
        soundex_encoded_strings = self.soundex_encoding(strings)
        return self.levenshtein_metric(soundex_encoded_strings)

    def bool_string_distance(self, strings: list):
        """Use string distance methods to compare a list of strings. If string distance is below similarity defined at
        class instantiation, return True.

        :param strings: Strings to be compared.
        :return: True if distance(all strings) <= self.min_similarity. Otherwise False.
        """
        if self.identity(strings):
            return True
        else:
            lev_metric = self.levenshtein_metric(strings)
            if lev_metric <= self.min_similarity:
                return True
        return False

    def bool_sound_distance(self, strings: list):
        """Use sound distance methods to compare a list of strings. If sound distance is below similarity defined at
        class instantiation, return True.

        :param strings: Strings to be compared.
        :return: True if distance(all strings) <= self.min_similarity. Otherwise False.
        """
        if self.identity(strings):
            return True
        else:
            soundex_metric = self.soundex_metric(strings)
            if soundex_metric <= self.min_similarity:
                return True
        return False