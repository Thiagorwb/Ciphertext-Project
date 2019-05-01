# coding: utf-8
import multiprocessing as mp
import numpy as np
import scipy as sp
import random
# From Wikipedia:
# > In the mathematics of shuffling playing cards, the Gilbert–Shannon–Reeds model is a probability distribution on riffle shuffle permutations that has been reported to be a good match for experimentally observed outcomes of human shuffling, and that forms the basis for a recommendation that a deck of cards should be riffled seven times in order to thoroughly randomize it. ... The deck of cards is cut into two packets... [t]hen, one card at a time is repeatedly moved from the bottom of one of the packets to the top of the shuffled deck.
# Here we implement the Gilbert–Shannon–Reeds model, and verify this recommendation of seven shuffles.
# Note that the functions below have `doctest` examples.
# To test the functions, just run `pytest` in the top level of the repository.
# First, define a function to determine how many cards to split into our right hand.
def get_random_number_for_right_deck(n, seed = None):
    """
    Return the number of cards to split into the right sub-deck.
    :param n: one above the highest number that could be returned by this
              function.
    :param seed: optional seed for the random number generator to enable
                 deterministic behavior.
    :return: a random integer (between 1 and n-1) that represents the
             desired number of cards.
    Examples:
    >>> get_random_number_for_right_deck(n=5, seed=0, )
    1
    """

    return random.randint(1, n)
# Next, define a function to determine which hand to drop a card from.
def should_drop_from_right_deck(n_left, n_right, seed = None):
    """
    Determine whether we drop a card from the right or left sub-deck.
    Either `n_left` or `n_right` (or both) must be greater than zero.
    :param n_left: the number of cards in the left sub-deck.
    :param n_right: the number of cards in the right sub-deck.
    :param seed: optional seed for the random number generator to
                 enable deterministic behavior.
    :return: True if we should drop a card from the right sub-deck,
             False otherwise.
    Examples:
    >>> should_drop_from_right_deck(n_left=32, n_right=5, seed=0, )
    True
    >>> should_drop_from_right_deck(n_left=0, n_right=5, )
    True
    >>> should_drop_from_right_deck(n_left=7, n_right=0, )
    False
    >>> should_drop_from_right_deck(n_left=0, n_right=0, )
    Traceback (most recent call last):
    ...
    ValueError: Either `n_left` or `n_right` (or both) must be greater than zero.
    """
    if n_left > 0 and n_right > 0:
        # There are cards left in both sub-decks, so pick a
        # sub-deck at random.
        num = random.randint(0, 2)
        boolean = (num == 0)
        return boolean
    elif n_left == 0 and n_right > 0:
        # There are no more cards in the left sub-deck, only
        # the right sub-deck, so we drop from the right sub-deck.
        return True
    elif n_left > 0 and n_right == 0:
        # There are no more cards in the right sub-deck, only
        # the left sub-deck, so we drop from the left sub-deck.
        return False
    else:
        # There are no more cards in either sub-deck.
        raise ValueError ('Either `n_left` or `n_right` '                          '(or both) must be greater than zero.')
# Now we can implement the 'Gilbert–Shannon–Reeds' shuffle.
def shuffle(deck, seed = None):
    """
    Shuffle the input 'deck' using the Gilbert–Shannon–Reeds method.
    :param seq: the input sequence of integers.
    :param seed: optional seed for the random number generator
                 to enable deterministic behavior.
    :return: A new deck containing shuffled integers from the
             input deck.
    Examples:
    >>> shuffle(deck=np.array([0, 7, 3, 8, 4, 9, ]), seed=0, )
    array([4, 8, 3, 7, 0, 9])
    """
    # First randomly divide the 'deck' into 'left' and 'right'
    # 'sub-decks'.
    num_cards_in_deck = len(deck)
    orig_num_cards_right_deck = get_random_number_for_right_deck(
        n=num_cards_in_deck,
        seed=seed,
    )
    # By definition of get_random_number_for_right_deck():
    n_right = orig_num_cards_right_deck
    n_left = num_cards_in_deck - orig_num_cards_right_deck
    shuffled_deck = np.empty(num_cards_in_deck, dtype=int)
    # We will drop a card n times.
    for index in range(num_cards_in_deck):
        drop_from_right_deck = should_drop_from_right_deck(
            n_left=n_left,
            n_right=n_right,
            seed=seed,
        )
        if drop_from_right_deck is True:
            # Drop from the bottom of right sub-deck
            # onto the shuffled pile.
            shuffled_deck[index] = deck[n_right - 1]
            n_right = n_right - 1
        else:
            # Drop from the bottom of left sub-deck
            # onto the shuffled pile.
            shuffled_deck[index] = deck[
                orig_num_cards_right_deck + n_left - 1
            ]
            n_left = n_left - 1
    return shuffled_deck
