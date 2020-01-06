from __future__ import annotations
import sys
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Set, Optional, Dict, List, Deque, Tuple, FrozenSet
from enum import Enum
assert sys.version_info >= (3, 6)  # requires ordered dicts


class GrammarError(Exception):
    pass


class ModAction(Enum):
    clear = 0
    add = 1
    remove = 2

    def symbol(self) -> str:
        return {
            ModAction.add: '+',
            ModAction.remove: '-',
            ModAction.clear: '*'
        }[self]

    @classmethod
    def from_symbol(cls, symbol: str) -> ModAction:
        return {
            '+': cls.add,
            '-': cls.remove,
            '*': cls.clear
        }[symbol]


@dataclass(frozen=True, eq=True, order=True)
class ModChange:
    action: ModAction
    mod: Optional[str]
    @classmethod
    def clear(cls) -> ModChange:
        return cls(ModAction.clear, None)

    @classmethod
    def add(cls, mod: str) -> ModChange:
        return cls(ModAction.add, mod)

    @classmethod
    def remove(cls, mod: str) -> ModChange:
        return cls(ModAction.remove, mod)

    def __str__(self) -> str:
        return self.action.symbol()+'' if self.mod is None else self.mod

    @classmethod
    def from_str(cls, mod: str) -> ModChange:
        mod_action = ModAction.from_symbol(mod[0])
        if mod_action is ModAction.clear:
            return cls.clear()
        else:
            return cls(mod_action, mod[1:])


class WeightAction(Enum):
    set = 0
    add = 1
    subtract = 2
    multiply = 3
    divide = 4

    def symbol(self) -> str:
        return {
            WeightAction.set: '=',
            WeightAction.add: '+',
            WeightAction.subtract: '-',
            WeightAction.multiply: '*',
            WeightAction.divide: '/'
        }[self]

    @classmethod
    def from_symbol(cls, symbol: str) -> WeightAction:
        return {
            '=': cls.set,
            '+': cls.add,
            '-': cls.subtract,
            '*': cls.multiply,
            '/': cls.divide,
        }[symbol]


@dataclass(frozen=True, eq=True, order=True)
class WeightChange:
    action: WeightAction
    value: int

    @classmethod
    def set(cls, value: int) -> WeightChange:
        return cls(WeightAction.set, value)

    @classmethod
    def add(cls, value: int) -> WeightChange:
        return cls(WeightAction.add, value)

    @classmethod
    def subtract(cls, value: int) -> WeightChange:
        return cls(WeightAction.subtract, value)

    @classmethod
    def multiply(cls, value: int) -> WeightChange:
        return cls(WeightAction.multiply, value)

    @classmethod
    def divide(cls, value: int) -> WeightChange:
        return cls(WeightAction.divide, value)

    def __str__(self) -> str:
        return self.action.symbol()+str(self.value)

    @classmethod
    def from_str(cls, change: str) -> WeightChange:
        action = WeightAction.from_symbol(change[0])
        return cls(action, int(change[1:]))

    def apply(self, weight: int) -> int:
        if self.action is WeightAction.set:
            return self.value
        elif self.action is WeightAction.add:
            return weight + self.value
        elif self.action is WeightAction.subtract:
            return weight - self.value
        elif self.action is WeightAction.multiply:
            return weight * self.value
        elif self.action is WeightAction.divide:
            return weight // self.value
        raise NotImplementedError


@dataclass
class Tag:
    head: str
    mod_changes: Set[ModChange] = field(default_factory=set)

    def canonical(self) -> CanonicalTag:
        return CanonicalTag(
            self.head,
            tuple(sorted(
                    mod.mod for mod in self.mod_changes
                    if mod.action is ModAction.add and mod.mod is not None)))

    def __str__(self):
        return str(self.canonical())

    @classmethod
    def from_str(cls, tag: str) -> Tag:
        unmodified_tag = tag.split('.')[0]
        elements = re.findall('(^[^-+*]*|[-+][^-+*]*|\\*)', tag)
        head = elements[0]
        mods = {ModChange.from_str(mod) for mod in elements[1:]}
        return cls(head, mods)

    def modify(self, *mods: str) -> Tag:
        result: Set[str] = set()
        if ModChange.clear() not in self.mod_changes:
            result.update(mods)
        result.update(
                mod.mod for mod in self.mod_changes
                if mod.action is ModAction.add and mod.mod is not None)
        result.difference_update(
                mod.mod for mod in self.mod_changes
                if mod.action is ModAction.remove)
        return Tag(self.head, {ModChange.add(mod) for mod in result})


@dataclass(frozen=True, eq=True)
class CanonicalTag:
    head: str
    mods: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, 'mods', tuple(sorted(self.mods)))

    def mutable(self) -> Tag:
        return Tag(self.head, {ModChange.add(mod) for mod in self.mods})

    def __str__(self) -> str:
        return '+'.join((self.head,)+self.mods)


class WordLists:
    word_lists: Dict[CanonicalTag, Tuple[str, ...]]
    word_list_mods: Dict[str, FrozenSet[str]]

    def __init__(self, word_lists):
        self.word_lists = {
            Tag.from_str(tag).canonical(): sorted(words)
            for (tag, words) in word_lists.items()
        }
        word_list_mods = defaultdict(set)
        for tag in self.word_lists:
            word_list_mods[tag.head].update(tag.mods)
        self.word_list_mods = {
                tag: frozenset(mods)
                for (tag, mods) in word_list_mods.items()}


class GrammarStructure:
    word_lists: WordLists
    structure: Dict[str, Dict[str, Dict[Optional[str], WeightChange]]]

    def __init__(self, grammar, word_lists):
        self.word_lists = WordLists(word_lists)
        self.structure = {
            tag: {
                production: self.normalise_mods(mods)
                for (production, mods) in productions.items()
            }
            for (tag, productions) in grammar.items()
        }

    # Expand abbreviated productions in input grammar
    def normalise_mods(self, mods) -> Dict[Optional[str], WeightChange]:
        if isinstance(mods, int):
            return {None: WeightChange.set(1)}
        normalised_mods: Dict[Optional[str], WeightChange] = {}
        if None not in mods:
            normalised_mods[None] = WeightChange.set(1)
        for (mod, change) in mods.items():
            if isinstance(change, int):
                normalised_mods[mod] = WeightChange.set(change)
            else:
                normalised_mods[mod] = WeightChange.from_str(change)
        return normalised_mods

    # generate standard tracery code
    def generate(self, origin: str = 'origin') -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        origin_tag = CanonicalTag(origin, tuple())
        tag_queue: Deque[Tuple[CanonicalTag, Tuple[CanonicalTag, ...]]] = (
                deque([(origin_tag, ())]))
        tags_queued: Set[CanonicalTag] = {origin_tag}
        while len(tag_queue) > 0:
            tag, history = tag_queue.pop()
            if tag.head in self.structure:
                rules, new_tags = self.process_rule(tag)
                result[str(tag)] = rules
                for t in new_tags:
                    if t not in tags_queued:
                        tag_queue.appendleft((t, history + (tag,)))
                        tags_queued.add(t)
            else:
                try:
                    words = self.process_words(tag)
                except KeyError:
                    raise GrammarError('\n' + '\n'.join(
                            str(t) for t in history + (tag,)))
                result[str(tag)] = words
        return result

    # return modified and weighted rule list for standard tracery grammar,
    # and tags called by those rules
    def process_rule(self, tag: CanonicalTag) -> (
            Tuple[List[str], Set[CanonicalTag]]):
        productions = self.structure[tag.head]
        rules: List[str] = []
        all_new_tags: Set[CanonicalTag] = set()
        for production, mod_effects in productions.items():
            production_updated, new_tags = self.apply_mods(
                    production, tag.mods)
            weight = self.get_weight(tag.mods, mod_effects)
            if weight > 0:
                rules.extend([production_updated]*weight)
                all_new_tags.update(new_tags)
        return rules, all_new_tags

    # return probability weight of a production
    # given a collection of active mods
    def get_weight(self, mods, mod_effects):
        value = 0
        for mod, effect in mod_effects.items():
            if mod is None or mod in mods:
                value = effect.apply(value)
        return value

    # return production with mods applied to all tag calls,
    # and collection of tags called
    def apply_mods(self, production: str, mods: Tuple[str, ...]) -> (
            Tuple[str, Tuple[CanonicalTag, ...]]):
        tag_strings = self.find_tags(production)
        tags_split = [tag.split('.') for tag in tag_strings]
        tags = [Tag.from_str(tag[0]) for tag in tags_split]
        tags_modified = [tag.modify(*mods).canonical() for tag in tags]
        tags_modified_reduced = tuple(
            tag if tag.head in self.structure
            else self.reduce_wordlist_tag(tag)
            for tag in tags_modified)
        tag_strings_updated = [str(tag) for tag in tags_modified_reduced]
        tags_with_modifiers = [
            '.'.join([tag]+modifiers[1:])
            for tag, modifiers in zip(tag_strings_updated, tags_split)
        ]
        production = self.replace_tags(
                production, tag_strings, tags_with_modifiers)
        return production, tags_modified_reduced

    def reduce_wordlist_tag(self, tag: CanonicalTag):
        allowable_mods = self.word_lists.word_list_mods[tag.head]
        mods = set(tag.mods).intersection(allowable_mods)
        return Tag(tag.head, {ModChange.add(mod) for mod in mods}).canonical()

    # finds the correct wordlist for a tag,
    # ignoring any mods irrelevant to the word list
    def process_words(self, tag: CanonicalTag):
        return self.word_lists.word_lists[self.reduce_wordlist_tag(tag)]

    # substitute tag calls found in a string from tags to modded_tags
    def replace_tags(self,
                     string: str,
                     tags: List[str],
                     modded_tags: List[str]) -> str:
        for (tag, modded) in zip(tags, modded_tags):
            string = string.replace(f'#{tag}#', f'#{modded}#')
        return string

    # extract all tracery tag calls from a string
    def find_tags(self, string: str) -> List[str]:
        # pattern = '#([^#.]*)(?:\.[^#]*)*#' # strips tracery modifiers
        pattern = '#([^#]*)#'
        return re.findall(pattern, string)
