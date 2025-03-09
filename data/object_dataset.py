from dataclasses import dataclass
from typing import List

@dataclass
class PromptObject:
    prompt: str
    ground_truth: str

@dataclass
class LocalityObject:
    relation_specificity: List[PromptObject]

@dataclass
class PortabilityObject:
    reasoning: List[PromptObject]

@dataclass
class JSONObject:
    subject: str
    target_new: str
    prompt: str
    ground_truth: List[str]
    rephrase_prompt: str
    cond: str
    locality: LocalityObject
    portability: PortabilityObject
