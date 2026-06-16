from .types import CognitiveResult, EmotionResult, EngagementLevel, MappingConfig


class EngagementMapper:
    def __init__(self, mappingRules: MappingConfig | None = None):
        self.mappingRules = mappingRules or MappingConfig()

    def map(self, emotion: EmotionResult, cognitive: CognitiveResult) -> EngagementLevel:
        score = (emotion.confidence + cognitive.score) / 2.0
        levels = self.mappingRules.levels
        if score >= 0.75:
            label = levels[3]
        elif score >= 0.50:
            label = levels[2]
        elif score >= 0.25:
            label = levels[1]
        else:
            label = levels[0]
        return EngagementLevel(label=label, score=score)
