"""Vision module for Kodikon - Person-bag linking and mismatch detection"""

from .baggage_linking import (
    # Enums
    ObjectClass,
    LinkingStatus,
    
    # Dataclasses
    BoundingBox,
    ColorHistogram,
    Detection,
    PersonBagLink,
    BaggageProfile,
    
    # Components
    YOLODetectionEngine,
    EmbeddingExtractor,
    ColorDescriptor,
    PersonBagLinkingEngine,
    HashIDGenerator,
    MismatchDetector,
    DescriptionSearchEngine,
    
    # Main Pipeline
    BaggageLinking,
)

__all__ = [
    'ObjectClass',
    'LinkingStatus',
    'BoundingBox',
    'ColorHistogram',
    'Detection',
    'PersonBagLink',
    'BaggageProfile',
    'YOLODetectionEngine',
    'EmbeddingExtractor',
    'ColorDescriptor',
    'PersonBagLinkingEngine',
    'HashIDGenerator',
    'MismatchDetector',
    'DescriptionSearchEngine',
    'BaggageLinking',
]
