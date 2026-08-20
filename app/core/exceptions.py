"""Project-specific exceptions."""


class MusicLabelAssistantError(Exception):
    """Base exception for recoverable project failures."""


class AudioDecodeError(MusicLabelAssistantError):
    """Audio could not be decoded."""


class AudioValidationError(MusicLabelAssistantError):
    """Decoded audio failed validation."""


class ProviderUnavailable(MusicLabelAssistantError):
    """A replaceable model provider is unavailable."""


class ModelIntegrityError(MusicLabelAssistantError):
    """A model file is missing or failed checksum validation."""
