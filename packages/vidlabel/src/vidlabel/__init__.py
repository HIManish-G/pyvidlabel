"""vidlabel: an open, directory-based format and library for video annotations."""
from vidlabel_spec import (SPEC_VERSION, FrameIndex, Label, Manifest, Rational, Region, Release, VideoMetadata, VideoSource, View,
                           Vocab)

from . import masks
from .build import Compiled, build
from .dataset import Dataset
from .samples import FrameSamples
from .validate import Issue, ValidationError, validate_dataset

__version__ = "0.5.0"
__all__ = ["Dataset", "FrameSamples", "Compiled", "build", "masks", "Issue", "ValidationError", "validate_dataset", "SPEC_VERSION",
           "FrameIndex", "Label", "Manifest", "Rational", "Region", "Release", "VideoMetadata", "VideoSource", "View", "Vocab"]
