"""vidlabel: an open, directory-based format and library for video annotations."""
from vidlabel_spec import (SPEC_VERSION, Label, Manifest, Rational, Region, Release, VideoMetadata, VideoSource, View,
                           Vocab)

from . import masks
from .build import Compiled, build
from .dataset import Dataset
from .validate import Issue, ValidationError, validate_dataset

__version__ = "0.3.0"
__all__ = ["Dataset", "Compiled", "build", "masks", "Issue", "ValidationError", "validate_dataset", "SPEC_VERSION",
           "Label", "Manifest", "Rational", "Region", "Release", "VideoMetadata", "VideoSource", "View", "Vocab"]
