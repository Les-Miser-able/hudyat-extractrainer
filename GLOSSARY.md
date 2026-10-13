# Gesture training

The shared language for recorded gestures and the training examples derived
from them.

## Language

**Landmark sequence**:
An ordered series of detected hand positions representing a gesture over time.

**Video augmentation**:
The creation of additional gesture examples by varying the landmark sequence
derived from a source video.
_Avoid_: Using this term for saving modified video files in this feature.

**Augmented sample**:
A synthetic variation of a source gesture that retains its gesture label and
source identity.

**Source group**:
Related gesture recordings that belong together when separating training,
validation, and test examples.

**Timing variation**:
A variation in how quickly different portions of a gesture progress while
preserving the full movement and its order.
_Avoid_: Using this term for changing playback-rate metadata alone.
