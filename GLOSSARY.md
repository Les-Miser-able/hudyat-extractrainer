# Gesture training

The shared language for recorded gestures and the training examples derived
from them.

## Language

**Sign vocabulary**:
The set of signs supported for learning and practice. A background class or a
no-hand state is not an additional sign in the vocabulary.

**Category**:
A named subset of the sign vocabulary with its own gesture classifier, such as
Alphabet, Numbers, Colors, or Greetings.

**Recorded source**:
An original gesture image or video, before additional training examples are
derived from it. Multiple augmented examples do not become independent sources.

**Recognition attempt**:
One performance of an expected sign, including attempts that are misrecognized,
not recognized, or interrupted.

**Post-capture feedback delay**:
The time from capturing the final input frame needed to recognize a sign until
the confirmed recognition feedback becomes visible.

**Full camera-to-result time**:
The time from capturing the first submitted frame with a usable hand until the
confirmed recognition feedback becomes visible, including gesture capture.

**Background class**:
A learned category for idle hands or movement outside the sign vocabulary.
It is distinct from observing that no hand was detected.

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
