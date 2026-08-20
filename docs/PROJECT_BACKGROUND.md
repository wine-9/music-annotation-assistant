# Project background

Music Annotation Assistant began as an internal productivity tool for repetitive music-audio annotation. A reviewer had to listen to a full mix, identify likely instruments, estimate where they appeared, inspect stereo characteristics, and transfer those observations into structured fields.

The project evolved into a local-first pre-labeling pipeline. Its design favors conservative automation: expose model scores, preserve unknowns, retain raw predictions, and keep a human review trail. This remains useful even when recall is imperfect, because high-confidence candidates can reduce repetitive work without pretending the model fully understands a dense mix.

The open-source edition is deliberately separated from private work data. It contains no private annotation corpus, no third-party platform capture archive, and no checkpoint trained on private data. Public evaluation and calibration use documented public datasets and upstream pretrained models.
