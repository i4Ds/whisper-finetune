import os
from audiomentations import (
    AddBackgroundNoise,
    OneOf,
    Compose,
    Aliasing,
    AddGaussianNoise,
    LoudnessNormalization,
    Gain,
    GainTransition,
    BandPassFilter,
    BandStopFilter,
    AddGaussianSNR,
    LowPassFilter,
    LowShelfFilter,
    HighPassFilter,
    HighShelfFilter,
    PitchShift,
    Shift,
    ClippingDistortion,
    AirAbsorption,
    PeakingFilter,
)


def get_audio_augments_baseline():
    current_dir = os.path.dirname(__file__)
    augment = Compose(
        [
            OneOf(
                [
                    AddBackgroundNoise(
                        sounds_path=os.path.join(current_dir, "bg_noise"),
                        noise_rms="absolute",
                        min_absolute_rms_db=-30,
                        max_absolute_rms_db=-10,
                    ),
                    AddBackgroundNoise(
                        sounds_path=os.path.join(current_dir, "bg_noise"),
                        min_snr_db=2,
                        max_snr_db=4,
                    ),
                ],
                p=0.3,
            ),
            OneOf(
                [
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
                    AddGaussianSNR(min_snr_db=5.0, max_snr_db=40.0, p=1.0),
                    LoudnessNormalization(p=1.0),
                    Aliasing(p=1.0),
                ],
                p=0.3,
            ),
            OneOf(
                [
                    LowPassFilter(p=1.0),
                    LowShelfFilter(p=1.0),
                    HighPassFilter(p=1.0),
                    HighShelfFilter(p=1.0),
                    BandPassFilter(p=1.0),
                    BandStopFilter(p=1.0),
                    ClippingDistortion(p=0.8),
                    AirAbsorption(p=0.8),
                    PeakingFilter(p=0.8),
                ],
                p=0.6,
            ),
            OneOf(
                [
                    Gain(min_gain_db=-6.0, max_gain_db=6.0, p=1.0),
                    GainTransition(p=1.0),
                    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                    Shift(p=0.5),
                ],
                p=0.3,
            ),
        ]
    )
    return augment