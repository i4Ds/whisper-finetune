import os

from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    AddGaussianSNR,
    AirAbsorption,
    Aliasing,
    BandPassFilter,
    BandStopFilter,
    BitCrush,
    ClippingDistortion,
    Compose,
    Gain,
    GainTransition,
    HighPassFilter,
    HighShelfFilter,
    LoudnessNormalization,
    LowPassFilter,
    LowShelfFilter,
    Mp3Compression,
    OneOf,
    PeakingFilter,
    PitchShift,
    RoomSimulator,
    Shift,
    TimeStretch,
)


def get_audio_augments_baseline(min_rate: float = 0.8, max_rate: float = 1.25):
    """
    Baseline augmentation pipeline with TimeStretch only.
    
    Args:
        min_rate: Minimum time-stretch rate (e.g., 0.8 = 20% slower)
        max_rate: Maximum time-stretch rate (e.g., 1.25 = 25% faster)
    """
    return Compose([
        TimeStretch(
            min_rate=min_rate,
            max_rate=max_rate,
            leave_length_unchanged=False,
            p=1.0,
        ),
    ])


def get_audio_augments_advanced():
    """
    Advanced augmentation pipeline with background noise, filters, gain changes, 
    pitch shifts, and other audio effects.
    """
    current_dir = os.path.dirname(__file__)
    transforms = [
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
    return Compose(transforms)


def get_audio_augments_office():
    """
    Augmentation pipeline that adds:
      • RoomSimulator tuned for a carpeted office
      • Mp3Compression (8-64 kbps) to mimic uploads from low-quality mics
      • BitCrush (6-14 bits) for cheap-ADC grit
    All transforms keep the sample count unchanged by default.
    """

    office_reverb = OneOf(
        [
            RoomSimulator(
                # Small-ish room (≈ 4 m × 3 m × 2.7 m)
                min_size_x=3.0,
                max_size_x=5.0,
                min_size_y=2.5,
                max_size_y=4.0,
                min_size_z=2.4,
                max_size_z=3.0,
                # Carpeted surfaces → absorption 0.10-0.20  (office/library value)
                calculation_mode="absorption",
                min_absorption_value=0.05,
                max_absorption_value=0.20,  # :contentReference[oaicite:0]{index=0}
                # Chop tail so clip length stays intact
                leave_length_unchanged=True,
                max_order=3,
                p=1.0,
            ),
        ],
        p=0.5,
    )

    lo_fi_codecs = OneOf(
        [
            Mp3Compression(
                min_bitrate=8, max_bitrate=64, backend="pydub", p=1.0
            ),  # :contentReference[oaicite:1]{index=1}
            BitCrush(min_bit_depth=6, max_bit_depth=14, p=1.0),  # :contentReference[oaicite:2]{index=2}
        ],
        p=0.5,
    )

    return Compose([lo_fi_codecs, office_reverb])


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import librosa
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Apply random audiomentations to a single file")
    parser.add_argument("infile", type=Path, help="Input audio file (wav/mp3/…)")
    parser.add_argument(
        "--out",
        dest="outfile",
        type=str,
        default=None,
        help="Output path (defaults to <in>_aug.wav)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16_000,
        help="Target sample-rate (will resample if different). Default 16000",
    )
    args = parser.parse_args()

    # ---------- load ----------
    samples, sr = sf.read(args.infile, always_2d=False)
    if sr != args.sr:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=args.sr)

    # ---------- augment ----------
    augment_office = get_audio_augments_office()
    augment_baseline = get_audio_augments_baseline(min_rate=0.8, max_rate=1.25)
    augment_advanced = get_audio_augments_advanced()
    augment = Compose([augment_office, augment_baseline, augment_advanced], p=1.0)
    augmented = augment(samples=samples, sample_rate=args.sr)

    # ---------- save ----------
    out_path = Path(args.outfile or args.infile.stem + "_aug.wav")
    sf.write(out_path, augmented, args.sr)
    print(f"✓ Augmented audio written to {out_path.resolve()}")
