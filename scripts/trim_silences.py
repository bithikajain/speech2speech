#!/usr/bin/env python
#
# Copyright (C) 2020 Bithika Jain
#
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""Trim silence in audio files and optionally normalize them. Can be asymmetric.
By default leaves 150ms of silence at beginning and end.  Meant for mono audio;
if given stereo, silently converts to mono.  Window size and hop length of
audio level calculating function may work best at 44.1kHz.  Experiment if using
other sample rates.

The audio is saved in the same format as it was read in (ogg, wav, flac, or mp3).
However, the target subtype is the default for sf.write for that format.  So, for
instance, wav will be saved in PCM_16 (even if the original was float32).

The script can also normalize audio in several way and remove clicks from the
end.  Note that normalized audio can be denormalized by up to about 10% (i.e.,
max abs.  value can become up to around 1.1 or also significantly less than 1)
by resampling.

A common data prep path would be to first run this script once on all the source
audio (passing all.csv to facilitate finding bad trims), and start an exclusions
file with all of the bad trims.  This exclusions file will be updated as you get
new target data, but the original data is kept, so at some point you could decide
to redo silence trimming from scratch, making a new exclusions file.  (So,
ideally you should keep two exclusions files: one for files with unrecoverable
audio problems in the source or target, such as bad splits, and one for bad
silence trimming, which could be fixed by changing silence trimming parameters.
You would always use the concatenation of these exclusions files.  Or, another
possibility would be to just delete the files with audio problems.)  Then, every
time new target audio is available, run trim_silences, passing the current
exclusions file and the all.csv file, and check the warnings to add bad trims to
a new exclusions file.  Then run export_partial_dataset and have a human proof the
audio.  Problematic audio can be added to an exclusions file (but if it is
possible to ask the talent to rerecord it, later it can be removed from the
exclusions file).  The export_partial_dataset can be rerun with the new exclusion.
From there, we just run split_dataset and proceed to feature extraction.

By the way, we use simple energy-based silence detection, but there are fancier
alternatives that may work better.  For instance, here is Kyle Kastner in a Hacker
News comment:


    It may be possible to do this with an LSTD VAD, I always had really good
    luck with that. I tried a few random ones in here for silence removal - no
    quality guarantee [0]

    I found LTSD pretty robust compared to simpler energy based things as long
    as you have a small chunk of background sound at the start. The LTSD
    implementation is largely from my friend Joao, so I can't take credit for
    the cool part, only the bugs

    [0] https://gist.github.com/kastnerkyle/a3661d6be10a0ae9e01fd429...

"""

import argparse
import os
import re
import sys

import librosa
import numpy as np
from scipy.signal import firwin
from scipy.signal import lfilter
import soundfile as sf

EPS = 1e-10


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio_dir', type=str,
                        help="Directory with audio files")
    parser.add_argument('target_dir', type=str,
                        help="Place to save trimmed audio")
    parser.add_argument('--all_csv', type=str,
                        help="Name of an all.csv file in which to look up files to"
                        " get line numbers where they occur.  This allows easy checking"
                        " of suspiciously trimmed files in parallel_recorder",
                        default=None)
    parser.add_argument('--exclusions', type=str,
                        help="Name of a directory of exclusions files with paths"
                        " (like CD1/10_33.ogg) to ignore",
                        default=None)
    parser.add_argument('--sr', type=int,
                        help="The input audio will be interpolated to this sample"
                        " rate, which must be 44100 or 48000.  If left unset,"
                        " audio at 44100 or 48000 will be left as is while audio at"
                        " any other sample rate will be resampled to 44100",
                        default=None)
    parser.add_argument('--output_sr', type=int,
                        help="Save the audio with this samplerate",
                        default=None)
    parser.add_argument('--trim_silence_db', type=float,
                        help="The threshold (in decibels) below reference to"
                             " consider as silence and trim it. This value if"
                             " ignored if trim_silence_db_{begin/end} are"
                             " specified",
                        default=30)
    parser.add_argument('--trim_silence_db_begin', type=float,
                        help="Same as trim_silence_db, but applied only to the"
                        " beginning",
                        default=None)
    parser.add_argument('--trim_silence_db_end', type=float,
                        help="Same as trim_silence_db, but applied only to the"
                        " end",
                        default=None)
    # Eliminating this option since it probably is almost never good to use, and
    # providing these subformats (or at least PCM_16) apparently doesn't work with ogg
    #
    # parser.add_argument('--save_as_float32', action='store_true',
    #                     help="By default, we save wav files as PCM_16 bit format."
    #                     " This is a standard format and should be preferred,"
    #                     " because using float32 format might cause problems with"
    #                     " some players and PyTorchWavenetVocoder")
    parser.add_argument('--backoff', type=float, default=150,
                        help="number of milliseconds of silence to leave at"
                        " beginning and end.  Ignored if backoff_{begin/end} are"
                        " specified.")
    parser.add_argument('--backoff_begin', type=float, default=None,
                        help="like backoff but applied only to the beginning")
    parser.add_argument('--backoff_end', type=float, default=None,
                        help="like backoff but applied only to the end")
    parser.add_argument('--low_cut_cutoff', type=float, default=150.,
                        help="filter out frequencies below this before determining"
                        " trim points.  Note that the output signal itself is not"
                        " filtered.")
    parser.add_argument('--normalize', action='store_true',
                        help="Whether to scale the signal to have max abs. value 1")
    parser.add_argument('--flipping_normalize', action='store_true',
                        help="Whether to scale and possibly flip the signal to have"
                        " max abs. value 1 and min value -1")
    parser.add_argument('--ultra_normalize', action='store_true',
                        help="Whether to scale and shift the signal to fill [-1,1]"
                        " (at the cost of introducing DC bias)")
    parser.add_argument('--declick_ms', type=float, default=None,
                        help="If given, this many milliseconds will be removed from"
                        " the end prior to silence trimming.  Useful for getting rid"
                        " of clicks that come from pressing the button to stop recording")
    parser.add_argument('--declick_safety_ms', type=float, default=100.,
                        help="If declick_ms is not None, print warning message if less than"
                        " declick_safety_ms milliseconds of audio are trimmed from the end."
                        " This helps ensure that declicking is not cutting off legit audio.")

    args = parser.parse_args()

    # Commenting out this check since currently we have non-None default for trim_silence_db,
    # which cannot be made None from command line
    # if (args.trim_silence_db is None and
    #     args.trim_silence_db_begin is None and
    #     args.trim_silence_db_end is None):
    #     sys.exit("At least one of the trim_silence_db* arguments must be specified")

    if args.sr is not None and args.sr != 44100 and args.sr != 48000:
        sys.exit("Error! This script uses librosa's default trimming window size and"
                 " hop length of 2048 and 512 samples.  This works well for audio at"
                 " 44.1kHz or 48kHz but may not for audio at other sample rates.")

    if args.normalize and args.ultra_normalize:
        sys.exit("Can only use one of `normalize` and `ultra_normalize`")

    return args


# Taken from feature_extract.py in kb-wavenet.  However, we use a length of
# 2047 instead of 255.  This gives better suppression of unwanted frequencies.
# Also, the signal is delayed by about 1024 samples.  I guess this doesn't matter
# though since we will just silence trim anyway.
def low_cut_filter(x, fs, cutoff, length=2047):
    """FUNCTION TO APPLY LOW CUT FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(length, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def normalize_(au, normalization_type):
    '''Normalize au in place'''

    max_val = au.max()
    minus_min_val = -au.min()
    assert max_val > 0 and minus_min_val > 0

    if normalization_type == 'normal' or normalization_type == 'flipping':
        au /= max(max_val, minus_min_val)
        if normalization_type == 'flipping':
            if minus_min_val < max_val:
                au *= -1.

    elif normalization_type == 'ultra':
        au /= .5 * (max_val + minus_min_val)
        au -= (au.max() - 1.)
    else:
        assert normalization_type is None


def asymmetric_trim(au, sr, trim_db, trim_db_begin, trim_db_end, backoff,
                    backoff_begin, backoff_end, low_cut_cutoff, base_name=None, lookup_file=None):
    filtered = low_cut_filter(au, sr, low_cut_cutoff)
    assert au.shape == filtered.shape
    # assuming at least one of the arguments is not None
    if trim_db_begin is None and trim_db_end is None:
        _, (begin, end) = librosa.effects.trim(filtered, trim_db)
    else:
        # if only one is given, don't trim the other
        trim_db_begin = np.inf if trim_db_begin is None else trim_db_begin
        trim_db_end = np.inf if trim_db_end is None else trim_db_end
        _, (begin, _) = librosa.effects.trim(filtered, trim_db_begin)
        _, (_, end) = librosa.effects.trim(filtered, trim_db_end)

    end_trimmed_ms = (len(au) - end) / sr * 1000

    if backoff_begin is None and backoff_end is None:
        backoff_begin = backoff_end = backoff
    else:
        if backoff_begin is None:
            backoff_begin = 0.
        if backoff_end is None:
            backoff_end = 0.

    backoff_begin = int(backoff_begin * sr / 1000)
    backoff_end = int(backoff_end * sr / 1000)

    au_len = len(au)
    trimmed_au = au[max(0, begin - backoff_begin): min(au_len, end + backoff_end)]

    # Now make sure that we have exactly backoff_begin and backoff_end amount of
    # silence by adding zeros if there was not enough natural silence in the
    # recording

    if begin - backoff_begin < 0:
        if (base_name and lookup_file):
            print('adding {}ms at left to {} ({}) '.format(
                int((backoff_begin - begin) / sr * 1000),
                base_name,
                lookup_file(base_name)
            ))
        trimmed_au = np.concatenate(
            (np.zeros(backoff_begin - begin), trimmed_au))

    if au_len < end + backoff_end:
        # Printing commented out because adding to the right is very common and not as important as
        # adding to the left
        # print('adding {}ms at right to {} ({}) '.format(
        #     int((end + backoff_end - au_len) / sr * 1000),
        #     base_name,
        #     lookup_file(base_name)
        # ))
        trimmed_au = np.concatenate(
            (trimmed_au, np.zeros(end + backoff_end - au_len)))

    return trimmed_au, end_trimmed_ms


def get_lookup_file(all_csv):
    if all_csv is None:
        return lambda x: ''

    lookup_dict = dict()
    with open(all_csv, 'r') as f:
        for zero_based_line_number, line in enumerate(f):
            vals = line.split(',')
            assert len(vals) == 2
            fname = vals[1].strip()
            lookup_dict[fname] = zero_based_line_number + 1

    def lookup_file(fname):
        prefix, last_component = os.path.split(fname)
        prefix, penultimate_component = os.path.split(prefix)
        last_two_components = os.path.join(
            penultimate_component, last_component)
        if os.path.split(prefix)[0] != '':
            sys.exit("Files audio files should be in folders by CD number either directly or under"
                     " one more level of directory hierarchy, namely the speaker (in case you are"
                     " trimming a whole exported dataset).  But in this case we got an extra level"
                     " of directory hierarchy.  Problematic path: {}".format(fname))

        if last_two_components not in lookup_dict:
            sys.exit("File {} not found in {}!".format(
                last_two_components, all_csv))

        return lookup_dict[last_two_components]

    return lookup_file


def main():
    args = get_args()
    lookup_file = get_lookup_file(args.all_csv)
    exclusions = set()
    if args.exclusions is not None:
        for root, dirs, files in os.walk(args.exclusions):
            for file in files:
                with open(os.path.join(root, file), 'r') as f:
                    for base_name in f:
                        exclusions.add(base_name.strip())

    fnames = []

    for root, dirs, files in os.walk(args.audio_dir, followlinks=True):
        for f in files:
            if bool(re.search(r'\.(ogg|wav|mp3|flac)$', f)):
                fnames.append(os.path.join(root, f))

    print("Found {} files to process".format(len(fnames)))

    if os.path.exists(args.target_dir):
        sys.exit(
            "the target directory '{}' already exists!".format(args.target_dir))

    os.makedirs(args.target_dir)
    with open(os.path.join(args.target_dir, 'trim_silences_args.txt'), 'w') as f:
        print(args, file=f)

    n_files = len(fnames)

    for i, fname in enumerate(fnames):
        base_name = os.path.relpath(fname, args.audio_dir)
        if base_name in exclusions:
            continue

        au, sr = librosa.load(fname, mono=True, sr=args.sr or None)
        if sr != 44100 and sr != 48000:
            print(
                "Your audio is at sample rate {}.  Will resample to 44.1kHz".format(sr))
            au, sr = librosa.load(fname, mono=True, sr=44100)

        if args.declick_ms is not None:
            au = au[:int(-args.declick_ms * sr / 1000)]

        if args.normalize:
            normalization_type = 'normal'
        elif args.flipping_normalize:
            normalization_type = 'flipping'
        elif args.ultra_normalize:
            normalization_type = 'ultra'
        else:
            normalization_type = None

        # Normalize before in the hope that it makes trimming computation more reliable
        # (probably it doesn't make any difference though)
        normalize_(au, normalization_type)

        trimmed_au, end_trimmed_ms = asymmetric_trim(
            au,
            sr,
            args.trim_silence_db,
            args.trim_silence_db_begin,
            args.trim_silence_db_end,
            args.backoff,
            args.backoff_begin,
            args.backoff_end,
            args.low_cut_cutoff,
            base_name=base_name,
            lookup_file=lookup_file
        )

        if args.declick_ms is not None and end_trimmed_ms < args.declick_safety_ms:
            print("Warning! {} ({}) has only {}ms of safety silence".format(
                base_name,
                lookup_file(base_name),
                int(end_trimmed_ms)
            ))
            with open(os.path.join(args.target_dir, 'declick_warnings.txt'), 'a+') as f:
                print("{},{}".format(fname, end_trimmed_ms), file=f)

        # Normalize after so end result is always perfectly normalized
        # (probably it would be perfectly normalized even without this though)
        normalize_(trimmed_au, normalization_type)

        target_file = os.path.join(args.target_dir, base_name)

        progress = i / n_files * 100
        print('{} ({}) -> {} [{:.1f}%]\r'.format(base_name, lookup_file(base_name), target_file, progress),
              end='', flush=True)

        if os.path.dirname(target_file) != '':
            os.makedirs(os.path.dirname(target_file), exist_ok=True)

        if args.output_sr is not None:
            trimmed_au = librosa.resample(trimmed_au, sr, args.output_sr)
            sr = args.output_sr

        sf.write(target_file, trimmed_au, sr)

    print('\ndone')


if __name__ == '__main__':
    main()
