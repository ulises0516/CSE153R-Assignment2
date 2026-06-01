"""
task2_model.py
==============
Task 2: Symbolic Conditioned Generation — Melody Harmonization
Owner: Ulises

Pipeline:
  Input  : 8-note melody window, each note = (pitch, dur_bin)
  Output : (chord_root, chord_quality) label  +  chord duration bin

Components:
  build_chord_label_vocab   — build (root, quality) → id mapping from windows
  HarmonizationDataset      — encode windows into tensors
  BahdanauAttention         — additive attention for decoder
  Encoder                   — Bidirectional LSTM over melody tokens
  Decoder                   — attention-conditioned LSTM with two output heads
  Seq2SeqHarmonizer         — wraps Encoder + one-step attention decoder
  train_epoch_t2 / evaluate_t2 / train_model_t2
  harmonize                 — greedy chord prediction for melody windows
  harmonization_to_midi     — render melody + predicted chords to .mid
"""

import os
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pretty_midi

# ── Constants (must match §2.1 preprocessing) ───────────────────────────────
ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
QUALITIES = ['major', 'minor', 'diminished', 'augmented', 'dominant7']
DURATION_BINS_BEATS = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]

# Chord tone intervals (semitones above root) for MIDI rendering
CHORD_INTERVALS = {
    'major':      [0, 4, 7],
    'minor':      [0, 3, 7],
    'diminished': [0, 3, 6],
    'augmented':  [0, 4, 8],
    'dominant7':  [0, 4, 7, 10],
}

# MIDI pitch for each root at octave 4 (C4 = 60)
ROOT_TO_PITCH = {r: 60 + i for i, r in enumerate(ROOTS)}
ROOT_TO_PITCH.update({'Db': 61, 'Eb': 63, 'Gb': 66, 'Ab': 68, 'Bb': 70})

SEED = 42


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. VOCABULARY                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_chord_label_vocab(windows):
    """
    Build a (root, quality) → int mapping from the provided training windows.
    Duration is handled separately (directly as bin index 0-5).
    Returns chord_label_vocab, id2chord_label.
    """
    pairs = sorted({(root, qual) for _, root, qual, *_ in windows})
    vocab = {pair: i for i, pair in enumerate(pairs)}
    id2vocab = {i: pair for pair, i in vocab.items()}
    return vocab, id2vocab


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. DATASET                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class HarmonizationDataset(Dataset):
    """
    Converts (melody_window, root, quality, dur_bin) tuples from §2.1 into
    batched tensors.

    Each sample:
        melody_ids    : LongTensor (window_size,)   — encoded (pitch, dur_bin) tokens
        chord_label   : LongTensor scalar            — (root, quality) id
        chord_dur     : LongTensor scalar            — dur_bin 0-5

    Special tokens in melody_vocab (e.g. <PAD>) are mapped to 0 for any
    melody token not seen in the vocabulary.
    """

    def __init__(self, windows, melody_vocab, chord_label_vocab):
        self.samples = []
        pad_id = melody_vocab.get('<PAD>', 0)

        for melody_window, root, quality, dur_bin, *_ in windows:
            melody_ids = [melody_vocab.get(tok, pad_id) for tok in melody_window]
            label_id   = chord_label_vocab.get((root, quality))
            if label_id is None:
                continue  # skip unseen chord labels

            self.samples.append((
                torch.tensor(melody_ids, dtype=torch.long),
                torch.tensor(label_id,   dtype=torch.long),
                torch.tensor(dur_bin,    dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. ATTENTION                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention.

    score(h_enc_t, s_dec) = v^T tanh(W_enc * h_enc_t + W_dec * s_dec)
    context = Σ softmax(scores) * encoder_outputs

    Why attention instead of a fixed context vector?
    A fixed vector (last encoder hidden state) compresses the entire melody
    into a single vector — information about individual notes is lost.
    Attention lets the decoder query which melody notes are most relevant
    for each window-level chord prediction instead of relying only on a
    single pooled encoder vector.
    """

    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int):
        super().__init__()
        self.W_enc = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.W_dec = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.v     = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            encoder_outputs : (batch, src_len, encoder_hidden_dim)
            decoder_hidden  : (batch, decoder_hidden_dim)  — top-layer decoder h_t

        Returns:
            context         : (batch, encoder_hidden_dim)
            attn_weights    : (batch, src_len)
        """
        # Expand decoder hidden to match encoder time dimension
        dec_exp = decoder_hidden.unsqueeze(1)                          # (B, 1, D)
        energy  = torch.tanh(self.W_enc(encoder_outputs) +
                             self.W_dec(dec_exp))                      # (B, T, D)
        scores  = self.v(energy).squeeze(-1)                           # (B, T)
        weights = torch.softmax(scores, dim=-1)                        # (B, T)
        context = torch.bmm(weights.unsqueeze(1),
                            encoder_outputs).squeeze(1)                # (B, enc_dim)
        return context, weights


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. ENCODER                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class Encoder(nn.Module):
    """
    Bidirectional LSTM encoder over a melody window.

    Why bidirectional?
    Chord harmony is determined by notes before AND after the current position.
    For example, the leading tone resolves forward — a unidirectional encoder
    cannot see this. A bidirectional encoder processes the window in both
    directions so every encoder state has full melodic context.

    Output encoder_hidden_dim = 2 * hidden_dim (concat fwd + bwd).
    """

    def __init__(
        self,
        melody_vocab_size: int,
        embed_dim:  int = 64,
        hidden_dim: int = 128,
        decoder_hidden_dim: int = 256,
        num_layers: int = 2,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(melody_vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Project concatenated bidirectional final state to decoder init size.
        self.fc_h = nn.Linear(hidden_dim * 2, decoder_hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * 2, decoder_hidden_dim)

    def forward(self, melody_ids):
        """
        Args:
            melody_ids : (batch, window_size)

        Returns:
            encoder_outputs : (batch, window_size, hidden_dim*2)
            (h_dec_init, c_dec_init) : each (1, batch, decoder_hidden_dim)
        """
        emb = self.dropout(self.embedding(melody_ids))    # (B, T, E)
        enc_out, (h_n, c_n) = self.lstm(emb)             # enc_out: (B, T, 2H)

        # h_n shape: (num_layers*2, B, H)
        # Take final layer forward [−2] and backward [−1], concatenate
        h_fwd = h_n[-2]                                  # (B, H)
        h_bwd = h_n[-1]                                  # (B, H)
        c_fwd = c_n[-2]
        c_bwd = c_n[-1]

        h_dec = torch.tanh(self.fc_h(torch.cat([h_fwd, h_bwd], dim=-1)))  # (B, 256)
        c_dec = torch.tanh(self.fc_c(torch.cat([c_fwd, c_bwd], dim=-1)))

        return enc_out, (h_dec.unsqueeze(0), c_dec.unsqueeze(0))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5. DECODER                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class Decoder(nn.Module):
    """
    Single-step attention decoder with two output heads.

    Input: [learned query embedding ‖ attention context]
    Hidden state: 1-layer LSTM with 256 units

    Two output heads — separating chord label from duration has two advantages:
      1. Each head has a smaller, more focused vocabulary
      2. Label accuracy and rhythm accuracy can be evaluated independently
    """

    def __init__(
        self,
        chord_label_vocab_size: int,
        chord_dur_vocab_size:   int = 6,
        embed_dim:              int = 64,
        hidden_dim:             int = 256,
        encoder_hidden_dim:     int = 256,   # 2 * encoder hidden_dim (bidirectional)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # The extra embedding is a dedicated query token. It is not a chord
        # class and is therefore excluded from the output head.
        self.start_id   = chord_label_vocab_size
        self.embedding = nn.Embedding(chord_label_vocab_size + 1, embed_dim)
        self.attention  = BahdanauAttention(encoder_hidden_dim, hidden_dim)

        # LSTM input = chord embedding + attention context
        self.lstm = nn.LSTM(
            embed_dim + encoder_hidden_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)

        # Head 1: predict chord root+quality label
        self.head_label    = nn.Linear(hidden_dim, chord_label_vocab_size)
        # Head 2: predict duration bin (0-5)
        self.head_duration = nn.Linear(hidden_dim, chord_dur_vocab_size)

    def forward(self, prev_chord_id, decoder_hidden, encoder_outputs):
        """
        One decoder step.

        Args:
            prev_chord_id    : (batch,)          — learned decoder query token id
            decoder_hidden   : tuple (h, c) each (1, batch, 256)
            encoder_outputs  : (batch, src_len, enc_dim)

        Returns:
            label_logits  : (batch, chord_label_vocab_size)
            dur_logits    : (batch, 6)
            decoder_hidden: updated (h, c)
        """
        emb = self.dropout(self.embedding(prev_chord_id))   # (B, embed_dim)

        # Attention: use top-layer decoder hidden state
        h_top   = decoder_hidden[0][-1]                     # (B, 256)
        context, _ = self.attention(encoder_outputs, h_top) # (B, enc_dim)

        lstm_in = torch.cat([emb, context], dim=-1).unsqueeze(1)  # (B, 1, E+enc)
        out, decoder_hidden = self.lstm(lstm_in, decoder_hidden)   # (B, 1, 256)
        out = self.dropout(out.squeeze(1))                          # (B, 256)

        label_logits = self.head_label(out)                         # (B, C_label)
        dur_logits   = self.head_duration(out)                      # (B, 6)
        return label_logits, dur_logits, decoder_hidden


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6. SEQ2SEQ WRAPPER                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class Seq2SeqHarmonizer(nn.Module):
    """
    Wires Encoder → Decoder for one melody window → one chord prediction.

    Each melody window maps to exactly ONE majority-overlap chord. The decoder
    therefore runs for a single step and acts as an attention-conditioned
    classifier. It does not claim to model chord-to-chord transitions.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, melody_ids):
        """
        Args:
            melody_ids           : (batch, window_size)
        Returns:
            label_logits : (batch, chord_label_vocab_size)
            dur_logits   : (batch, 6)
        """
        enc_out, dec_hidden = self.encoder(melody_ids)

        batch_size = melody_ids.size(0)
        query = torch.full(
            (batch_size,),
            self.decoder.start_id,
            dtype=torch.long,
            device=melody_ids.device,
        )
        label_logits, dur_logits, _ = self.decoder(query, dec_hidden, enc_out)
        return label_logits, dur_logits


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7. TRAINING                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_epoch_t2(model, loader, optimizer, criterion_label, criterion_dur,
                   device, clip: float = 1.0):
    """
    One training epoch.
    Total loss = L_label + 0.5 * L_duration

    The 0.5 weight downscales the duration loss because chord label accuracy
    is the primary objective; duration is a useful auxiliary target.
    """
    model.train()
    total_loss, correct_label, total = 0.0, 0, 0

    for melody_ids, chord_labels, chord_durs in loader:
        melody_ids   = melody_ids.to(device)
        chord_labels = chord_labels.to(device)
        chord_durs   = chord_durs.to(device)

        optimizer.zero_grad()
        label_logits, dur_logits = model(melody_ids)

        loss = (criterion_label(label_logits, chord_labels) +
                0.5 * criterion_dur(dur_logits, chord_durs))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss   += loss.item()
        correct_label += (label_logits.argmax(-1) == chord_labels).sum().item()
        total         += len(chord_labels)

    return total_loss / max(len(loader), 1), correct_label / max(total, 1)


@torch.no_grad()
def evaluate_t2(model, loader, criterion_label, criterion_dur, device):
    """
    Evaluate on a DataLoader.
    Returns (avg_loss, chord_root_quality_accuracy, duration_accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct_label, correct_dur, total = 0, 0, 0

    for melody_ids, chord_labels, chord_durs in loader:
        melody_ids   = melody_ids.to(device)
        chord_labels = chord_labels.to(device)
        chord_durs   = chord_durs.to(device)

        label_logits, dur_logits = model(melody_ids)
        loss = (criterion_label(label_logits, chord_labels) +
                0.5 * criterion_dur(dur_logits, chord_durs))

        total_loss    += loss.item()
        correct_label += (label_logits.argmax(-1) == chord_labels).sum().item()
        correct_dur   += (dur_logits.argmax(-1) == chord_durs).sum().item()
        total         += len(chord_labels)

    n = max(len(loader), 1)
    t = max(total, 1)
    return total_loss / n, correct_label / t, correct_dur / t


def train_model_t2(
    model,
    train_windows,
    val_windows,
    melody_vocab,
    chord_label_vocab,
    device,
    batch_size:       int   = 32,
    lr:               float = 5e-4,
    weight_decay:     float = 1e-5,
    max_epochs:       int   = 25,
    patience:         int   = 5,
    checkpoint_path:  str   = 'task2_best_model.pt',
):
    """
    Full training loop with:
      - Adam(lr=5e-4, weight_decay=1e-5)
      - Early stopping on validation chord label accuracy (patience=5)
      - Best model saved to checkpoint_path

    Returns train_losses, val_losses, train_accs, val_accs.
    """
    train_ds = HarmonizationDataset(train_windows, melody_vocab, chord_label_vocab)
    val_ds   = HarmonizationDataset(val_windows,   melody_vocab, chord_label_vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    criterion_label = nn.CrossEntropyLoss()
    criterion_dur   = nn.CrossEntropyLoss()
    optimizer       = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc  = -1.0
    no_improve    = 0

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = train_epoch_t2(model, train_loader, optimizer,
                                         criterion_label, criterion_dur,
                                         device)
        vl_loss, vl_acc, vl_dur_acc = evaluate_t2(model, val_loader,
                                                   criterion_label, criterion_dur,
                                                   device)
        train_losses.append(tr_loss);  val_losses.append(vl_loss)
        train_accs.append(tr_acc);     val_accs.append(vl_acc)

        print(f"Epoch {epoch:02d} | "
              f"Train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | "
              f"Val loss: {vl_loss:.4f} acc: {vl_acc:.3f} dur_acc: {vl_dur_acc:.3f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve   = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Best val chord accuracy: {best_val_acc:.3f}")
    return train_losses, val_losses, train_accs, val_accs


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  8. GENERATION                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def harmonize(model, melody_windows_encoded, id2chord_label, device, window_spans=None):
    """
    Greedy chord prediction for a sequence of encoded melody windows.

    Args:
        model                  : trained Seq2SeqHarmonizer
        melody_windows_encoded : list of LongTensors (window_size,)
        id2chord_label         : int → (root, quality)
        device

    Returns:
        list of (root, quality, dur_bin) tuples, one per window. When
        window_spans is provided, each tuple also includes start_sec, end_sec.
    """
    model.eval()
    predictions = []

    for i, mel_ids in enumerate(melody_windows_encoded):
        mel_ids_batch = mel_ids.unsqueeze(0).to(device)       # (1, window_size)
        label_logits, dur_logits = model(mel_ids_batch)

        pred_label = label_logits.argmax(-1).item()
        pred_dur   = dur_logits.argmax(-1).item()
        root, quality = id2chord_label.get(pred_label, ('C', 'major'))
        prediction = (root, quality, pred_dur)
        if window_spans is not None:
            prediction += tuple(window_spans[i])
        predictions.append(prediction)

    return predictions


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  9. MIDI EXPORT                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def harmonization_to_midi(
    melody_notes,
    predicted_chords,
    output_path: str   = 'symbolic_conditioned.mid',
    tempo:       float = 120.0,
    melody_program:   int = 0,    # piano
    chord_program:    int = 48,   # strings
    chord_velocity:   int = 60,
    melody_velocity:  int = 90,
    chord_octave:     int = 3,    # place chords below melody
):
    """
    Render melody + predicted chords to a two-track MIDI file.

    Channel 0 — original melody notes (from pretty_midi note dicts from §2.1)
    Channel 1 — predicted block chords: root + third + fifth as simultaneous notes

    Args:
        melody_notes     : list of note dicts from extract_melody_tokens()
                           each has keys: pitch, dur_bin, start_sec, end_sec
        predicted_chords : list of either (root, quality, dur_bin) or
                           (root, quality, dur_bin, start_sec, end_sec)
        output_path      : .mid filename
        tempo            : BPM

    Returns:
        output_path (str)
    """
    pm  = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # ── Track 0: melody ──────────────────────────────────────────────────────
    melody_inst = pretty_midi.Instrument(program=melody_program, name='melody')
    for nd in melody_notes:
        start_sec = nd['start_sec']
        end_sec   = nd['end_sec']
        note = pretty_midi.Note(
            velocity=melody_velocity,
            pitch=int(nd['pitch']),
            start=start_sec,
            end=max(end_sec, start_sec + 0.05),
        )
        melody_inst.notes.append(note)
    pm.instruments.append(melody_inst)

    # ── Track 1: chords ───────────────────────────────────────────────────────
    chord_inst = pretty_midi.Instrument(program=chord_program, name='chords')
    if melody_notes and predicted_chords:
        bps = tempo / 60.0
        cursor_sec = melody_notes[0]['start_sec']
        for chord in predicted_chords:
            root, quality, dur_bin = chord[:3]
            if len(chord) >= 5:
                start_sec, window_end_sec = chord[3], chord[4]
                end_sec = min(
                    start_sec + DURATION_BINS_BEATS[dur_bin] / bps,
                    window_end_sec,
                )
            else:
                start_sec = cursor_sec
                end_sec = start_sec + DURATION_BINS_BEATS[dur_bin] / bps
                cursor_sec = end_sec

            root_pitch = ROOT_TO_PITCH.get(root, 60) - 12  # one octave down
            intervals  = CHORD_INTERVALS.get(quality, [0, 4, 7])
            for interval in intervals:
                p = root_pitch + interval
                if 0 <= p <= 127:
                    chord_inst.notes.append(pretty_midi.Note(
                        velocity=chord_velocity,
                        pitch=p,
                        start=start_sec,
                        end=max(end_sec, start_sec + 0.05),
                    ))
    pm.instruments.append(chord_inst)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pm.write(output_path)
    total_notes = len(melody_inst.notes) + len(chord_inst.notes)
    print(f"MIDI saved → {output_path}  "
          f"({len(melody_inst.notes)} melody notes, "
          f"{len(chord_inst.notes)} chord notes)")
    return output_path
