"""
task1_model.py
==============
Task 1: Symbolic Unconditioned Generation — LSTM Language Model
Owner: Ulises

This file contains the complete implementation:
  - NottinghamDataset  : sliding-window PyTorch Dataset
  - LSTMLanguageModel  : Embedding → LSTM → Linear
  - train_epoch / evaluate : one-epoch helpers
  - train_model        : full loop with early stopping + LR scheduling
  - generate_melody    : autoregressive temperature sampling
  - tokens_to_midi     : decode token IDs → pretty_midi → .mid file

The notebook (simple_notebook.ipynb §1.2) imports from this module and
walks through each component with explanation.
"""

import os
import math
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pretty_midi

# ── Duration bins (must match §1.1 preprocessing) ──────────────────────────
DURATION_BINS_BEATS = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]

SEED = 42


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. DATASET                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class NottinghamDataset(Dataset):
    """
    Converts a list of encoded token-ID sequences into (input, target) pairs
    using a sliding window.

    For a sequence  [s0, s1, s2, ..., sN]:
      input  = [s_i,   ..., s_{i+L-1}]   (length L)
      target = [s_{i+1}, ..., s_{i+L}]   (shifted by one → next-token prediction)

    Args:
        sequences : list of list[int]  — encoded sequences from §1.1
        seq_len   : window length L (default 128)
        stride    : hop between windows (default 64, giving 50 % overlap)
        pad_id    : token ID used for padding short windows (default 0 = <PAD>)
    """

    def __init__(self, sequences, seq_len=128, stride=64, pad_id=0):
        self.seq_len = seq_len
        self.pairs: list[tuple[list[int], list[int]]] = []

        for seq in sequences:
            # Short sequences are retained and padded to one complete window.
            if len(seq) < 2:
                continue
            for start in range(0, max(1, len(seq) - 1), stride):
                inp = seq[start : start + seq_len]
                tgt = seq[start + 1 : start + seq_len + 1]

                # Pad each side independently — tgt can be shorter than inp
                # when the slice hits the end of the sequence
                inp = inp + [pad_id] * (seq_len - len(inp))
                tgt = tgt + [pad_id] * (seq_len - len(tgt))

                self.pairs.append((inp, tgt))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        inp, tgt = self.pairs[idx]
        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. MODEL                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class LSTMLanguageModel(nn.Module):
    """
    Token-level LSTM language model for (pitch, duration) sequences.

    Architecture:
        Embedding(vocab_size, embed_dim)
        → Dropout
        → LSTM(embed_dim → hidden_dim, num_layers layers, dropout between layers)
        → Dropout
        → Linear(hidden_dim → vocab_size)

    The model outputs raw logits (pre-softmax) at every timestep; loss is
    computed externally with nn.CrossEntropyLoss.

    Why LSTM over alternatives:
      • Markov chain    — only first-order dependencies; cannot model phrase structure
      • Transformer     — ~1 k short sequences makes it prone to overfitting;
                          also requires positional encodings and much more memory
      • LSTM            — hidden state carries information across the entire
                          prefix, training is stable, and the dataset is small
                          enough that a 2-layer 256-unit LSTM converges quickly
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embed discrete tokens into a dense space
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Core recurrence — dropout is applied *between* layers, not after the last
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,   # input/output shape: (batch, time, features)
        )

        self.dropout = nn.Dropout(dropout)

        # Project hidden state to vocabulary logits
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x      : LongTensor (batch, seq_len)
            hidden : tuple (h_0, c_0) from previous step, or None to start fresh

        Returns:
            logits : FloatTensor (batch, seq_len, vocab_size)
            hidden : tuple (h_n, c_n)  — carry state across batches during generation
        """
        emb = self.dropout(self.embedding(x))   # (B, T, E)
        out, hidden = self.lstm(emb, hidden)     # (B, T, H)
        logits = self.fc(self.dropout(out))      # (B, T, V)
        return logits, hidden

    def init_hidden(self, batch_size: int, device):
        """Return zeroed initial hidden and cell states."""
        zeros = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (zeros, zeros.clone())


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. TRAINING                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_epoch(model, loader, optimizer, criterion, device, clip: float = 1.0) -> float:
    """
    One full pass over the training DataLoader.

    Gradient clipping (clip=1.0) prevents exploding gradients — a common issue
    with vanilla RNNs/LSTMs on long sequences.

    Returns:
        mean cross-entropy loss over all batches
    """
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)   # (B, T)

        optimizer.zero_grad()
        logits, _ = model(x)                 # (B, T, V) — fresh hidden each batch

        # Flatten time and batch dims for cross-entropy
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> float:
    """
    Evaluate the model on a DataLoader (val or test).

    Returns:
        mean cross-entropy loss (convert to perplexity with math.exp(loss))
    """
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def train_model(
    model,
    train_seqs,
    val_seqs,
    token2id,
    device,
    seq_len: int = 128,
    stride: int = 64,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 5,
    checkpoint_path: str = "task1_best_model.pt",
):
    """
    Full training loop with:
      - Adam optimiser at lr=1e-3
      - ReduceLROnPlateau scheduler (factor=0.5, patience=2)
      - Early stopping on validation loss (patience=5)
      - Best model saved to checkpoint_path

    Returns:
        train_losses : list[float]
        val_losses   : list[float]
    """
    pad_id = token2id.get("<PAD>", 0)

    train_ds = NottinghamDataset(train_seqs, seq_len=seq_len, stride=stride, pad_id=pad_id)
    val_ds   = NottinghamDataset(val_seqs,   seq_len=seq_len, stride=stride, pad_id=pad_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, max_epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss = evaluate(model, val_loader,   criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        scheduler.step(vl_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss: {tr_loss:.4f} | "
            f"Val loss: {vl_loss:.4f} | "
            f"Val PPL: {math.exp(vl_loss):.2f}"
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            no_improve    = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # Reload the best checkpoint before returning
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Best val loss: {best_val_loss:.4f}  (PPL {math.exp(best_val_loss):.2f})")
    return train_losses, val_losses


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. GENERATION                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def generate_melody(
    model,
    seed_ids: list[int],
    id2token: dict,
    token2id: dict,
    device,
    temperature: float = 1.0,
    max_len: int = 200,
) -> list[int]:
    """
    Autoregressively sample a melody token sequence.

    Algorithm:
      1. Encode seed_ids through the model to prime the hidden state.
      2. At each step: divide logits by temperature, softmax → multinomial sample.
         - temperature < 1  →  sharper, more repetitive  (safer)
         - temperature = 1  →  unmodified distribution
         - temperature > 1  →  flatter, more surprising   (risk of incoherence)
      3. Stop when <EOS> is sampled or max_len is reached.

    Args:
        model       : trained LSTMLanguageModel
        seed_ids    : list of token IDs to prime the model (e.g. [token2id['<START>']])
        id2token    : reverse vocabulary
        token2id    : forward vocabulary (to look up <EOS>)
        device      : torch device
        temperature : sampling temperature (try 0.8, 1.0, 1.2)
        max_len     : maximum tokens to generate after the seed

    Returns:
        list of generated token IDs (including the seed)
    """
    model.eval()
    eos_id = token2id.get("<EOS>")

    generated = list(seed_ids)

    # Prime hidden state on the seed sequence
    x = torch.tensor([seed_ids], dtype=torch.long, device=device)   # (1, seed_len)
    logits, hidden = model(x)

    # Autoregressively sample one token at a time
    for _ in range(max_len):
        # last time step logits
        last_logit = logits[0, -1, :] / max(temperature, 1e-6)      # (V,)
        probs      = torch.softmax(last_logit, dim=-1)
        next_id    = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_id)

        if next_id == eos_id:
            break

        # Feed just the sampled token for the next step (carry hidden state)
        x      = torch.tensor([[next_id]], dtype=torch.long, device=device)  # (1, 1)
        logits, hidden = model(x, hidden)

    return generated


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5. MIDI EXPORT                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tokens_to_midi(
    token_ids: list[int],
    id2token: dict,
    output_path: str = "symbolic_unconditioned.mid",
    tempo: float = 120.0,
    velocity: int = 80,
    program: int = 0,   # 0 = Acoustic Grand Piano
) -> str:
    """
    Decode a list of token IDs back to (pitch, duration) pairs and write a MIDI file.

    Only tokens that are (pitch, dur_bin) tuples are written as notes; special
    tokens (<PAD>, <EOS>, <START>) are silently skipped.

    Args:
        token_ids   : output of generate_melody()
        id2token    : reverse vocabulary from §1.1
        output_path : destination .mid filename
        tempo       : BPM for playback (default 120)
        velocity    : MIDI note velocity 0-127 (default 80)
        program     : General MIDI instrument number (0 = piano)

    Returns:
        output_path (for convenience)
    """
    bps = tempo / 60.0   # beats per second

    pm         = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=program, name="melody")

    current_time = 0.0

    for tid in token_ids:
        tok = id2token.get(tid)
        if not isinstance(tok, tuple) or len(tok) != 2:
            # Skip special / unknown tokens
            continue

        pitch, dur_bin = tok

        # Guard against out-of-range pitch or bin
        if not (0 <= pitch <= 127):
            continue
        if not (0 <= dur_bin < len(DURATION_BINS_BEATS)):
            continue

        dur_secs = DURATION_BINS_BEATS[dur_bin] / bps
        note     = pretty_midi.Note(
            velocity=velocity,
            pitch=int(pitch),
            start=current_time,
            end=current_time + dur_secs,
        )
        instrument.notes.append(note)
        current_time += dur_secs

    pm.instruments.append(instrument)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pm.write(output_path)
    print(f"MIDI saved → {output_path}  ({len(instrument.notes)} notes, {current_time:.1f}s)")
    return output_path
