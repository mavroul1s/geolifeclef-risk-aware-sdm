"""Finite-completion diagnostic with externally established positive genera.

All unknown labels must comprise fully censored eligible genera, each known to
contain at least one eligible positive. This assumption is supplied by the T1
simulator, and must NOT be inferred from real unknown-label masks alone.
"""
import numpy as np


def positive_genus_f1_bounds(truth, known, eligible, genus_codes, rank, count):
    y, known, eligible = np.asarray(truth), np.asarray(known, bool), np.asarray(eligible, bool)
    codes, rank, count = np.asarray(genus_codes, int), np.asarray(rank, int), np.asarray(count, int)
    if y.shape != known.shape or y.shape != eligible.shape or codes.shape != (y.shape[1],):
        raise ValueError('Genus-completion identity mismatch')
    if np.any(known&~eligible) or np.any((y > 0)&~known) or np.any(~np.isin(y[known], [0, 1])):
        raise ValueError('Invalid known binary labels')
    if np.any(codes < 0) or len(rank) != len(y) or count.shape != (len(y),):
        raise ValueError('Invalid genus/prediction identity')
    lower, upper = np.empty(len(y)), np.empty(len(y))
    for row in range(len(y)):
        pred = np.zeros(y.shape[1], bool)
        selected = rank[row, :count[row]]
        if len(selected) != len(np.unique(selected)) or np.any(~eligible[row, selected]):
            raise ValueError('Invalid fixed predicted set')
        pred[selected] = True
        unknown = eligible[row]&~known[row]
        hidden = np.unique(codes[unknown])
        if np.any(np.isin(codes[known[row]], hidden)):
            raise ValueError('Unknown genera must be fully censored within eligible space')
        predicted_unknown = np.bincount(codes[unknown&pred], minlength=int(codes.max()+1))
        outside_unknown = np.bincount(codes[unknown&~pred], minlength=int(codes.max()+1))
        required_inside = int((outside_unknown[hidden] == 0).sum())
        required_outside = int((predicted_unknown[hidden] == 0).sum())
        hit = int(((y[row] > 0)&pred).sum()); positive = int((y[row] > 0).sum())
        up, uo = int(predicted_unknown.sum()), int(outside_unknown.sum())
        low_den = count[row]+positive+uo+required_inside
        high_den = count[row]+positive+up+required_outside
        lower[row] = 2*(hit+required_inside)/low_den if low_den else 1.
        upper[row] = 2*(hit+up)/high_den if high_den else 1.
    if np.any(lower > upper+1e-12):
        raise ArithmeticError('Inverted constrained interval')
    return lower, upper
