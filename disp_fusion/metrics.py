"""Diagnostics only; none of these measurements affect refinement."""
import numpy as np
from .alignment import valid_disparity


def five_point_mask(valid):
    return (valid[1:-1, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
            & valid[:-2, 1:-1] & valid[2:, 1:-1])


def laplacian(field):
    return field[1:-1, 1:-1]-(field[1:-1, :-2]+field[1:-1, 2:]
                             +field[:-2, 1:-1]+field[2:, 1:-1])/4


def summarize_result(original, result):
    kept = valid_disparity(result.disparity)
    valid = valid_disparity(original)
    changes = np.abs(result.disparity[kept]-original[kept])
    common = five_point_mask(kept)
    before = np.abs(laplacian(original)[common])
    after = np.abs(laplacian(result.disparity)[common])
    return dict(original_valid=int(valid.sum()), retained=int(kept.sum()), removed=int(result.removed_mask.sum()),
                original_invalid_filled=int((kept & ~valid).sum()),
                changed_over_005px=int((changes > .05).sum()),
                maximum_change_px=float(changes.max()) if len(changes) else 0.,
                common_stencils=int(common.sum()), roughness_before_sum=float(before.astype(float).sum()),
                roughness_after_sum=float(after.astype(float).sum()),
                warning='Roughness reduction is not independent physical accuracy.')


def audit_against_reference(original, field, reference, reliable, common):
    """Post-hoc matching diagnostics; a cached reference is not ground truth."""
    if not (original.shape == field.shape == reference.shape == reliable.shape == common.shape):
        raise ValueError('audit arrays must have identical shapes')
    checked = valid_disparity(original) & valid_disparity(reference) & reliable.astype(bool)
    error = np.abs(original-reference)
    supported = checked & (error <= .75)
    conflicting = checked & (error > 2.)
    removed = valid_disparity(original) & ~valid_disparity(field)
    paired = checked & common
    return dict(original_checked=int(checked.sum()), original_supported=int(supported.sum()),
        original_conflicting=int(conflicting.sum()), removed_supported=int((removed & supported).sum()),
        removed_conflicting=int((removed & conflicting).sum()), paired_points=int(paired.sum()),
        paired_abs_error_sum=float(np.abs(field-reference)[paired].astype(float).sum()))
