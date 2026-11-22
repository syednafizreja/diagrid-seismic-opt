from collections import Counter

from diagrid_opt.dataset import sample_lhs_designs


def test_lhs_designs_are_reproducible_stratified_and_within_bounds():
    designs = sample_lhs_designs()
    repeated = sample_lhs_designs()

    assert designs == repeated
    assert len(designs) == 80
    assert len(
        {
            (
                design.span,
                design.arch_rise,
                design.diagrid_divisions,
                design.glulam_depth,
                design.niti_fuse_area,
            )
            for design in designs
        }
    ) == 80
    assert Counter(design.diagrid_divisions for design in designs) == {
        6: 20,
        8: 20,
        10: 20,
        12: 20,
    }
    for design in designs:
        assert 14.0 <= design.span <= 28.0
        assert 0.08 <= design.arch_rise / design.span <= 0.22
        assert 0.22 <= design.glulam_depth <= 0.48
        assert 60.0 <= design.niti_fuse_area <= 260.0
        assert design.branch_spread == 0.17 * design.span
