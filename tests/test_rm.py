import pytest

from great_tables import GT, exibble
from great_tables._gt_data import Heading
from great_tables.loc import body


@pytest.fixture
def gt_tbl() -> GT:
    return GT(exibble, rowname_col="row", groupname_col="group")


def test_rm_header(gt_tbl: GT):
    res = gt_tbl.tab_header(title="the title", subtitle="the subtitle").rm_header()
    assert res._heading == Heading()


def test_rm_header_no_header(gt_tbl: GT):
    assert gt_tbl.rm_header()._heading == Heading()


def test_rm_stubhead(gt_tbl: GT):
    res = gt_tbl.tab_stubhead(label="the label").rm_stubhead()
    assert res._stubhead is None


def test_rm_source_notes_all(gt_tbl: GT):
    res = gt_tbl.tab_source_note("a").tab_source_note("b").rm_source_notes()
    assert res._source_notes == []


def test_rm_source_notes_by_index(gt_tbl: GT):
    res = gt_tbl.tab_source_note("a").tab_source_note("b").rm_source_notes(source_notes=0)
    assert res._source_notes == ["b"]


def test_rm_source_notes_by_index_list(gt_tbl: GT):
    res = (
        gt_tbl.tab_source_note("a")
        .tab_source_note("b")
        .tab_source_note("c")
        .rm_source_notes(source_notes=[0, 2])
    )
    assert res._source_notes == ["b"]


def test_rm_source_notes_index_out_of_range(gt_tbl: GT):
    with pytest.raises(ValueError):
        gt_tbl.tab_source_note("a").rm_source_notes(source_notes=5)


def test_rm_footnotes_all(gt_tbl: GT):
    res = (
        gt_tbl.tab_footnote("fn1", locations=body(columns="num", rows=[0]))
        .tab_footnote("fn2", locations=body(columns="char", rows=[0]))
        .rm_footnotes()
    )
    assert res._footnotes == []


def test_rm_footnotes_by_index(gt_tbl: GT):
    res = (
        gt_tbl.tab_footnote("fn1", locations=body(columns="num", rows=[0]))
        .tab_footnote("fn2", locations=body(columns="char", rows=[0]))
        .rm_footnotes(footnotes=0)
    )
    assert len(res._footnotes) == 1
    assert res._footnotes[0].footnotes == ["fn2"]


def test_rm_footnotes_index_out_of_range(gt_tbl: GT):
    with pytest.raises(ValueError):
        gt_tbl.tab_footnote("fn1", locations=body(columns="num", rows=[0])).rm_footnotes(
            footnotes=5
        )


def test_rm_spanners_by_id(gt_tbl: GT):
    res = (
        gt_tbl.tab_spanner(label="perf", columns=["num", "currency"], id="perf")
        .tab_spanner(label="econ", columns=["char"], id="econ")
        .rm_spanners(spanners="perf")
    )
    assert [span.spanner_id for span in res._spanners] == ["econ"]


def test_rm_spanners_by_id_list(gt_tbl: GT):
    res = (
        gt_tbl.tab_spanner(label="perf", columns=["num"], id="perf")
        .tab_spanner(label="econ", columns=["char"], id="econ")
        .rm_spanners(spanners=["perf", "econ"])
    )
    assert list(res._spanners) == []


def test_rm_spanners_all(gt_tbl: GT):
    res = (
        gt_tbl.tab_spanner(label="perf", columns=["num"], id="perf")
        .tab_spanner(label="econ", columns=["char"], id="econ")
        .rm_spanners()
    )
    assert list(res._spanners) == []


def test_rm_spanners_by_level(gt_tbl: GT):
    # `top` sits on level 1 (above `perf`); `perf` and `econ` are on level 0
    res = (
        gt_tbl.tab_spanner(label="perf", columns=["num"], id="perf")
        .tab_spanner(label="econ", columns=["char"], id="econ")
        .tab_spanner(label="top", columns=["num"], id="top")
        .rm_spanners(levels=0)
    )
    assert [span.spanner_id for span in res._spanners] == ["top"]


def test_rm_spanners_id_and_level_intersection(gt_tbl: GT):
    # `perf` is on level 0, so restricting to level 1 spares it
    res = (
        gt_tbl.tab_spanner(label="perf", columns=["num"], id="perf")
        .tab_spanner(label="top", columns=["num"], id="top")
        .rm_spanners(spanners="perf", levels=1)
    )
    assert [span.spanner_id for span in res._spanners] == ["perf", "top"]


def test_rm_spanners_no_spanners(gt_tbl: GT):
    assert list(gt_tbl.rm_spanners()._spanners) == []


def test_rm_spanners_bad_id(gt_tbl: GT):
    with pytest.raises(ValueError):
        gt_tbl.tab_spanner(label="perf", columns=["num"], id="perf").rm_spanners(spanners="nope")


def test_rm_spanners_bad_level(gt_tbl: GT):
    with pytest.raises(ValueError):
        gt_tbl.tab_spanner(label="perf", columns=["num"], id="perf").rm_spanners(levels=9)
