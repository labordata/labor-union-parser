"""Tests for CLI."""

import csv
import io

from click.testing import CliRunner

from labor_union_parser.cli import main


def invoke(args, input_data):
    """Helper to invoke CLI."""
    runner = CliRunner()
    return runner.invoke(main, args, input=input_data)


class TestBasicExtraction:
    """Test basic extraction functionality."""

    def test_single_column_no_header(self):
        result = invoke(["--no-header"], "SEIU Local 1199\n")

        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["text"] == "SEIU Local 1199"
        assert rows[0]["pred_aff"] == "SEIU"
        assert rows[0]["pred_desig"] == "1199"
        assert rows[0]["pred_is_union"] == "True"

    def test_single_column_with_header(self):
        result = invoke([], "name\nIBT 123\n")

        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["name"] == "IBT 123"
        assert rows[0]["pred_aff"] == "IBT"
        assert rows[0]["pred_desig"] == "123"

    def test_multi_column_with_column_flag(self):
        result = invoke(["-c", "union"], "id,union\n1,UAW 42\n")

        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["id"] == "1"
        assert rows[0]["union"] == "UAW 42"
        assert rows[0]["pred_aff"] == "UAW"
        assert rows[0]["pred_desig"] == "42"

    def test_multiple_rows(self):
        result = invoke([], "text\nSEIU Local 1199\nIBT 123\nUAW 42\n")

        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["pred_aff"] == "SEIU"
        assert rows[1]["pred_aff"] == "IBT"
        assert rows[2]["pred_aff"] == "UAW"


class TestNonUnionDetection:
    """Test non-union text detection."""

    def test_non_union_text(self):
        result = invoke(["--no-header"], "Random Company Inc\n")

        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        rows = list(reader)
        assert rows[0]["pred_is_union"] == "False"
        assert rows[0]["pred_aff"] == ""


class TestErrorHandling:
    """Test error cases."""

    def test_empty_input(self):
        result = invoke([], "")

        assert result.exit_code != 0
        assert "Empty CSV" in result.output

    def test_no_header_with_column_flag(self):
        result = invoke(["--no-header", "-c", "text"], "test\n")

        assert result.exit_code != 0
        assert "cannot be used with" in result.output

    def test_multi_column_without_column_flag(self):
        result = invoke([], "id,name\n1,test\n")

        assert result.exit_code != 0
        assert "multiple columns" in result.output

    def test_invalid_column_name(self):
        result = invoke(["-c", "missing"], "id,name\n1,test\n")

        assert result.exit_code != 0
        assert "not found" in result.output


class TestOutputFields:
    """Test that all expected output fields are present."""

    def test_all_prediction_fields_present(self):
        result = invoke(["--no-header"], "SEIU Local 1199\n")

        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        rows = list(reader)
        row = rows[0]

        expected_fields = [
            "text",
            "pred_is_union",
            "pred_aff",
            "pred_unknown",
            "pred_desig",
            "pred_union_score",
            "pred_fnum",
            "pred_fnum_multiple",
        ]
        for field in expected_fields:
            assert field in row, f"Missing field: {field}"

    def test_union_score_is_float(self):
        result = invoke(["--no-header"], "SEIU Local 1199\n")

        reader = csv.DictReader(io.StringIO(result.output))
        row = next(reader)
        score = float(row["pred_union_score"])
        assert 0.0 <= score <= 1.0

    def test_fnum_lookup(self):
        result = invoke(["--no-header"], "IBT 123\n")

        reader = csv.DictReader(io.StringIO(result.output))
        row = next(reader)
        # IBT 123 should have an fnum
        assert row["pred_fnum"] != ""


class TestPreservesInputColumns:
    """Test that input columns are preserved in output."""

    def test_preserves_all_input_columns(self):
        result = invoke(["-c", "name"], "id,name,extra\n1,SEIU Local 1199,foo\n")

        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        row = next(reader)
        assert row["id"] == "1"
        assert row["name"] == "SEIU Local 1199"
        assert row["extra"] == "foo"
        assert row["pred_aff"] == "SEIU"
