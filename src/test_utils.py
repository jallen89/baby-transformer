import pytest
import torch
from .utils import FrenchEnglishDataset
from pathlib import Path

@pytest.fixture
def sample_data_file(tmp_path: Path):
    """Create a dummy data file for testing."""
    data = (
        "Hello.\tBonjour.\n"
        "How are you?\tComment ça va ?\n"
        "I am fine.\tJe vais bien.\n"
        "\n"  # Empty line to be ignored
        "This is a test.\tCeci est un test.\n"
        "One part only\n" # Malformed line to be ignored
    )
    file_path = tmp_path / "test_data.txt"
    file_path.write_text(data, encoding="utf-8")
    return file_path

def test_load_data_counts(sample_data_file: Path):
    """Tests that load_data correctly counts translation pairs, ignoring empty/malformed lines."""
    dataset = FrenchEnglishDataset(filename=str(sample_data_file))
    translation_pairs = dataset.load_data(str(sample_data_file))
    assert len(translation_pairs) == 4

def test_load_data_content(sample_data_file: Path):
    """Tests that load_data correctly parses the content of translation pairs."""
    dataset = FrenchEnglishDataset(filename=str(sample_data_file))
    translation_pairs = dataset.load_data(str(sample_data_file))
    assert ("Hello.", "Bonjour.") in translation_pairs
    assert ("How are you?", "Comment ça va ?") in translation_pairs
    assert ("I am fine.", "Je vais bien.") in translation_pairs
    assert ("This is a test.", "Ceci est un test.") in translation_pairs


def test_encode_and_pad(sample_data_file: Path):
    """Tests that _encode_and_pad correctly encodes and pads the translation pairs."""

    dataset = FrenchEnglishDataset(filename=str(sample_data_file))
    dataset.intialize_dataset()

    print(dataset.english_tensor)
    print(dataset.french_tensor)

    # Check that tensors are created
    assert hasattr(dataset, 'english_tensor')
    assert hasattr(dataset, 'french_tensor')
    assert isinstance(dataset.english_tensor, torch.Tensor)
    assert isinstance(dataset.french_tensor, torch.Tensor)

    # Check dimensions
    # We have 4 valid pairs in the sample data
    assert dataset.english_tensor.size(0) == 4
    assert dataset.french_tensor.size(0) == 4

    # Check padding
    # Since sentences have different lengths, padding should be present (value 0)
    # We check if at least one zero exists in the tensors (assuming not all sentences are same length)
    assert (dataset.english_tensor == 0).any()
    assert (dataset.french_tensor == 0).any()
    