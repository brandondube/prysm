"""Tests for samplefiles."""

import io

from prysm import sample_data


def test_fetch_if_not_present_returns_existing_file_without_fetching(tmp_path, monkeypatch):
    local = tmp_path / 'cached.txt'
    local.write_text('already here')

    def fail_if_called(remote):
        raise AssertionError(f'urlopen called for {remote}')

    monkeypatch.setattr(sample_data, 'urlopen', fail_if_called)

    assert sample_data.fetch_if_not_present(local, 'https://example.com/cached.txt') == local
    assert local.read_text() == 'already here'


def test_fetch_if_not_present_downloads_missing_file(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

    local = tmp_path / 'downloaded.txt'

    def fake_urlopen(remote):
        assert remote == 'https://example.com/downloaded.txt'
        return Response(b'fresh data')

    monkeypatch.setattr(sample_data, 'urlopen', fake_urlopen)

    assert sample_data.fetch_if_not_present(local, 'https://example.com/downloaded.txt') == local
    assert local.read_bytes() == b'fresh data'


def test_sample_files_dat_alias_uses_named_file(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(local, remote):
        calls.append((local, remote))
        return local

    monkeypatch.setattr(sample_data, 'root', tmp_path)
    monkeypatch.setattr(sample_data, 'fetch_if_not_present', fake_fetch)

    out = sample_data.SampleFiles()('DAT')

    expected = (tmp_path / 'valid_zygo_dat_file.dat').absolute()
    assert out == expected
    assert calls == [
        (expected, sample_data.baseremote + 'valid_zygo_dat_file.dat'),
    ]


def test_sample_files_generic_filename_uses_lowercase_remote(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(local, remote):
        calls.append((local, remote))
        return local

    monkeypatch.setattr(sample_data, 'root', tmp_path)
    monkeypatch.setattr(sample_data, 'fetch_if_not_present', fake_fetch)

    out = sample_data.SampleFiles()('Boat.PNG')

    expected = tmp_path / 'boat.png'
    assert out == expected
    assert calls == [
        (expected, sample_data.baseremote + 'boat.png'),
    ]
