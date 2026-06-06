from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from xml.etree import ElementTree as ET


class EndOfXml(Exception):
    """End of Xml has been reached."""


class XmlStream:
    def __init__(self, filepath: Path):
        self.filepath: Path = filepath
        self._event: str = "end"
        self._elem: ET.Element

    @cached_property
    def _cursor(self) -> Iterator[tuple[str, ET.Element]]:
        return ET.iterparse(str(self.filepath), events=("start", "end"))

    def _to_next_start(self) -> ET.Element:
        for self._event, self._elem in self._cursor:
            if self._event == "start":
                return self._elem
            self._elem.clear()
        raise EndOfXml("Reached end of file")

    @property
    def current(self) -> ET.Element:
        """Element at "start" event."""
        if self._event == "start":
            return self._elem
        return self._to_next_start()

    def advance(self) -> None:
        """Advance to next "start" event."""
        self.current  # make sure to be at current "start" event
        self._to_next_start()

    def skip_to_tag(self, tag: str) -> None:
        """Progress in file (both width and depth) until reaching the given tag."""
        while self.current.tag != tag:
            self.advance()

    def iter_tag(self, tag: str) -> Iterator[None]:
        try:
            while True:
                self.skip_to_tag(tag)
                yield None
        except EndOfXml:
            pass

    def iter_load_successive_tag(self, tag: str) -> Iterator[ET.Element]:
        try:
            while self.current.tag == tag:
                with self.load() as elt:
                    yield elt
        except EndOfXml:
            pass

    def drop(self) -> None:
        """Discard the current element and its children."""
        self.current  # make sure to be at current "start" event
        for self._event, elem in self._cursor:
            if self._event == "start":
                self.drop()
            else:
                elem.clear()
                return

    @contextmanager
    def load(self) -> Iterator[ET.Element]:
        """Fully read the current element and its children."""
        self.current  # make sure to be at current "start" event
        elem = None
        for self._event, elem in self._cursor:
            if self._event == "start":
                self.load().__enter__()
            else:
                yield elem
                break
        assert elem is not None
        elem.clear()
