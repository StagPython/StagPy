from __future__ import annotations

from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET


class XmlStream:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._event = "end"
        self._elem: ET.Element

    @cached_property
    def _cursor(self) -> Iterator[tuple[str, ET.Element]]:
        return ET.iterparse(str(self.filepath), events=("start", "end"))

    def _to_next_start(self) -> ET.Element:
        for self._event, self._elem in self._cursor:
            if self._event == "start":
                return self._elem
            self._elem.clear()
        raise RuntimeError("Reached end of file")

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
        except RuntimeError:
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
        for self._event, elem in self._cursor:
            if self._event == "start":
                self.load().__enter__()
            else:
                yield elem
                break
        elem.clear()
