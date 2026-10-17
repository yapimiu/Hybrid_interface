from dataclasses import dataclass

@dataclass
class Tile:
    id: int
    row: int
    col: int
    active: bool = False
