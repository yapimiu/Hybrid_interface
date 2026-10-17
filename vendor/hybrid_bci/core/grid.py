from typing import List
from .tile import Tile

class Grid:

    def __init__(self, size: int=3):
        self.size = size
        self.tiles: List[Tile] = []
        self._build()

    def _build(self) -> None:
        tile_id = 0
        for row in range(self.size):
            for col in range(self.size):
                self.tiles.append(Tile(tile_id, row, col))
                tile_id += 1

    def reset(self) -> None:
        for tile in self.tiles:
            tile.active = False
