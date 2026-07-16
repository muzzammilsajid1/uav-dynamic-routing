from __future__ import annotations

from envs.grid_environment import GridEnvironment, Node


def render_ascii(env: GridEnvironment, path: list[Node] | None = None) -> str:
    path_nodes = set(path or [])
    lines: list[str] = []

    for row in range(env.size):
        cells: list[str] = []
        for col in range(env.size):
            node = (row, col)
            if node == env.start:
                cells.append("S")
            elif node == env.goal:
                cells.append("G")
            elif node in path_nodes:
                cells.append("*")
            elif node in env.blocked:
                cells.append("#")
            else:
                cells.append(".")
        lines.append(" ".join(cells))

    return "\n".join(lines)

