import re

from ._inputs import _normalize_parts


_NUMBERED_SLOT = re.compile(r"\{INPUT_(0|[1-9][0-9]*)\}")


def parse_template(template: str, placeholder: str) -> tuple:
    """Return literal strings interleaved with positional argument indices."""

    if placeholder == "{INPUT_PLACEHOLDER}":
        # Preserve the existing text-template semantics exactly.
        if template.count(placeholder) != 1:
            raise ValueError(
                "Compiled prompt template must contain exactly one "
                "{INPUT_PLACEHOLDER} placeholder."
            )

        prefix, suffix = template.split(placeholder)
        return prefix, 0, suffix

    if placeholder != "{INPUT_N}":
        raise ValueError(f"Unsupported placeholder syntax: {placeholder!r}")

    segments = []
    indices = set()
    end = 0

    for match in _NUMBERED_SLOT.finditer(template):
        index = int(match.group(1))
        segments.extend((template[end:match.start()], index))
        indices.add(index)
        end = match.end()

    segments.append(template[end:])

    if sorted(indices) != list(range(len(indices))):
        raise ValueError("Template indices must start at 0 without gaps.")

    return tuple(segments)


def bind_template(segments: tuple, *args) -> tuple:
    """Bind arguments to segments returned by parse_template."""
    arity = 1 + max(
        (segment for segment in segments if isinstance(segment, int)),
        default=-1,
    )
    if len(args) != arity:
        raise TypeError(
            f"Template requires {arity} positional arguments; got {len(args)}."
        )

    if args:
        _normalize_parts(*args)

    return tuple(
        args[segment] if isinstance(segment, int) else segment
        for segment in segments
    )
