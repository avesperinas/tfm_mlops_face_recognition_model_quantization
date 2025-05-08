import os


if os.environ.get("EXE_MODE", "prod") == "debug":

    from .decorator_definition import (
        i_decorator as manage_exceptions,
    )

else:

    from .decorator_definition import (
        manage_exceptions,
    )


__all__ = [
    "manage_exceptions",
]
