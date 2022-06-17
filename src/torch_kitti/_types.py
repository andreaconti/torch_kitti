import sys

__all__ = ["Literal"]

if sys.version_info.minor > 7:
    from typing import Literal
else:
    from typing_extensions import Literal
