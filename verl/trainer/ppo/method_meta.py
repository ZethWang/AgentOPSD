"""Single source of truth for the method's user-facing display name.

The implementation is internally named ``opsd`` (modules, classes, the
``algorithm.opsd.*`` config namespace). The public/paper name is ``AgentOPSD``.
Override the display name at runtime with the ``AGENTOPSD_METHOD_NAME`` env var
so logs / wandb project names / experiment tags can be re-branded in one place.
"""

import os

METHOD_DISPLAY_NAME = os.getenv("AGENTOPSD_METHOD_NAME", "AgentOPSD")
