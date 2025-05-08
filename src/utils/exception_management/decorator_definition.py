"""Define decorators to manage exceptions and identity decorators."""

import logging
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def manage_exceptions(
    default_output: Any | None = None,  # noqa: ANN401
    terminate: bool | None = False,
) -> Any:  # noqa: ANN401
    """
    Define a decorator with the needed number of outputs.

    Parameters
    ------------
    default_output: Any
        default output if any exception occurs.
    terminate: bool
        boolean to indicate if the script may be shout down if the function has
        an exception associated.

    Return
    ------------
    decorate_the_function: callable
        decorator
    """

    def decorate_the_function_with_error_management(
        function_to_decorate: Callable,
    ) -> Callable:
        """
        Manage all possible errors while calling drive's functions.
        If an exception is noted, the decorator will stop the execution or
        return the given default value.

        Parameters
        ------------
        function_to_decorate: Callable
            function to decorate

        Return
        ------------
        secure_function_execution: callable
            decorated function
        """

        @wraps(function_to_decorate)
        def secure_function_execution(
            *args,  # noqa: ANN002
            **kwargs,  # noqa: ANN003
        ) -> Any:  # noqa: ANN401
            """
            Manage the exceptions of the given function. It orchestrates
            the error handling process.

            Parameters
            ------------
            *args: Any
                function_to_decorate positional arguments
            **kwargs: Any
                function_to_decorate keyword arguments

            Return
            ------------
            output: Any
                function_to_decorate output, default output if exception occurs
            """
            output = default_output
            try:
                output = function_to_decorate(*args, **kwargs)

            except (Exception, SystemExit) as exception:
                logger.error(  # noqa: TRY400
                    f"Exception occurred in {function_to_decorate.__name__}: {exception}",  # noqa: G004
                )
                if terminate:
                    sys.exit()

            return output

        return secure_function_execution

    return decorate_the_function_with_error_management


def i_decorator(*args, **kwargs) -> Any:  # noqa: ANN002, ANN003, ANN401, ARG001
    """
    Define an identity decorator with the needed number of outputs.

    Return
    ------------
    decorate_the_function: callable
        identity decorator
    """

    def decorate_the_function(function_to_decorate: Callable) -> Callable:
        """
        Identity decorator, where decorated function = undecorated function. Used
        in substitution of manage_errors in debug mode.

        Return
        ------------
        decorate_the_function: callable
            identity decorator
        """

        @wraps(function_to_decorate)
        def i_function(*args, **kwargs) -> Any:  # noqa: ANN002, ANN003, ANN401
            """
            Return the decorated function with the given arguments.

            Parameters
            ------------
            *args: Any
                function_to_decorate positional arguments
            **kwargs: Any
                function_to_decorate keyword arguments

            Return
            ------------
            output: Any
                function_to_decorate output
            """
            return function_to_decorate(*args, **kwargs)

        return i_function

    return decorate_the_function
