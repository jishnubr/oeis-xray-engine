"""Versioned registry for importable transformer implementations."""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import TypeVar

from . import TransformError
from .contracts import TransformerContract

_TransformerType = TypeVar("_TransformerType", bound=type[TransformerContract])


class TransformerRegistry:
    """Resolve transformers by their stable semantic name and version.

    Only importable classes with no required constructor arguments can be
    registered. This lets workers created with multiprocessing ``spawn``
    reconstruct the implementation from an immutable ``module:qualname``
    reference instead of sharing mutable plugin instances.
    """

    def __init__(self) -> None:
        self._transformers: dict[tuple[str, str], type[TransformerContract]] = {}

    def register(
        self,
        transformer_type: type[TransformerContract],
    ) -> None:
        """Register one concrete, worker-importable transformer class."""

        if not inspect.isclass(transformer_type) or not issubclass(
            transformer_type, TransformerContract
        ):
            raise TypeError("transformer_type must subclass TransformerContract")
        if inspect.isabstract(transformer_type):
            raise TypeError("abstract transformer classes cannot be registered")

        name = getattr(transformer_type, "name", None)
        version = getattr(transformer_type, "version", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("transformer name must be a non-empty string")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("transformer version must be a non-empty string")
        if "<locals>" in transformer_type.__qualname__:
            raise ValueError("transformer class must be defined at module scope")

        signature = inspect.signature(transformer_type)
        required_parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        if required_parameters:
            raise ValueError(
                "registered transformers must have a no-argument constructor"
            )

        key = (name, version)
        existing = self._transformers.get(key)
        if existing is not None and existing is not transformer_type:
            raise ValueError(
                f"a different transformer is already registered for {name}@{version}"
            )
        self._transformers[key] = transformer_type

    def resolve(self, name: str, version: str) -> TransformerContract:
        """Return a fresh stateless transformer instance for a name/version."""

        try:
            transformer_type = self._transformers[(name, version)]
        except KeyError as exc:
            raise TransformError(
                "transformer is not registered",
                context={"name": name, "version": version},
            ) from exc
        return transformer_type()

    def worker_reference(self, name: str, version: str) -> str:
        """Return the import reference workers use to reconstruct a plugin."""

        try:
            transformer_type = self._transformers[(name, version)]
        except KeyError as exc:
            raise TransformError(
                "transformer is not registered",
                context={"name": name, "version": version},
            ) from exc
        return f"{transformer_type.__module__}:{transformer_type.__qualname__}"

    @staticmethod
    def from_worker_reference(
        reference: str,
        *,
        expected_name: str,
        expected_version: str,
    ) -> TransformerContract:
        """Import and instantiate a registered worker implementation.

        The task manifest carries this reference so recovery uses the
        implementation planned by the coordinator, not a name-based fallback.
        """

        if not isinstance(reference, str) or ":" not in reference:
            raise TransformError("invalid transformer worker reference")
        module_name, qualified_name = reference.split(":", 1)
        if not module_name or not qualified_name or "<locals>" in qualified_name:
            raise TransformError("invalid transformer worker reference")

        try:
            target: object = import_module(module_name)
            for component in qualified_name.split("."):
                target = getattr(target, component)
        except (ImportError, AttributeError) as exc:
            raise TransformError(
                "worker could not import the registered transformer",
                context={"reference": reference, "error": str(exc)},
            ) from exc

        if not inspect.isclass(target) or not issubclass(target, TransformerContract):
            raise TransformError(
                "worker reference does not resolve to a TransformerContract",
                context={"reference": reference},
            )
        if target.name != expected_name or target.version != expected_version:
            raise TransformError(
                "worker transformer identity differs from the planned spec",
                context={
                    "reference": reference,
                    "expected": f"{expected_name}@{expected_version}",
                    "actual": f"{target.name}@{target.version}",
                },
            )
        return target()


def default_transformer_registry() -> TransformerRegistry:
    """Build the registry for transforms currently supported by the pipeline."""

    from .builtins import EulerTransformer, InverseEulerTransformer
    from .denumerant import DenumerantTransformer

    registry = TransformerRegistry()
    registry.register(EulerTransformer)
    registry.register(InverseEulerTransformer)
    registry.register(DenumerantTransformer)
    return registry
