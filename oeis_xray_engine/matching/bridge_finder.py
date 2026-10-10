"""Find exact finite bridges from registered transforms to indexed sequences."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from ..errors import ValidationError
from ..indexing.hash_index import (
    Candidate,
    FingerprintProfile,
    HashIndex,
)
from ..models import SequenceRef, TransformSpec
from ..storage.term_store import TermStore
from ..transformers import TransformError
from .collision_detector import (
    CollisionDetector,
    RelationSpec,
    VerificationResult,
)


@dataclass(frozen=True, slots=True)
class BridgeResult:
    """A transform candidate and its exact finite verification outcome."""

    candidate: Candidate
    transform_spec: TransformSpec
    verification: VerificationResult


_RefSource = Iterable[SequenceRef] | Callable[[], Iterable[SequenceRef]]


class BridgeFinder:
    """Search for ``Transform(sequence_A) == sequence_B`` finite bridges.

    Target sequences are fingerprinted once per target range/profile and cached
    in the catalog. Each transformed source is hashed using the same exact
    profile; matching target hashes create E0 candidates which are then passed
    to :class:`CollisionDetector`. Only the verifier can return an E1 claim.

    The initial implementation supports one transform hop per search. This
    avoids silently composing index maps or domains; multi-hop searches require
    explicit composition semantics in a later version.
    """

    def __init__(
        self,
        term_store: TermStore,
        hash_index: HashIndex,
        collision_detector: CollisionDetector,
    ) -> None:
        if not isinstance(term_store, TermStore):
            raise TypeError("term_store must be a TermStore")
        if not isinstance(hash_index, HashIndex):
            raise TypeError("hash_index must be a HashIndex")
        if not isinstance(collision_detector, CollisionDetector):
            raise TypeError("collision_detector must be a CollisionDetector")
        if hash_index.term_store is not term_store:
            raise ValueError("hash index and bridge finder must share a term store")
        if collision_detector.term_store is not term_store:
            raise ValueError(
                "collision detector and bridge finder must share a term store"
            )
        self.term_store = term_store
        self.hash_index = hash_index
        self.collision_detector = collision_detector

    def find_bridges(
        self,
        seeds: _RefSource,
        targets: _RefSource,
        transform_specs: Iterable[TransformSpec],
        *,
        compare_terms: int,
        max_depth: int = 1,
    ) -> Iterator[BridgeResult]:
        """Yield candidates and exact results for configured one-hop transforms.

        ``compare_terms`` is the number of output and target terms to compare.
        Transform-specific integer parameters may override:

        - ``input_start_index`` (default 0)
        - ``input_term_count`` (default ``compare_terms``; inverse Euler uses
          ``compare_terms + 1`` because ``a_0..a_N`` yield ``c_1..c_N``)
        - ``target_start_index`` (default 0)
        - ``output_start_index`` (default target start)

        The full ``TransformSpec`` is retained in every generated E1 claim.
        For multiple transforms, use re-iterable collections or factories that
        create a fresh reference iterator for each pass.
        """

        if type(compare_terms) is not int or compare_terms <= 0:
            raise ValueError("compare_terms must be a positive integer")
        if max_depth != 1:
            raise TransformError(
                "BridgeFinder currently supports exactly one transform hop",
                context={"requested_depth": max_depth},
            )

        specs = tuple(self._validate_specs(transform_specs))
        seed_factory = self._ref_factory(
            seeds, "seeds", require_reusable=len(specs) > 1
        )
        target_factory = self._ref_factory(
            targets, "targets", require_reusable=len(specs) > 1
        )

        for spec in specs:
            if not self.collision_detector.has_transform(spec):
                raise TransformError(
                    "bridge transform is not registered",
                    context={"name": spec.name, "version": spec.version},
                )
            input_start = self._integer_parameter(spec, "input_start_index", 0)
            target_start = self._integer_parameter(spec, "target_start_index", 0)
            output_start = self._integer_parameter(
                spec, "output_start_index", target_start
            )
            input_count = self._integer_parameter(
                spec,
                "input_term_count",
                compare_terms + 1 if spec.name == "inverse_euler" else compare_terms,
            )
            if input_count <= 0:
                raise TransformError(
                    "transform input_term_count must be positive",
                    context={"transform": spec.name, "input_term_count": input_count},
                )
            if output_start != target_start:
                raise TransformError(
                    "bridge output and target ranges must start at the same index",
                    context={
                        "output_start_index": output_start,
                        "target_start_index": target_start,
                    },
                )

            target_profile = FingerprintProfile(
                name="bridge-target-exact",
                version="1",
                mode="exact",
                term_count=compare_terms,
                start_index=target_start,
            )
            # Persist/reuse target fingerprints. This stream is bounded by the
            # caller's target iterator, and each fingerprint reads only the
            # requested range from the term store.
            for target in target_factory():
                self.hash_index.fingerprint(target, target_profile)

            for seed in seed_factory():
                source_terms = self.term_store.read_range(
                    seed, input_start, input_count
                )
                dense_input: list[int] = []
                missing_source_index: int | None = None
                for offset in range(input_count):
                    source_index = input_start + offset
                    value = source_terms.value_at(source_index)
                    if value is None:
                        missing_source_index = source_index
                        break
                    dense_input.append(value)
                if missing_source_index is not None:
                    # No transform candidate can be formed from incomplete
                    # input. Do not substitute zero or guess missing terms.
                    continue

                transformed = self.collision_detector.apply_transform(
                    spec, tuple(dense_input)
                )
                if len(transformed) < compare_terms:
                    continue
                output_values = transformed[:compare_terms]
                output_fingerprint = self.hash_index.fingerprint_values(
                    seed,
                    target_profile,
                    output_values,
                    start_index=output_start,
                )

                for target in self.hash_index.find_matching_refs(output_fingerprint):
                    if target == seed:
                        continue
                    candidate = Candidate(
                        source=seed,
                        target=target,
                        profile=target_profile,
                        range_start=target_start,
                        range_end=target_start + compare_terms - 1,
                        digest=output_fingerprint.digest,
                    )
                    self.hash_index.record_candidate(candidate)
                    relation = RelationSpec(
                        relation_kind=(
                            f"transform:{spec.name}@{spec.version}"
                        ),
                        source_start_index=input_start,
                        target_start_index=target_start,
                        term_count=compare_terms,
                        transform_spec=spec,
                        input_term_count=input_count,
                        output_start_index=output_start,
                    )
                    verification = self.collision_detector.verify(
                        candidate, relation
                    )
                    yield BridgeResult(
                        candidate=candidate,
                        transform_spec=spec,
                        verification=verification,
                    )

    def find_snapshot_bridges(
        self,
        snapshot_id: str,
        transform_specs: Iterable[TransformSpec],
        *,
        compare_terms: int,
        max_depth: int = 1,
        page_size: int = 500,
    ) -> Iterator[BridgeResult]:
        """Search one snapshot using bounded Catalog keyset streams.

        Each configured transform performs a target pass followed by a seed
        pass. Catalog queries fetch at most ``page_size`` references at once.
        """

        catalog = self.hash_index.catalog

        def refs() -> Iterator[SequenceRef]:
            return catalog.iter_sequence_refs(snapshot_id, page_size=page_size)

        return self.find_bridges(
            refs,
            refs,
            transform_specs,
            compare_terms=compare_terms,
            max_depth=max_depth,
        )

    @classmethod
    def _ref_factory(
        cls,
        refs: _RefSource,
        name: str,
        *,
        require_reusable: bool,
    ) -> Callable[[], Iterator[SequenceRef]]:
        if callable(refs):
            def from_factory() -> Iterator[SequenceRef]:
                return cls._validate_refs(refs(), name)

            return from_factory

        first_iterator = iter(refs)
        if require_reusable and first_iterator is refs:
            raise ValidationError(
                f"multiple transforms require a re-iterable {name} source "
                "or an iterator factory"
            )
        if first_iterator is refs:
            consumed = False

            def from_single_iterator() -> Iterator[SequenceRef]:
                nonlocal consumed
                if consumed:
                    return iter(())
                consumed = True
                return cls._validate_refs(first_iterator, name)

            return from_single_iterator

        def from_iterable() -> Iterator[SequenceRef]:
            return cls._validate_refs(iter(refs), name)

        return from_iterable

    @staticmethod
    def _validate_refs(
        refs: Iterable[SequenceRef],
        name: str,
    ) -> Iterator[SequenceRef]:
        for position, ref in enumerate(refs):
            if not isinstance(ref, SequenceRef):
                raise TypeError(f"{name}[{position}] must be a SequenceRef")
            yield ref

    @staticmethod
    def _validate_specs(
        specs: Iterable[TransformSpec],
    ) -> Iterator[TransformSpec]:
        for position, spec in enumerate(specs):
            if not isinstance(spec, TransformSpec):
                raise TypeError(
                    f"transform_specs[{position}] must be a TransformSpec"
                )
            yield spec

    @staticmethod
    def _integer_parameter(
        spec: TransformSpec,
        name: str,
        default: int,
    ) -> int:
        value = spec.parameters.get(name, default)
        if type(value) is not int:
            raise ValidationError(
                f"transform parameter {name!r} must be an integer",
                context={"transform": spec.name, "value": value},
            )
        return value
