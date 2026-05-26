"""
Distributed Transaction Orchestrator using the Saga Pattern.

Pipeline Improvement:
  Provides consistency guarantees when writing to both GraphDB (Neo4j) and
  VectorDB (Qdrant). Each step in a saga has a corresponding compensating
  transaction (rollback) that undoes its effects if a later step fails.

Saga Pattern Overview:
  A Saga is a sequence of local transactions where each transaction updates
  the database and publishes an event or signal to trigger the next one.
  If a failure occurs, compensating transactions are executed in reverse order
  to undo all previously completed steps.

Usage:
  orchestrator = SagaOrchestrator()
  result = orchestrator.execute(
      doc_id="doc_001",
      content="# My Document\n...",
      metadata={"source": "upload"},
  )
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Type alias for a saga step: (forward_fn, compensating_fn)
SagaStep = Tuple[
    Callable[..., Dict[str, Any]],
    Callable[[Dict[str, Any]], None],
]


class SagaExecutionError(Exception):
    """Raised when a Saga execution fails and compensation has been triggered."""
    def __init__(self, message: str, failed_step: int, errors: List[str]):
        super().__init__(message)
        self.failed_step = failed_step
        self.errors = errors


class SagaOrchestrator:
    """
    Orchestrates distributed transactions using the Saga Pattern.

    Coordinates writes to GraphDB and VectorDB with automatic compensating
    transactions (rollback) when any step fails.
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the Saga orchestrator.

        Args:
            max_retries: Maximum number of retries for each individual step.
            retry_delay: Base delay between retries (exponential backoff).
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def execute(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        steps: Optional[List[SagaStep]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a Saga with the given steps.

        Each step is a tuple of (forward_fn, compensating_fn). The forward
        function performs the actual work and returns a result dict. The
        compensating function undoes the work given the result dict.

        Args:
            doc_id: Unique document identifier.
            content: Document content to store.
            metadata: Optional metadata to attach.
            steps: List of saga steps. If None, uses default GraphDB+VectorDB steps.

        Returns:
            A dict with execution results for all steps.

        Raises:
            SagaExecutionError: If any step fails after compensation.
        """
        if steps is None:
            steps = self._build_default_steps(doc_id, content, metadata)

        completed_steps: List[Tuple[int, Dict[str, Any]]] = []
        errors: List[str] = []

        try:
            for step_index, (forward_fn, _compensate_fn) in enumerate(steps):
                step_name = forward_fn.__name__ if hasattr(forward_fn, '__name__') else f"step_{step_index}"
                result = self._execute_step_with_retry(forward_fn, step_name)
                completed_steps.append((step_index, result))
                logger.info(f"Saga step {step_index} ({step_name}) completed successfully")

            logger.info(f"Saga for document {doc_id} completed successfully ({len(completed_steps)} steps)")
            return {
                'status': 'success',
                'doc_id': doc_id,
                'steps_completed': len(completed_steps),
                'results': {i: r for i, r in completed_steps},
            }

        except Exception as e:
            error_msg = f"Saga step failed for document {doc_id}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

            # Execute compensating transactions in reverse order
            self._compensate(completed_steps, steps, doc_id)

            raise SagaExecutionError(
                message=f"Saga failed for document {doc_id} after step {len(completed_steps)}. Compensation executed.",
                failed_step=len(completed_steps),
                errors=errors,
            )

    def _execute_step_with_retry(
        self,
        fn: Callable[..., Dict[str, Any]],
        step_name: str,
    ) -> Dict[str, Any]:
        """
        Execute a saga step with retry and exponential backoff.

        Args:
            fn: The forward function to execute.
            step_name: Human-readable name for logging.

        Returns:
            Result dict from the forward function.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"Step '{step_name}' attempt {attempt}/{self.max_retries} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Step '{step_name}' failed after {self.max_retries} attempts: {e}"
                    )
        raise last_error  # type: ignore[misc]

    def _compensate(
        self,
        completed_steps: List[Tuple[int, Dict[str, Any]]],
        all_steps: List[SagaStep],
        doc_id: str,
    ) -> None:
        """
        Execute compensating transactions in reverse order.

        Args:
            completed_steps: List of (step_index, result) for completed steps.
            all_steps: Full list of saga steps (for accessing compensate functions).
            doc_id: Document identifier for logging.
        """
        logger.info(
            f"Starting compensation for document {doc_id} "
            f"({len(completed_steps)} steps to undo)"
        )

        for step_index, result in reversed(completed_steps):
            _forward_fn, compensate_fn = all_steps[step_index]
            step_name = _forward_fn.__name__ if hasattr(_forward_fn, '__name__') else f"step_{step_index}"
            try:
                compensate_fn(result)
                logger.info(f"Compensating transaction for '{step_name}' completed")
            except Exception as e:
                logger.error(
                    f"ERROR in compensating transaction for '{step_name}': {e}. "
                    f"Data inconsistency may exist for document {doc_id}."
                )

    # --- Default Saga Steps (GraphDB + VectorDB) ---

    def _build_default_steps(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[SagaStep]:
        """
        Build the default saga steps for storing a document in both GraphDB and VectorDB.

        Steps:
          1. Store in GraphDB (Neo4j) -> compensate: delete from GraphDB
          2. Store in VectorDB (Qdrant) -> compensate: delete from VectorDB

        Args:
            doc_id: Unique document identifier.
            content: Document content.
            metadata: Optional metadata.

        Returns:
            List of saga steps as (forward_fn, compensating_fn) tuples.
        """

        # --- Step 1: GraphDB ---
        def store_graph_db() -> Dict[str, Any]:
            from graph_db import get_graph_db
            db = get_graph_db()
            success = db.create_nodes_and_relationships(content, doc_id)
            if not success:
                raise RuntimeError(f"Failed to store document {doc_id} in GraphDB")
            return {'doc_id': doc_id, 'stored_in': 'graph_db'}

        def compensate_graph_db(result: Dict[str, Any]) -> None:
            from graph_db import get_graph_db
            rid = result.get('doc_id', doc_id)
            try:
                db = get_graph_db()
                with db.driver.session() as session:
                    session.run(
                        "MATCH (d:Document {id: $doc_id}) DETACH DELETE d",
                        doc_id=rid,
                    )
                logger.info(f"Compensated: deleted document {rid} from GraphDB")
            except Exception as e:
                logger.error(f"Failed to compensate GraphDB for {rid}: {e}")

        # --- Step 2: VectorDB ---
        def store_vector_db() -> Dict[str, Any]:
            from vector_db import get_vector_db
            db = get_vector_db()
            success = db.store_document_embeddings(
                doc_id=doc_id,
                content=content,
                metadata=metadata,
            )
            if not success:
                raise RuntimeError(f"Failed to store document {doc_id} in VectorDB")
            return {'doc_id': doc_id, 'stored_in': 'vector_db'}

        def compensate_vector_db(result: Dict[str, Any]) -> None:
            from vector_db import get_vector_db
            rid = result.get('doc_id', doc_id)
            try:
                db = get_vector_db()
                vector_db_config = db.client.get_collection(
                    collection_name=db.client.get_collection_info(
                        collection_name='markdown_embeddings'
                    ).name if hasattr(db, 'client') else 'markdown_embeddings'
                )
                # Delete all points for this document
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                db.client.delete(
                    collection_name='markdown_embeddings',
                    points_selector=Filter(
                        must=[FieldCondition(key='doc_id', match=MatchValue(value=rid))]
                    ),
                )
                logger.info(f"Compensated: deleted document {rid} from VectorDB")
            except Exception as e:
                logger.error(f"Failed to compensate VectorDB for {rid}: {e}")

        return [
            (store_graph_db, compensate_graph_db),
            (store_vector_db, compensate_vector_db),
        ]


# --- Convenience function ---

def store_document_saga(
    doc_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Store a document in both GraphDB and VectorDB using the Saga Pattern.

    This is a convenience wrapper around SagaOrchestrator that uses the
    default steps (GraphDB write -> VectorDB write).

    Args:
        doc_id: Unique document identifier.
        content: Document content to store.
        metadata: Optional metadata to attach.

    Returns:
        Execution result dict with status and step details.

    Raises:
        SagaExecutionError: If any step fails (compensation is automatic).
    """
    orchestrator = SagaOrchestrator()
    return orchestrator.execute(
        doc_id=doc_id,
        content=content,
        metadata=metadata,
    )
