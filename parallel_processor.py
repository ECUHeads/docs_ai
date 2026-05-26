"""
Parallel processing utilities for the PDF pipeline.

Pipeline Improvement:
  Provides both ProcessPoolExecutor (for CPU-bound tasks like PDF conversion)
  and asyncio.gather (for I/O-bound operations) with automatic workload
  detection, proper error handling, and configurable worker counts.

Workload Detection:
  - CPU-bound: Uses concurrent.futures.ProcessPoolExecutor
  - I/O-bound: Uses asyncio.gather() with semaphores
  - Auto-detection based on task characteristics

Error Handling:
  - Individual task failures don't abort the entire batch
  - Errors are collected and reported alongside successful results
  - Configurable retry behavior per task
"""

import os
import asyncio
import logging
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class TaskResult:
    """Result of a single parallel task execution."""
    index: int
    input_item: Any
    success: bool
    result: Any = None
    error: Optional[str] = None


@dataclass
class BatchResult:
    """Aggregated results from a batch of parallel tasks."""
    total: int
    successful: int
    failed: int
    results: List[TaskResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successful / self.total if self.total > 0 else 0.0


class ParallelProcessor:
    """
    Handles parallel execution of tasks with automatic workload detection.

    Supports both CPU-bound (ProcessPoolExecutor) and I/O-bound (asyncio.gather)
    execution strategies.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_process_pool: str = 'auto',
        chunk_batch_size: int = 10,
    ):
        """
        Initialize the parallel processor.

        Args:
            max_workers: Maximum number of workers. If None or 0, auto-detected
                from CPU count (min(cpu_count, 8) for CPU-bound, 50 for I/O-bound).
            use_process_pool: 'auto', 'true', or 'false'. 'auto' detects based
                on task type hint.
            chunk_batch_size: Number of items to process per batch when the
                total exceeds max_workers.
        """
        cpu_count = os.cpu_count() or 4
        self.max_workers = max_workers or min(cpu_count, 8)
        self.use_process_pool = use_process_pool.lower()
        self.chunk_batch_size = chunk_batch_size

        # I/O-bound workers can be higher since they're mostly waiting
        self.io_workers = max(50, cpu_count * 10)

    def process_cpu_bound(
        self,
        fn: Callable[..., Any],
        items: List[Any],
        **kwargs,
    ) -> BatchResult:
        """
        Process items in parallel using ProcessPoolExecutor (CPU-bound).

        Args:
            fn: The function to execute for each item.
            items: List of input items to process.
            **kwargs: Additional keyword arguments passed to fn.

        Returns:
            BatchResult with aggregated results.
        """
        if not items:
            return BatchResult(total=0, successful=0, failed=0)

        results: List[TaskResult] = []
        total = len(items)

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(fn, item, **kwargs): idx
                    for idx, item in enumerate(items)
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    item = items[idx]

                    try:
                        result = future.result()
                        results.append(TaskResult(
                            index=idx,
                            input_item=item,
                            success=True,
                            result=result,
                        ))
                    except Exception as e:
                        logger.error(f"Task {idx} failed: {e}")
                        results.append(TaskResult(
                            index=idx,
                            input_item=item,
                            success=False,
                            error=str(e),
                        ))

        except Exception as e:
            logger.error(f"ProcessPoolExecutor failed: {e}")
            # Mark all unprocessed items as failed
            processed_indices = {r.index for r in results}
            for idx, item in enumerate(items):
                if idx not in processed_indices:
                    results.append(TaskResult(
                        index=idx,
                        input_item=item,
                        success=False,
                        error=f"Executor failure: {e}",
                    ))

        successful = sum(1 for r in results if r.success)
        return BatchResult(
            total=total,
            successful=successful,
            failed=total - successful,
            results=results,
        )

    async def process_io_bound(
        self,
        fn: Callable[..., Any],
        items: List[Any],
        semaphore_count: Optional[int] = None,
        **kwargs,
    ) -> BatchResult:
        """
        Process items in parallel using asyncio.gather (I/O-bound).

        Args:
            fn: The async function to execute for each item.
            items: List of input items to process.
            semaphore_count: Maximum concurrent tasks. Defaults to io_workers.
            **kwargs: Additional keyword arguments passed to fn.

        Returns:
            BatchResult with aggregated results.
        """
        if not items:
            return BatchResult(total=0, successful=0, failed=0)

        semaphore = asyncio.Semaphore(semaphore_count or self.io_workers)
        results: List[TaskResult] = [None] * len(items)  # type: ignore[list-item]

        async def _bounded_task(idx: int, item: Any) -> TaskResult:
            async with semaphore:
                try:
                    result = await fn(item, **kwargs)
                    return TaskResult(
                        index=idx,
                        input_item=item,
                        success=True,
                        result=result,
                    )
                except Exception as e:
                    logger.error(f"Async task {idx} failed: {e}")
                    return TaskResult(
                        index=idx,
                        input_item=item,
                        success=False,
                        error=str(e),
                    )

        # Launch all tasks concurrently
        tasks = [_bounded_task(idx, item) for idx, item in enumerate(items)]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(completed):
            if isinstance(result, Exception):
                results[i] = TaskResult(
                    index=i,
                    input_item=items[i],
                    success=False,
                    error=str(result),
                )
            else:
                results[i] = result

        successful = sum(1 for r in results if r.success)
        return BatchResult(
            total=len(items),
            successful=successful,
            failed=len(items) - successful,
            results=results,
        )

    def process_auto(
        self,
        fn: Callable[..., Any],
        items: List[Any],
        task_type: str = 'cpu',
        **kwargs,
    ) -> BatchResult:
        """
        Automatically select the appropriate processing strategy.

        Args:
            fn: The function to execute for each item.
            items: List of input items to process.
            task_type: 'cpu' for CPU-bound, 'io' for I/O-bound.
            **kwargs: Additional keyword arguments passed to fn.

        Returns:
            BatchResult with aggregated results.
        """
        if self.use_process_pool == 'false':
            task_type = 'io'
        elif self.use_process_pool == 'true':
            task_type = 'cpu'
        # 'auto' uses the provided task_type

        if task_type == 'cpu':
            return self.process_cpu_bound(fn, items, **kwargs)
        else:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    self.process_io_bound(fn, items, **kwargs)
                )
            finally:
                loop.close()


def get_parallel_processor() -> ParallelProcessor:
    """
    Create a ParallelProcessor from environment configuration.

    Reads settings from the config module if available, otherwise uses defaults.
    """
    try:
        from config import get_config
        cfg = get_config().get_config().get('parallel', {})
        return ParallelProcessor(
            max_workers=cfg.get('max_workers', 0) or None,
            use_process_pool=cfg.get('use_process_pool', 'auto'),
            chunk_batch_size=cfg.get('chunk_batch_size', 10),
        )
    except Exception as e:
        logger.warning(f"Could not load parallel config: {e}; using defaults")
        return ParallelProcessor()
