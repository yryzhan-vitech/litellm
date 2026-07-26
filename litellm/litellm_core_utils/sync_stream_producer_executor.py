# [ARC-BUG-03] dedicated bounded executor for sync-stream producers, kept separate from
# the general-purpose thread_pool_executor so streaming traffic cannot starve it.
from concurrent.futures import ThreadPoolExecutor

from litellm.constants import MAX_SYNC_STREAM_PRODUCER_THREADS

sync_stream_producer_executor = ThreadPoolExecutor(max_workers=MAX_SYNC_STREAM_PRODUCER_THREADS)
