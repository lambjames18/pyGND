"""Small shared helpers used across the pygnd package."""

import contextlib
import joblib


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager that patches joblib to report progress into a tqdm bar.

    Args:
        tqdm_object: a `tqdm` progress bar instance to update as joblib tasks complete.

    Yields:
        The same `tqdm_object`, updated automatically as each parallel batch finishes.
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
