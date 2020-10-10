import ray, os
from typing import List, Callable

ray.init(num_cpus=os.cpu_count(), num_gpus=1)


@ray.remote
def proxy(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def apply_para(func: List[Callable], args_list: List):
    rid_list = [proxy.remote(func, *args) for args in args_list]
    return [ray.get(rid) for rid in rid_list]

