import io
import logging
import os
import os.path as osp
import pickle
from datetime import timedelta
from typing import Any, List, Tuple

from manifold.clients.python.manifold_client import ManifoldClient

logger = logging.getLogger(__name__)

# CONSTANTS
MANIFOLD_TTL = timedelta(days=60)
MANIFOLD_TIMEOUT = timedelta(seconds=60)
MANIFOLD_RETRIES = 100


def create_manifold_directory(path: str) -> None:
    manifold_bucket, manifold_directory = split_bucket_and_file_location(path)
    with ManifoldClient(manifold_bucket) as manifold_client:
        if not manifold_client.sync_exists(manifold_directory):
            manifold_client.sync_mkdir(
                path=manifold_directory,
                numRetries=MANIFOLD_RETRIES,
                ttl=MANIFOLD_TTL,
                timeout=MANIFOLD_TIMEOUT,
                userData=False,
                recursive=True,
            )


def delete_manifold_directory_files(path: str) -> None:
    bucket, dir_location = split_bucket_and_file_location(path)
    with ManifoldClient(bucket) as manifold_client:
        files = list(manifold_client.sync_ls(path=dir_location))
        for file_name, _ in files:
            manifold_client.sync_rm(os.path.join(dir_location, file_name))


def delete_manifold_file(path: str) -> None:
    bucket, file_path = split_bucket_and_file_location(path)
    with ManifoldClient(bucket) as manifold_client:
        manifold_client.sync_rm(file_path)


def get_location_on_manifold(directory: str, filename: str) -> str:
    return os.path.join("tree", directory, filename)


def split_bucket_and_file_location(path: str) -> Tuple[str, str]:
    path_split = path.split("/")
    bucket = path_split[0]
    file_location = osp.join(*path_split[1:])
    return bucket, file_location


def file_exists_on_manifold(path: str) -> bool:
    bucket, file_location = split_bucket_and_file_location(path)
    with ManifoldClient(bucket) as manifold_client:
        return manifold_client.sync_exists(file_location)


def list_all_files(path: str) -> List[str]:
    bucket, file_location = split_bucket_and_file_location(path)
    with ManifoldClient(bucket) as manifold_client:
        files = list(manifold_client.sync_ls(path=file_location))

    return [file_name for file_name, _ in files]


def save_data_to_manifold(path: str, data: Any, is_pkl=True) -> None:
    bucket, file_location = split_bucket_and_file_location(path)
    with ManifoldClient(bucket) as manifold_client:

        p = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL) if is_pkl else data

        manifold_client.sync_put(
            path=file_location,
            input=io.BytesIO(p),
            ttl=MANIFOLD_TTL,
            timeout=MANIFOLD_TIMEOUT,
            numRetries=MANIFOLD_RETRIES,
            userData=False,
            predicate=ManifoldClient.Predicates.AllowOverwrite,
        )


def read_data_from_manifold(path: str, is_from_pkl: bool = True) -> Any:
    bucket, file_location = split_bucket_and_file_location(path)

    with ManifoldClient(bucket) as manifold_client:

        # container to store the blob
        output = io.BytesIO()

        manifold_client.sync_get(
            path=file_location,
            output=output,
            timeout=MANIFOLD_TIMEOUT,
            numRetries=MANIFOLD_RETRIES,
        )

        # load the model from BytesIO stream
        data = pickle.loads(output.getvalue()) if is_from_pkl else output

    return data
