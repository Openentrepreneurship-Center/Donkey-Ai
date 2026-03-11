"""
임시 파일 관리 - TemporaryDirectory 기반, 자동 삭제
"""
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


@contextmanager
def temp_directory() -> Generator[str, None, None]:
    """
    임시 디렉토리 생성, 컨텍스트 종료 시 자동 삭제
    Yields:
        임시 디렉토리 경로
    """
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def save_upload_to_disk(file_obj, dest_path: str) -> str:
    """
    UploadFile.file (file-like)을 디스크에 저장 (RAM 전체 로드 방지)
    shutil.copyfileobj로 스트리밍 저장
    """
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file_obj, buffer)
    return dest_path
