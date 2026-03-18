import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from streamlit_langgraph.utils.file_handler import FileHandler
except Exception as exc:
    raise unittest.SkipTest(f"FileHandler tests unavailable in this environment: {exc}")


class FakeUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content
        self.file_id = name

    def getvalue(self):
        return self._content


class FakeVectorStores:
    def __init__(self, retrieve_exception=None):
        self.retrieve_exception = retrieve_exception
        self.created = 0

    def retrieve(self, vector_store_id):
        if self.retrieve_exception is not None:
            raise self.retrieve_exception
        return SimpleNamespace(id=vector_store_id)

    def create(self, name):
        self.created += 1
        return SimpleNamespace(id="vs_new")


class FakeOpenAIClient:
    def __init__(self, vector_stores):
        self.vector_stores = vector_stores


class FileHandlerTests(unittest.TestCase):
    def test_vector_store_not_found_creates_new_store(self):
        class FakeNotFound(Exception):
            pass

        vector_stores = FakeVectorStores(retrieve_exception=FakeNotFound("missing"))
        handler = FileHandler(openai_client=FakeOpenAIClient(vector_stores), allow_file_search=True)
        handler._vector_store_ids = ["vs_old"]

        with patch("streamlit_langgraph.utils.file_handler.NotFoundError", FakeNotFound):
            vector_store = handler._get_or_create_vector_store()

        self.assertEqual(vector_store.id, "vs_new")
        self.assertEqual(vector_stores.created, 1)
        shutil.rmtree(handler.temp_dir, ignore_errors=True)

    def test_vector_store_unexpected_error_is_raised(self):
        vector_stores = FakeVectorStores(retrieve_exception=RuntimeError("boom"))
        handler = FileHandler(openai_client=FakeOpenAIClient(vector_stores), allow_file_search=True)
        handler._vector_store_ids = ["vs_old"]

        with self.assertRaises(RuntimeError):
            handler._get_or_create_vector_store()
        shutil.rmtree(handler.temp_dir, ignore_errors=True)

    def test_reset_does_not_delete_uploaded_file_from_temp_dir(self):
        temp_dir = tempfile.mkdtemp(prefix="slg-test-")
        handler = FileHandler(temp_dir=temp_dir)
        handler.track(FakeUploadedFile("upload.txt", b"hello"))

        uploaded_path = Path(temp_dir) / "upload.txt"
        self.assertTrue(uploaded_path.exists())

        handler.reset()

        self.assertTrue(uploaded_path.exists())
        self.assertTrue(Path(temp_dir).exists())
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
