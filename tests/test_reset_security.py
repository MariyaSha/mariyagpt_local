"""Exercise the real reset route and proxy without loading models or PostgreSQL."""
import ast
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]


class ResetSecurityTests(unittest.TestCase):
    def setUp(self):
        # Compile the actual route and CORS configuration, isolating only the
        # unrelated model imports and database initialization.
        tree = ast.parse((ROOT / "backend/app.py").read_text())
        nodes = [node for node in tree.body if (
            isinstance(node, ast.FunctionDef) and node.name == "reset_setup"
        ) or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "add_middleware"
        )]
        self.runtime = {"name": "Test", "model": "test-model"}
        self.unload = Mock()
        app = FastAPI()
        namespace = dict(app=app, HTTPException=HTTPException, Request=Request,
                         CORSMiddleware=CORSMiddleware,
                         FRONTEND_URL="http://127.0.0.1:5000",
                         runtime=self.runtime, unload_model=self.unload)
        exec(compile(ast.Module(body=nodes, type_ignores=[]),
                     "backend/app.py", "exec"), namespace)
        self.backend = TestClient(app)
        spec = importlib.util.spec_from_file_location("frontend_app", ROOT / "frontend/app.py")
        frontend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frontend)
        self.frontend = frontend.app.test_client()
        self.forward = patch.object(frontend.requests, "request", side_effect=self.proxy)
        self.forward.start()
        self.addCleanup(self.forward.stop)

    def proxy(self, method, url, headers, data, params, **kwargs):
        result = self.backend.request(method, url, headers=headers, content=data, params=params)
        return Mock(status_code=result.status_code, headers=result.headers,
                    iter_content=lambda chunk_size: iter([result.content]))

    def assert_unchanged(self):
        self.assertEqual(self.runtime, {"name": "Test", "model": "test-model"})
        self.unload.assert_not_called()

    def test_form_posts_cannot_reset_directly_or_through_proxy(self):
        for client in (self.backend, self.frontend):
            for origin in (None, "http://attacker.example", "null",
                           "http://127.0.0.1:5000"):
                for content_type in ("application/x-www-form-urlencoded",
                                     "multipart/form-data", "text/plain"):
                    with self.subTest(client=type(client).__name__, origin=origin,
                                      content_type=content_type):
                        headers = {"Content-Type": content_type}
                        if origin is not None:
                            headers["Origin"] = origin
                        response = client.post("/api/reset", headers=headers)
                        self.assertEqual(response.status_code, 403)
                        self.assert_unchanged()

    def test_wrong_header_cannot_reset(self):
        response = self.frontend.post("/api/reset", headers={"X-Requested-With": "wrong"})
        self.assertEqual(response.status_code, 403)
        self.assert_unchanged()

    def test_untrusted_preflight_cannot_authorize_custom_header(self):
        for client in (self.backend, self.frontend):
            response = client.options("/api/reset", headers={
                "Origin": "http://attacker.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-Requested-With",
            })
            self.assertNotIn("access-control-allow-origin", response.headers)
            self.assert_unchanged()

    def test_frontend_reset_still_works_through_proxy(self):
        response = self.frontend.post("/api/reset", headers={
            "Origin": "http://localhost:5000",
            "X-Requested-With": "XMLHttpRequest",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"ready": False})
        self.assertEqual(self.runtime, {"name": None, "model": None})
        self.unload.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
