"""
Shared pytest fixtures for QuantImage v2 backend tests.

Provides:
- Flask test application with in-memory SQLite database
- Database session management with automatic cleanup
- Mocked authentication (Keycloak JWT bypass)
- Celery task mocking
- Test client for HTTP endpoint testing
"""

import os
import sys
import json
import types
import datetime
from datetime import timezone
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Environment stubs – must be set BEFORE any application code is imported
# because several modules read os.environ at import time.
# ---------------------------------------------------------------------------
_ENV_DEFAULTS = {
    "DB_DATABASE": "test_quantimage2",
    "DB_USER": "test_user",
    "KEYCLOAK_BASE_URL": "http://localhost:8081/auth/",
    "KEYCLOAK_REALM_NAME": "QuantImage-v2",
    "KEYCLOAK_QUANTIMAGE2_FRONTEND_CLIENT_ID": "quantimage2-frontend",
    "KEYCLOAK_FRONTEND_ADMIN_ROLE": "admin",
    "KHEOPS_BASE_URL": "http://localhost",
    "CELERY_BROKER_URL": "memory://",
    "CELERY_RESULT_BACKEND": "cache+memory://",
    "SOCKET_MESSAGE_QUEUE": "memory://",
    "CORS_ALLOWED_ORIGINS": "http://localhost:3000",
    "GRID_SEARCH_CONCURRENCY": "1",
}

for key, value in _ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

# Add project paths so imports resolve (handles both local and Docker layouts)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for subdir in ("shared", "webapp", "workers"):
    path = os.path.join(_PROJECT_ROOT, subdir)
    if path not in sys.path:
        sys.path.insert(0, path)

# Docker layout: workers are mounted at /usr/src/workers
_DOCKER_WORKERS = "/usr/src/workers"
if os.path.isdir(_DOCKER_WORKERS) and _DOCKER_WORKERS not in sys.path:
    sys.path.insert(0, _DOCKER_WORKERS)

# ---------------------------------------------------------------------------
# Mocks that must be in place before importing application modules
# ---------------------------------------------------------------------------

# Mock docker secrets (no secret files in test environment)
_mock_secret = patch(
    "get_docker_secret.get_docker_secret", return_value="test_password"
)
_mock_secret.start()

# Mock Keycloak client – must be injected into sys.modules BEFORE
# ``config`` is imported so that the module-level ``oidc_client`` uses it.
_mock_keycloak = MagicMock()
_mock_keycloak.public_key.return_value = "FAKEPUBLICKEY"
_mock_keycloak.decode_token.return_value = {
    "sub": "test-user-uuid-1234",
    "preferred_username": "testuser",
    "resource_access": {
        "quantimage2-frontend": {"roles": ["admin"]},
    },
    "exp": int(datetime.datetime.now(timezone.utc).timestamp()) + 3600,
}

# Create a synthetic ``config`` module so that ``from config import oidc_client``
# works without having a real Keycloak server.
_fake_config = types.ModuleType("config")
_fake_config.oidc_client = _mock_keycloak
_fake_config.EXTRACTIONS_BASE_DIR = "/tmp/quantimage2-test/extractions"
_fake_config.CONFIGS_SUBDIR = "configs"
_fake_config.FEATURES_CACHE_BASE_DIR = "/tmp/quantimage2-test/features-cache"
sys.modules.setdefault("config", _fake_config)

# Create a synthetic ``config_workers`` module used by workers/utils.py
_fake_config_workers = types.ModuleType("config_workers")
_fake_config_workers.MODELS_BASE_DIR = "/tmp/quantimage2-test/models"
sys.modules.setdefault("config_workers", _fake_config_workers)


# ---------------------------------------------------------------------------
# Flask app fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def app():
    """Create a Flask application configured for testing (in-memory SQLite).

    We build the Flask app manually instead of using create_app() to avoid
    connecting to MySQL.  Flask-SQLAlchemy 3.x caches the engine config at
    init_app() time, so overriding after create_app() is too late.
    """
    from flask import Flask
    from quantimage2_backend_common.models import db

    test_app = Flask(__name__)
    test_app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    test_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    test_app.config["SQLALCHEMY_ECHO"] = False
    test_app.config["TESTING"] = True
    test_app.config["SERVER_NAME"] = "localhost"
    test_app.config["UPLOAD_FOLDER"] = "/tmp/quantimage2-test/feature-presets"
    test_app.config["CELERY_BROKER_URL"] = "memory://"
    test_app.config["CELERY_RESULT_BACKEND"] = "cache+memory://"
    test_app.json.sort_keys = False

    # Ensure upload folder exists
    os.makedirs(test_app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Initialize SQLAlchemy once with the test app
    db.init_app(test_app)

    # Register MySQL 'latin1_bin' collation for SQLite (used by ROI model)
    from sqlalchemy import event

    with test_app.app_context():

        @event.listens_for(db.engine, "connect")
        def _register_collations(dbapi_conn, connection_record):
            dbapi_conn.create_collation("latin1_bin", lambda a, b: (a > b) - (a < b))

        db.create_all()

    # Register blueprints for route testing
    from routes.features import bp as features_bp
    from routes.feature_presets import bp as feature_presets_bp
    from routes.feature_collections import bp as feature_collections_bp
    from routes.tasks import bp as tasks_bp
    from routes.models import bp as models_bp
    from routes.labels import bp as labels_bp
    from routes.charts import bp as charts_bp
    from routes.navigation_history import bp as navigation_bp
    from routes.albums import bp as albums_bp
    from routes.clinical_features import bp as clinical_features_bp

    with test_app.app_context():
        test_app.register_blueprint(features_bp)
        test_app.register_blueprint(feature_presets_bp)
        test_app.register_blueprint(feature_collections_bp)
        test_app.register_blueprint(tasks_bp)
        test_app.register_blueprint(models_bp)
        test_app.register_blueprint(labels_bp)
        test_app.register_blueprint(charts_bp)
        test_app.register_blueprint(navigation_bp)
        test_app.register_blueprint(albums_bp)
        test_app.register_blueprint(clinical_features_bp)

    # Attach mock Celery + SocketIO to the app
    test_app.my_celery = MagicMock()
    test_app.my_celery.send_task = MagicMock(return_value=MagicMock(id="mock-task-id"))
    test_app.my_socketio = MagicMock()

    # Error handler
    from quantimage2_backend_common.utils import InvalidUsage
    from flask import jsonify

    @test_app.errorhandler(InvalidUsage)
    def handle_invalid_usage(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @test_app.route("/")
    def entrypoint():
        return {"status": "QuantImage v2 Backend is running"}

    @test_app.route("/test-error")
    def trigger_error():
        raise InvalidUsage("Test error message")

    yield test_app

    with test_app.app_context():
        db.drop_all()


@pytest.fixture(scope="function")
def db_session(app):
    """Provide a clean database session for each test.

    Drops and recreates all tables so each test starts with an empty DB.
    """
    from quantimage2_backend_common.models import db

    with app.app_context():
        db.drop_all()
        db.create_all()
        yield db.session
        db.session.rollback()
        db.session.remove()


@pytest.fixture(scope="function")
def client(app):
    """Flask test client with authentication headers pre-set."""
    with app.test_client() as test_client:
        # Inject a fake Bearer token in every request
        test_client.environ_base["HTTP_AUTHORIZATION"] = "Bearer fake-jwt-token"
        yield test_client


@pytest.fixture(scope="function")
def auth_context(app):
    """Push an authenticated request context (sets g.user and g.token)."""
    from flask import g

    with app.test_request_context(headers={"Authorization": "Bearer fake-jwt-token"}):
        g.user = "test-user-uuid-1234"
        g.token = "fake-jwt-token"
        yield g


# ---------------------------------------------------------------------------
# Celery mocking
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_celery(app):
    """Mock Celery send_task so no real broker is needed."""
    mock = MagicMock()
    mock.send_task = MagicMock(return_value=MagicMock(id="mock-task-id"))
    original = getattr(app, "my_celery", None)
    app.my_celery = mock
    yield mock
    app.my_celery = original


@pytest.fixture
def mock_socketio():
    """Mock SocketIO so no real Redis message queue is needed."""
    mock = MagicMock()
    mock.emit = MagicMock()
    yield mock


# ---------------------------------------------------------------------------
# Kheops API mocking
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_kheops(responses_fixture):
    """Pre-configure responses mock for common Kheops API calls."""
    import responses as responses_lib

    base_url = os.environ["KHEOPS_BASE_URL"]

    # Mock album details
    responses_lib.add(
        responses_lib.GET,
        f"{base_url}/api/albums/test-album-id",
        json={"album_id": "test-album-id", "name": "Test Album"},
        status=200,
    )

    # Mock studies list
    responses_lib.add(
        responses_lib.GET,
        f"{base_url}/api/studies",
        json=[],
        status=200,
    )

    yield


@pytest.fixture
def responses_fixture():
    """Activate the responses library for HTTP mocking."""
    import responses as responses_lib

    with responses_lib.RequestsMock() as rsps:
        yield rsps


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_feature_extraction(db_session):
    """Create a sample FeatureExtraction record."""
    from quantimage2_backend_common.models import FeatureExtraction

    extraction = FeatureExtraction(
        user_id="test-user-uuid-1234",
        album_id="test-album-id",
    )
    extraction.save_to_db()
    return extraction


@pytest.fixture
def sample_feature_preset(db_session):
    """Create a sample FeaturePreset record."""
    from quantimage2_backend_common.models import FeaturePreset

    preset = FeaturePreset("Test Preset", "/tmp/test_preset.yaml")
    preset.save_to_db()
    return preset
