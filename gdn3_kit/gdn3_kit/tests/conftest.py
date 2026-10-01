import os


def pytest_generate_tests(metafunc):
    if "kind" in metafunc.fixturenames:
        metafunc.parametrize("kind", os.environ.get("GDN3_BACKEND", "serial").split(","))
