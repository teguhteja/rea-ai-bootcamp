# conftest.py
import pytest

@pytest.hookimpl(tryfirst=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    total = terminalreporter._numcollected
    passed = len(terminalreporter.stats.get('passed', []))
    failed = len(terminalreporter.stats.get('failed', []))
    skipped = len(terminalreporter.stats.get('skipped', []))
    duration = terminalreporter._sessionstarttime - terminalreporter._sessionstarttime

    terminalreporter.write_sep("=", "Test Summary")
    terminalreporter.write_line(f"Test Suites: {passed + failed + skipped} total")
    terminalreporter.write_line(f"Tests: {passed} passed, {failed} failed, {skipped} skipped, {total} total")
    terminalreporter.write_line(f"Time: {duration:.3f} s")
    terminalreporter.write_line("Ran all test suites.")