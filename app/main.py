from __future__ import annotations

import argparse
import sys

from PySide6 import QtCore, QtWidgets

from .gui.main_window import MainWindow


def run(smoke_test: bool = False) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("FP-APF UAV Lab")

    window = MainWindow()
    if smoke_test:
        QtCore.QTimer.singleShot(50, app.quit)
        return app.exec()

    window.show()
    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the FP-APF UAV GUI lab.")
    parser.add_argument("--smoke-test", action="store_true", help="Create the app offscreen and exit quickly.")
    args = parser.parse_args()
    return run(smoke_test=args.smoke_test)


if __name__ == "__main__":
    raise SystemExit(main())
