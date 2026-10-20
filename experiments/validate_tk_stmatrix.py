"""Run existing GEMM contracts against both factored-store library variants."""
from functools import partial
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.tk_mxfp8_gemm import Library,build
from tests import test_tk_mxfp8_gemm as plain
from tests import test_tk_mxfp8_epilogue as quantized
from tests import test_tk_movmatrix_epilogue as movmatrix


def main():
    directory=ROOT/'runs/tk_stmatrix_address'
    plain.OUTPUT=directory/'factored'
    plain.Library=lambda:Library(plain.OUTPUT)
    plain.build=partial(build,output=plain.OUTPUT,factored_stores=True)
    quantized.Library=plain.Library
    movmatrix.OUTPUT=directory/'factored_movm'
    movmatrix.build=partial(build,factored_stores=True)
    class CheckModules:
        def pytest_collection_modifyitems(self,items):
            # Default pytest "prepend" imports these files a second time under
            # top-level names, silently bypassing the library overrides.
            assert all(item.module in [plain,quantized,movmatrix] for item in items)
    raise SystemExit(pytest.main([
        '--import-mode=importlib',
        'tests/test_tk_mxfp8_gemm.py','tests/test_tk_mxfp8_epilogue.py',
        'tests/test_tk_movmatrix_epilogue.py',*sys.argv[1:]],plugins=[CheckModules()]))


if __name__=='__main__':main()
