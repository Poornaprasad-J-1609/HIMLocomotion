"""Run focused walking unittest checks; --all also runs existing function tests."""
import argparse
import ast
import importlib.util
from pathlib import Path
import unittest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    for base in (root/'src',root/'scripts',root/'tests'):
        for source in base.rglob('*.py'):
            ast.parse(source.read_text(),filename=str(source))
    suite=unittest.TestSuite()
    for p in sorted((root/'tests').glob('test_*.py' if args.all else 'test_native_walking.py')):
        spec=importlib.util.spec_from_file_location(p.stem,p)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        for name,fn in vars(module).items():
            if name.startswith('test_') and callable(fn):
                suite.addTest(unittest.FunctionTestCase(fn,description=p.name+':'+name))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())


if __name__=='__main__': main()
