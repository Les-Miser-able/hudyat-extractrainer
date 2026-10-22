"""Build and exercise the real Windows launcher against an isolated workspace."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == 'win32', 'Windows executable launcher')
class WindowsLauncherTests(unittest.TestCase):
    def test_relocated_launcher_preserves_python_workers_arguments_and_logs(self):
        framework = Path(os.environ['WINDIR']) / 'Microsoft.NET'
        if not any((framework / folder / 'v4.0.30319/csc.exe').is_file()
                   for folder in ('Framework64', 'Framework')):
            self.skipTest('.NET Framework C# compiler is unavailable')
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='fsl launcher ') as temporary:
            root = Path(temporary) / 'trainer with spaces ü'
            tools = root / 'tools'
            (tools / 'windows').mkdir(parents=True)
            shutil.copy2(project / 'tools/build_windows_launcher.ps1', tools)
            shutil.copy2(project / 'tools/windows/StudioLauncher.cs', tools / 'windows')
            build = subprocess.run(
                ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                 '-File', str(tools / 'build_windows_launcher.ps1')],
                cwd=temporary, capture_output=True, text=True, errors='replace', timeout=30)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            executable = root / 'FSL Studio.exe'
            binary = executable.read_bytes()
            # A GUI subsystem executable never opens its own terminal window.
            pe_offset = struct.unpack_from('<I', binary, 0x3c)[0]
            self.assertEqual(struct.unpack_from('<H', binary, pe_offset + 24 + 68)[0], 2)

            environment = subprocess.run(
                [sys.executable, '-m', 'venv', '--without-pip', str(root / '.venv')],
                capture_output=True, text=True, errors='replace', timeout=30)
            self.assertEqual(environment.returncode, 0, environment.stdout + environment.stderr)
            (root / 'dashboard.py').write_text('''\
import ctypes, json, pathlib, subprocess, sys
worker = subprocess.run(
    [sys.executable, '-c', 'import sys; print(sys.executable)'],
    capture_output=True, text=True, encoding='utf-8', check=True,
    creationflags=subprocess.CREATE_NO_WINDOW)
pathlib.Path('result.json').write_text(json.dumps({
    'argv': sys.argv[1:], 'cwd': str(pathlib.Path.cwd()),
    'python': sys.executable, 'worker_python': worker.stdout.strip(),
    'console': ctypes.windll.kernel32.GetConsoleWindow(),
}), encoding='utf-8')
for _ in range(200):
    print('stdout ' + 'x' * 1024)
    print('stderr ' + 'x' * 1024, file=sys.stderr)
print('Unicode: 雙手')
print('stderr finished', file=sys.stderr)
''', encoding='utf-8')
            arguments = ['--example', 'two words', '', 'C:\\folder with spaces\\',
                         'a "quoted" value', 'backslash\\"quote', '& echo literal', '雙手']
            launched = subprocess.run([str(executable), *arguments], cwd=temporary,
                                      capture_output=True, timeout=30)
            self.assertEqual(launched.returncode, 0)
            result = json.loads((root / 'result.json').read_text(encoding='utf-8'))
            self.assertEqual(result['argv'], arguments)
            self.assertEqual(Path(result['cwd']), root)
            python = root / '.venv/Scripts/python.exe'
            self.assertEqual(Path(result['python']), python)
            self.assertEqual(Path(result['worker_python']), python)
            self.assertEqual(result['console'], 0)
            logs = list((root / 'outputs/launcher').glob('*.log'))
            self.assertEqual(len(logs), 1)
            log = logs[0].read_text(encoding='utf-8')
            self.assertEqual(log.count('stdout '), 200)
            self.assertEqual(log.count('stderr ' + 'x'), 200)
            self.assertIn('Unicode: 雙手', log)
            self.assertIn('stderr finished', log)
            self.assertIn('Exit code: 0', log)


if __name__ == '__main__':
    unittest.main()
