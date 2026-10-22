// Build with tools/build_windows_launcher.ps1. Uses the project's Python so
// training and camera subprocesses keep the same interpreter and dependencies.
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;
using System.Windows.Forms;

[assembly: AssemblyTitle("FSL Studio")]
[assembly: AssemblyDescription("Launcher for the local FSL Studio workspace")]
[assembly: AssemblyProduct("FSL Studio")]
[assembly: AssemblyVersion("1.0.0.0")]

internal static class StudioLauncher
{
    // Quote using Windows command-line rules, including trailing backslashes.
    private static string Quote(string value)
    {
        StringBuilder result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char character in value)
        {
            if (character == '\\')
            {
                slashes++;
                continue;
            }
            if (character == '"')
            {
                result.Append('\\', slashes * 2 + 1);
                result.Append('"');
            }
            else
            {
                result.Append('\\', slashes);
                result.Append(character);
            }
            slashes = 0;
        }
        result.Append('\\', slashes * 2);
        return result.Append('"').ToString();
    }

    private static int Fail(string message)
    {
        MessageBox.Show(message, "FSL Studio", MessageBoxButtons.OK, MessageBoxIcon.Error);
        return 1;
    }

    [STAThread]
    private static int Main(string[] arguments)
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string python = Path.Combine(root, @".venv\Scripts\python.exe");
        string dashboard = Path.Combine(root, "dashboard.py");
        if (!File.Exists(dashboard))
            return Fail("Keep FSL Studio.exe in the trainer folder, beside dashboard.py.\n\n"
                        + "To put it on your desktop, create a shortcut to this executable.");
        if (!File.Exists(python))
            return Fail("The trainer's Python environment is missing.\n\n"
                        + "Open PowerShell in the trainer folder and run:\n"
                        + "py -3.12 -m venv .venv\n"
                        + @".\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-desktop.txt");

        string logPath = null;
        try
        {
            string logDirectory = Path.Combine(root, "outputs", "launcher");
            Directory.CreateDirectory(logDirectory);
            logPath = Path.Combine(logDirectory, DateTime.Now.ToString("yyyyMMdd-HHmmss-fff")
                                   + "-" + Process.GetCurrentProcess().Id + ".log");
            using (StreamWriter log = new StreamWriter(logPath, false, new UTF8Encoding(false)))
            using (Process process = new Process())
            {
                log.AutoFlush = true;
                log.WriteLine("FSL Studio started: " + DateTime.Now.ToString("O"));
                log.WriteLine("Python: " + python);
                StringBuilder command = new StringBuilder("-u " + Quote(dashboard));
                foreach (string argument in arguments)
                    command.Append(" ").Append(Quote(argument));
                process.StartInfo = new ProcessStartInfo(python, command.ToString())
                {
                    WorkingDirectory = root,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    StandardOutputEncoding = Encoding.UTF8,
                    StandardErrorEncoding = Encoding.UTF8
                };
                process.StartInfo.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
                process.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e)
                {
                    if (e.Data != null) { lock (log) { log.WriteLine(e.Data); } }
                };
                process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e)
                {
                    if (e.Data != null) { lock (log) { log.WriteLine(e.Data); } }
                };
                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                process.WaitForExit();
                log.WriteLine("Exit code: " + process.ExitCode);
                if (process.ExitCode != 0)
                    return Fail("FSL Studio stopped with exit code " + process.ExitCode
                                + ".\n\nDetails are saved in:\n" + logPath
                                + "\n\nIf dependencies are missing, open PowerShell in the trainer folder and run:\n"
                                + @".\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-desktop.txt");
            }
            return 0;
        }
        catch (Exception error)
        {
            return Fail("FSL Studio could not start.\n\n" + error.Message
                        + (logPath == null ? "" : "\n\nLog location:\n" + logPath));
        }
    }
}
