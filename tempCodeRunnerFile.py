import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import os

class KMeansGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parallel K-Means Clustering")
        self.root.geometry("800x600")

        # Configuration Frame
        config_frame = ttk.LabelFrame(root, text="Configuration", padding="10")
        config_frame.pack(fill="x", padx=10, pady=10)

        # N, D, K, Iters
        ttk.Label(config_frame, text="Points (N):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.entry_n = ttk.Entry(config_frame)
        self.entry_n.insert(0, "1000000")
        self.entry_n.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(config_frame, text="Dimensions (D):").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.entry_d = ttk.Entry(config_frame)
        self.entry_d.insert(0, "100")
        self.entry_d.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(config_frame, text="Clusters (K):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.entry_k = ttk.Entry(config_frame)
        self.entry_k.insert(0, "100")
        self.entry_k.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(config_frame, text="Max Iterations:").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.entry_iters = ttk.Entry(config_frame)
        self.entry_iters.insert(0, "50")
        self.entry_iters.grid(row=1, column=3, padx=5, pady=5)

        # Mode Selection Frame
        mode_frame = ttk.LabelFrame(root, text="Execution Mode", padding="10")
        mode_frame.pack(fill="x", padx=10, pady=5)

        self.mode_var = tk.StringVar(value="Serial")
        ttk.Radiobutton(mode_frame, text="Serial", variable=self.mode_var, value="Serial", command=self.toggle_parallel_options).grid(row=0, column=0, padx=5, pady=5)
        ttk.Radiobutton(mode_frame, text="Parallel (MPI+OpenMP+CUDA)", variable=self.mode_var, value="Parallel", command=self.toggle_parallel_options).grid(row=0, column=1, padx=5, pady=5)

        # Parallel Options
        self.lbl_mpi = ttk.Label(mode_frame, text="MPI Ranks:")
        self.lbl_mpi.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.entry_mpi = ttk.Entry(mode_frame)
        self.entry_mpi.insert(0, "2")
        self.entry_mpi.grid(row=1, column=1, padx=5, pady=5)

        self.lbl_omp = ttk.Label(mode_frame, text="OpenMP Threads/Rank:")
        self.lbl_omp.grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.entry_omp = ttk.Entry(mode_frame)
        self.entry_omp.insert(0, "4")
        self.entry_omp.grid(row=1, column=3, padx=5, pady=5)

        self.toggle_parallel_options()

        # Action Buttons
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill="x", padx=10, pady=5)

        self.btn_compile = ttk.Button(btn_frame, text="Compile", command=self.compile_code)
        self.btn_compile.pack(side="left", padx=5)

        self.btn_run = ttk.Button(btn_frame, text="Run", command=self.run_code)
        self.btn_run.pack(side="left", padx=5)

        # Output Text Area
        output_frame = ttk.LabelFrame(root, text="Output", padding="10")
        output_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.output_text = scrolledtext.ScrolledText(output_frame, state="disabled", bg="black", fg="white", font=("Consolas", 10))
        self.output_text.pack(fill="both", expand=True)

    def write_output(self, text):
        self.output_text.configure(state="normal")
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.configure(state="disabled")

    def toggle_parallel_options(self):
        if self.mode_var.get() == "Parallel":
            self.lbl_mpi.grid()
            self.entry_mpi.grid()
            self.lbl_omp.grid()
            self.entry_omp.grid()
        else:
            self.lbl_mpi.grid_remove()
            self.entry_mpi.grid_remove()
            self.lbl_omp.grid_remove()
            self.entry_omp.grid_remove()

    def run_process_async(self, command, env=None):
        self.btn_run.configure(state="disabled")
        self.btn_compile.configure(state="disabled")
        self.write_output(f"\n> {' '.join(command) if isinstance(command, list) else command}\n")
        
        def task():
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    shell=True
                )
                for line in process.stdout:
                    self.root.after(0, self.write_output, line)
                
                process.wait()
                self.root.after(0, self.write_output, f"\nProcess exited with code {process.returncode}\n")
            except Exception as e:
                self.root.after(0, self.write_output, f"\nError: {str(e)}\n")
            
            self.root.after(0, lambda: self.btn_run.configure(state="normal"))
            self.root.after(0, lambda: self.btn_compile.configure(state="normal"))
            
        threading.Thread(target=task, daemon=True).start()

    def compile_code(self):
        cmd = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", ".\\build.ps1"]
        self.run_process_async(cmd)

    def run_code(self):
        n = self.entry_n.get()
        d = self.entry_d.get()
        k = self.entry_k.get()
        iters = self.entry_iters.get()

        if self.mode_var.get() == "Serial":
            exe = ".\\kmeans_serial.exe"
            if not os.path.exists("kmeans_serial.exe"):
                messagebox.showerror("Error", "kmeans_serial.exe not found. Compile first!")
                return
            cmd = f"{exe} {n} {d} {k} {iters}"
            self.run_process_async(cmd)
        else:
            exe = ".\\kmeans_parallel.exe"
            mpi_ranks = self.entry_mpi.get()
            omp_threads = self.entry_omp.get()
            
            if not os.path.exists("kmeans_parallel.exe"):
                messagebox.showerror("Error", "kmeans_parallel.exe not found. Compile first!")
                return
                
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = omp_threads
            
            cmd = f"mpiexec -n {mpi_ranks} {exe} {n} {d} {k} {iters}"
            self.run_process_async(cmd, env=env)

if __name__ == "__main__":
    root = tk.Tk()
    app = KMeansGUI(root)
    root.mainloop()
