
from common.modules import tk, ttk, messagebox, filedialog, threading
from common.modules import FigureCanvasTkAgg, Figure
from gui.experiment_runner import ExperimentRunner
from src.utils.logging import ExperimentLogger


class ExperimentLauncher:
    
    """
    Main GUI application for launching experiments
    """
    
    def __init__(self):
        self.runner = ExperimentRunner()
        self.results = None
        self._setup_gui()
        
    def _setup_gui(self):
        """Set up the main GUI interface"""
        
        # create main window
        self.root = tk.Tk()
        self.root.title("Computer Vision Trading - Experiment Launcher")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # create GUI elements
        self._create_experiment_selection(main_frame)
        self._create_control_buttons(main_frame)
        self._create_progress_section(main_frame)
        self._create_results_section(main_frame)
        
        # initialize selections
        self._update_experiment_list()
    
    def _create_experiment_selection(self, parent):
        """Create experiment selection section"""
        # experiment selection frame
        exp_frame = ttk.LabelFrame(parent, text="Experiment Selection", padding="5")
        exp_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        exp_frame.columnconfigure(1, weight=1)
        
        # experiment label and combobox
        ttk.Label(exp_frame, text="Experiment:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5)
        )
        self.experiment_var = tk.StringVar()
        self.experiment_combo = ttk.Combobox(
            exp_frame, textvariable=self.experiment_var, state="readonly", width=30
        )
        self.experiment_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # refresh button
        ttk.Button(exp_frame, text="Refresh", 
                  command=self._update_experiment_list).grid(row=0, column=2
        )
    
    def _create_control_buttons(self, parent):
        """Create control buttons section"""
        # control buttons frame
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # run experiment button
        self.run_button = ttk.Button(control_frame, text="Run Experiment", 
            command=self._run_experiment, style="Accent.TButton"
        )
        self.run_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # clear results button
        ttk.Button(control_frame, text="Clear Results", 
                  command=self._clear_results).pack(side=tk.LEFT, padx=(0, 10))
        
        # exit button
        ttk.Button(control_frame, text="Exit", 
                  command=self.root.quit).pack(side=tk.LEFT)
    
    def _create_progress_section(self, parent):
        """Create progress section"""
        # progress frame
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="5")
        progress_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        # progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame,
            variable=self.progress_var, maximum=100, length=300
        )
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.grid(row=1, column=0, sticky=tk.W)
    
    def _create_results_section(self, parent):
        """Create results display section."""
        # results frame
        results_frame = ttk.LabelFrame(parent, text="Results", padding="5")
        results_frame.grid(
            row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10)
        )
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(5, weight=1)
        
        # create notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # summary tab
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        
        # summary text widget
        self.summary_text = tk.Text(self.summary_frame, wrap=tk.WORD, height=10)
        summary_scrollbar = ttk.Scrollbar(self.summary_frame, orient=tk.VERTICAL, 
                                        command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)
        
        self.summary_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        summary_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.summary_frame.columnconfigure(0, weight=1)
        self.summary_frame.rowconfigure(0, weight=1)
        
        # plots tab
        self.plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plots_frame, text="Plots")
        
        # matplotlib figure for plots
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, self.plots_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.plots_frame.columnconfigure(0, weight=1)
        self.plots_frame.rowconfigure(0, weight=1)
    
    def _update_experiment_list(self):
        """Update the list of available experiments."""
        try:
            experiments = self.runner.available_experiments
            self.experiment_combo['values'] = experiments
            if experiments:
                self.experiment_var.set(experiments[0])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update experiment list: {e}")
    
    def _run_experiment(self):
        """Run the selected experiment in a separate thread"""
        
        # validate selections
        if not self.experiment_var.get():
            messagebox.showerror("Error", "Please select an experiment")
            return
        
        # disable run button and start progress
        self.run_button.config(state="disabled")
        self.progress_var.set(0)
        self.status_var.set("Starting experiment...")
        
        # initialize an experiment logger
        logger = ExperimentLogger(self.experiment_var.get())
        logger.info("ExperimentLauncher GUI initialized")
        
        # run experiment in separate thread
        self._run_experiment_thread()
        
        thread = threading.Thread(target=self._run_experiment_thread)
        thread.daemon = True
        thread.start()
    
    def _run_experiment_thread(self):
        """Run experiment in separate thread"""
        try:
            # update progress
            self.root.after(0, lambda: self.progress_var.set(25))
            self.root.after(0, lambda: self.status_var.set("Loading configuration..."))
            
            # run the experiment
            self.root.after(0, lambda: self.progress_var.set(50))
            self.root.after(0, lambda: self.status_var.set("Running experiment..."))
            
            results = self.runner.run_experiment(
                experiment_name=self.experiment_var.get()
            )
            
            # update progress
            self.root.after(0, lambda: self.progress_var.set(75))
            self.root.after(0, lambda: self.status_var.set("Processing results..."))
            
            # store results and update display
            self.results = results
            self.root.after(0, self._update_results_display)
            
            # complete progress
            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, lambda: self.status_var.set("Experiment completed"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(f"Error. Experiment failed: {e}"))
            self.root.after(0, lambda: self.status_var.set("Experiment failed"))
        
        finally: # re-enable run button
            self.root.after(0, lambda: self.run_button.config(state="normal"))
    
    def _update_results_display(self):
        """Update the results display with experiment results."""
        if not self.results:
            return
        
        try:
            # Update summary tab
            summary = self.runner.get_experiment_summary(self.results)
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(1.0, summary)
            
            # Update plots tab
            self._create_results_plots()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update results display: {e}")
    
    def _create_results_plots(self):
        """Create plots for the results."""
        try:
            # Clear previous plots
            self.fig.clear()
            
            # Create a simple example plot
            ax = self.fig.add_subplot(111)
            
            # Example data - in a real implementation, this would use actual results
            if self.results and self.results.get('success'):
                # Create a simple success indicator plot
                categories = ['Success', 'Duration (s)', 'Config Loaded']
                values = [1, self.results['duration_seconds'], 1]
                colors = ['green', 'blue', 'orange']
                
                bars = ax.bar(categories, values, color=colors)
                ax.set_title(f"Experiment Results: {self.results['experiment_name']}")
                ax.set_ylabel("Value")
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.2f}', ha='center', va='bottom')
            else:
                # Show error plot
                ax.text(0.5, 0.5, 'Experiment Failed\nNo data to display', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, color='red')
                ax.set_title("Experiment Results: Failed")
            
            # Refresh canvas
            self.canvas.draw()
            
        except Exception as e:
            return e
    
    def _clear_results(self):
        """Clear the results display"""
        self.results = None
        self.summary_text.delete(1.0, tk.END)
        self.fig.clear()
        self.canvas.draw()
        self.status_var.set("Results cleared")
        self.progress_var.set(0)
    
    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except Exception as e:
            messagebox.showerror("Error", f"GUI error: {e}")
            