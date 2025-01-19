# DO NOT CHANGE. THIS FILE MODIFIED LATEXMK TO NOT SPAM THE DIRECTORIES WITH THE AUX FILES.
# KEEP THE AUX FOLDER. IF THOSE FILES ARE DELETED, COMPILATION IS SEVEN TIMES SLOWER (TESTED)

# Enable PDF mode
$pdf_mode = 1;

# Set auxiliary files directory
$aux_dir = ".aux";

# Keep PDF in main directory but aux files in aux directory
$out_dir = ".";

# Use pdflatex with nonstop mode
$pdflatex = 'pdflatex -interaction=nonstopmode';