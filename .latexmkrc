$out_dir = "build";
$log_file = "build/logs/build.log";
$pdf_mode = 1;
$shell_escape = 1;
$interaction = "nonstopmode";

add_cus_dep('tex', 'pdf', 0, 'tikz2pdf');
sub tikz2pdf {
    system("pdflatex -shell-escape -interaction=nonstopmode -output-directory=build $_[0]");
}
