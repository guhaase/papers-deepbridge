#!/bin/bash

##
## Script to compile NVIDIA Academic Grant Proposal
## Usage: ./compile_proposal.sh
##

echo "=========================================="
echo "Compiling NVIDIA Research Proposal"
echo "=========================================="
echo ""

# Set variables
TEX_FILE="nvidia_research_proposal.tex"
PDF_FILE="nvidia_research_proposal.pdf"
LOG_FILE="compilation.log"

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found. Please install LaTeX (texlive-full or mactex)."
    exit 1
fi

echo "Step 1: First compilation pass..."
pdflatex -interaction=nonstopmode "$TEX_FILE" > "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ First pass completed successfully"
else
    echo "✗ First pass failed. Check $LOG_FILE for errors."
    tail -n 50 "$LOG_FILE"
    exit 1
fi

echo ""
echo "Step 2: Second compilation pass (for TOC and references)..."
pdflatex -interaction=nonstopmode "$TEX_FILE" >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Second pass completed successfully"
else
    echo "✗ Second pass failed. Check $LOG_FILE for errors."
    tail -n 50 "$LOG_FILE"
    exit 1
fi

echo ""
echo "Step 3: Cleaning auxiliary files..."
rm -f *.aux *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz

echo "✓ Cleanup completed"
echo ""
echo "=========================================="
echo "PDF generated successfully: $PDF_FILE"
echo "=========================================="
echo ""

# Check PDF size
if [ -f "$PDF_FILE" ]; then
    FILE_SIZE=$(du -h "$PDF_FILE" | cut -f1)
    PAGE_COUNT=$(pdfinfo "$PDF_FILE" 2>/dev/null | grep "Pages:" | awk '{print $2}')

    echo "File size: $FILE_SIZE"
    if [ -n "$PAGE_COUNT" ]; then
        echo "Page count: $PAGE_COUNT pages"
    fi
    echo ""
    echo "You can now view the proposal:"
    echo "  xdg-open $PDF_FILE    (Linux)"
    echo "  open $PDF_FILE        (Mac)"
else
    echo "ERROR: PDF file was not generated."
    exit 1
fi
