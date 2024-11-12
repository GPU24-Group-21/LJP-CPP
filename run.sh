prog="./ljp"
infile="Rap_2_LJP.in"

# find any .in file and set it as the input file
if [ -f *.in ]; then
    infile=$(ls *.in)
fi

# Run the program with the given arguments
$prog $infile 0